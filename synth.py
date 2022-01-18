import torch
from torch import nn
import torch.nn.functional as F
import librosa
import numpy as np


class Synth:
    def __init__(self, device):
        self.device = device

    def fit(
        self,
        target,
        sr,
        minimize_num_freqs=False,
        carrier_stereo_detune=0.0,
        mod_stereo_detune=0.0,
        time_stretch=1,
        learning_rate=0.01,
        n_iter=1000,
        out_sr=44100,
    ):
        if len(target.shape) == 2:
            target = target[:, 0]
        target = torch.tensor(target).to(self.device).to(torch.float32)
        target -= torch.mean(target)
        target /= torch.max(torch.abs(target))
        dur = len(target) / sr
        if dur > 5:
            raise ValueError("Duration must be less than 5 seconds")

        t = torch.linspace(0, dur, len(target)).to(self.device)

        net = FMNet().to(self.device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        for i in range(n_iter):
            optimizer.zero_grad()
            out = net(t)
            carrier_spiky = torch.sum(net.carrier_weights_ ** 2)
            mod_spiky = torch.sum(net.mod_weights_ ** 2)
            sig_loss = loss_fn(out, target)
            if minimize_num_freqs:
                loss = sig_loss * 100 - carrier_spiky - mod_spiky
            else:
                loss = sig_loss
            loss.backward()
            if i % 100 == 0:
                print(
                    f"i: {i}, sig_loss: {sig_loss.item():.3f}, carrier_spiky: {carrier_spiky.item():.3f}, mod_spiky: {mod_spiky.item():.3f}"
                )
            optimizer.step()

        out = self.make_output(
            net, dur * time_stretch, out_sr, carrier_stereo_detune, mod_stereo_detune
        )

        del net

        return out

    def make_output(self, net, dur, sr, carrier_stereo_detune, mod_stereo_detune):
        t2 = torch.linspace(0, dur, int(dur * sr)).to(self.device)
        carrier_fq = net.carrier_fq.data
        mod_fq = net.mod_fq.data

        net.carrier_fq.data = carrier_fq * (1 + carrier_stereo_detune)
        net.mod_fq.data = mod_fq * (1 + mod_stereo_detune)
        left = net(t2)
        net.carrier_fq.data = carrier_fq / (1 + carrier_stereo_detune)
        net.mod_fq.data = mod_fq / (1 + mod_stereo_detune)
        right = net(t2)

        out = torch.vstack([left, right]).T.cpu().detach().numpy()

        net.carrier_fq.data = carrier_fq
        net.mod_fq.data = mod_fq

        # remove clicks
        n_fade = 100
        out[:n_fade] *= np.repeat(np.linspace(0, 1, n_fade).reshape([-1, 1]), 2, axis=1)
        out[-n_fade:] *= np.repeat(
            np.linspace(1, 0, n_fade).reshape([-1, 1]), 2, axis=1
        )

        # normalize
        out /= np.max(out)

        return out


class Envelope(nn.Module):
    def __init__(self, n_freqs, min_slope=-2.0, max_slope=8.0):
        super().__init__()

        self.slope = nn.Parameter(torch.rand(n_freqs) - 0.5)
        self.offset = nn.Parameter(torch.rand(n_freqs) - 0.5)
        self.min_slope = min_slope
        self.max_slope = max_slope

    def forward(self, t):
        slope = 2.0 ** (
            torch.sigmoid(self.slope) * (self.max_slope - self.min_slope)
            + self.min_slope
        )
        offset = torch.tanh(self.offset) / 2

        t = t / t.max() - 0.5
        bell = 1 / torch.sqrt(
            (
                1
                + (
                    ((t.reshape([-1, 1]) @ slope.reshape([1, -1])) + (slope * offset))
                    ** 2
                )
            )
        )
        return bell


class FMNet(nn.Module):
    def __init__(self, fq_grad=True):
        super().__init__()
        carrier_hz = librosa.midi_to_hz(np.arange(1, 128))
        self.carrier_fq = nn.Parameter(
            torch.tensor(carrier_hz).to(torch.float32), requires_grad=fq_grad
        )
        self.carrier_weight = nn.Parameter(torch.rand(self.carrier_fq.shape) - 0.5)
        self.carrier_env = Envelope(n_freqs=self.carrier_fq.shape)

        mod_hz = librosa.midi_to_hz(np.arange(1, 128))
        self.mod_fq = nn.Parameter(
            torch.tensor(mod_hz).to(torch.float32), requires_grad=fq_grad
        )
        self.mod_weight = nn.Parameter(torch.rand(self.mod_fq.shape) - 0.5)
        self.mod_env = Envelope(n_freqs=self.mod_fq.shape)

        self.phase_offset = nn.Parameter(torch.tensor([0.0]))

    def forward(self, t):
        # self.phase_offset_ = 2 * np.pi * torch.sigmoid(self.phase_offset)
        self.phase_offset_ = 2 * np.pi * torch.sigmoid(self.phase_offset)

        self.mod_phases_ = (
            2 * np.pi * self.mod_fq.reshape([-1, 1]) @ t.reshape([1, -1])
            + self.phase_offset_
        )
        self.mod_waves_ = torch.cos(self.mod_phases_)
        self.mod_weights_ = F.softmax(self.mod_weight)
        self.mod_amps_ = self.mod_env(t) * self.mod_weights_
        # self.mod_amps_ = self.mod_weights_
        self.mods_ = (self.mod_waves_.T * self.mod_amps_).T
        self.mod_ = torch.sum(self.mods_, axis=0)

        self.carrier_phases_ = (
            2 * np.pi * self.carrier_fq.reshape([-1, 1]) @ t.reshape([1, -1])
            + self.mod_
            + self.phase_offset_
        )
        self.carrier_waves_ = torch.sin(self.carrier_phases_)
        self.carrier_weights_ = F.softmax(self.carrier_weight)
        self.carrier_amps_ = self.carrier_env(t) * self.carrier_weights_
        # self.carrier_amps_ = self.carrier_weights_
        self.carriers_ = (self.carrier_waves_.T * self.carrier_amps_).T
        self.carrier_ = torch.sum(self.carriers_, axis=0)

        return self.carrier_
