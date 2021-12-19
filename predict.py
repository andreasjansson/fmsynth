# Prediction interface for Cog ⚙️
# Reference: https://github.com/replicate/cog/blob/main/docs/python.md

import os
from pathlib import Path
import tempfile
import subprocess
import cog
import torch
from scipy.io import wavfile
import numpy as np
import librosa

from synth import Synth


class Predictor(cog.Predictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.synth = Synth(torch.device("cuda"))

    @cog.input("audio", type=cog.Path, help="Audio input file, max 5 seconds")
    @cog.input(
        "minimize_num_freqs",
        type=bool,
        default=False,
        help="Minimize the number of active carrier and modulator frequencies",
    )
    @cog.input(
        "carrier_stereo_detune",
        type=float,
        default=0.005,
        help="Carrier frequency stereo detune",
    )
    @cog.input(
        "mod_stereo_detune",
        type=float,
        default=0.01,
        help="Modulator frequency stereo detune",
    )
    @cog.input(
        "time_stretch",
        type=float,
        min=0.1,
        max=5,
        default=1,
        help="Time stretch factor",
    )
    @cog.input(
        "learning_rate",
        type=float,
        min=0.001,
        max=0.1,
        default=0.01,
        help="ADAM learning rate",
    )
    @cog.input(
        "n_iter",
        type=int,
        min=100,
        max=5000,
        default=1000,
        help="Number of optimization iterations",
    )
    @cog.input("seed", type=int, default=-1, help="Random seed, -1 for random")
    def predict(
        self,
        audio,
        minimize_num_freqs,
        carrier_stereo_detune,
        mod_stereo_detune,
        time_stretch,
        learning_rate,
        n_iter,
        seed,
    ):
        """Run a single prediction on the model"""
        if seed < 0:
            seed = int.from_bytes(os.urandom(2), "big")
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"Prediction seed: {seed}")

        target, sr = librosa.load(str(audio))
        out_sr = 44100
        out = self.synth.fit(
            target=target,
            sr=sr,
            minimize_num_freqs=minimize_num_freqs,
            carrier_stereo_detune=carrier_stereo_detune,
            mod_stereo_detune=mod_stereo_detune,
            time_stretch=time_stretch,
            learning_rate=learning_rate,
            n_iter=n_iter,
            out_sr=out_sr,
        )
        out_dir = Path(tempfile.mkdtemp())
        wav_path = out_dir / "out.wav"
        mp3_path = out_dir / "out.mp3"
        wavfile.write(str(wav_path), out_sr, out)
        try:
            subprocess.check_output(
                [
                    "ffmpeg",
                    "-loglevel",
                    "error",
                    "-i",
                    str(wav_path),
                    "-af",
                    "silenceremove=1:0:-50dB,aformat=dblp,areverse,silenceremove=1:0:-50dB,aformat=dblp,areverse",  # strip silence
                    str(mp3_path),
                ],
            )
            return mp3_path
        finally:
            wav_path.unlink()
