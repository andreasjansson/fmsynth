# Prediction interface for Cog ⚙️
# Reference: https://github.com/replicate/cog/blob/main/docs/python.md

import os
import tempfile
import subprocess
from cog import BasePredictor, Path, Input
import torch
from scipy.io import wavfile
import numpy as np
import librosa

from synth import Synth


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.synth = Synth(torch.device("cuda"))

    def predict(
        self,
        audio: Path = Input(description="Audio input file, max 5 seconds"),
        minimize_num_freqs: bool = Input(
            description="Minimize the number of active carrier and modulator frequencies",
            default=False,
        ),
        carrier_stereo_detune: float = Input(
            description="Carrier frequency stereo detune", default=0.005
        ),
        mod_stereo_detune: float = Input(
            description="Modulator frequency stereo detune", default=0.01
        ),
        time_stretch: float = Input(
            description="Time stretch factor", ge=0.1, le=5, default=1
        ),
        learning_rate: float = Input(description="ADAM learning rate", default=0.01),
        n_iter: int = Input(
            description="Number of optimization iterations",
            ge=100,
            le=5000,
            default=1000,
        ),
        seed: int = Input(description="Random seed, -1 for random", default=-1),
    ) -> Path:
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
