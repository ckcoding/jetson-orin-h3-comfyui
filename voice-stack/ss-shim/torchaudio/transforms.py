"""Transform classes backed by :mod:`torchaudio.functional`."""

from __future__ import annotations

import torch

from . import functional

__all__ = ["Resample"]


class Resample(torch.nn.Module):
    """Resample a waveform from ``orig_freq`` to ``new_freq``."""

    def __init__(
        self,
        orig_freq: int = 16000,
        new_freq: int = 16000,
        lowpass_filter_width: int = 6,
        rolloff: float = 0.99,
        resampling_method: str = "sinc_interp_hann",
        beta: float | None = None,
        **kwargs,
    ):
        super().__init__()
        self.orig_freq = int(orig_freq)
        self.new_freq = int(new_freq)
        self.lowpass_filter_width = lowpass_filter_width
        self.rolloff = rolloff
        self.resampling_method = resampling_method
        self.beta = beta

    def forward(self, waveform):
        return functional.resample(
            waveform,
            self.orig_freq,
            self.new_freq,
            lowpass_filter_width=self.lowpass_filter_width,
            rolloff=self.rolloff,
            resampling_method=self.resampling_method,
            beta=self.beta,
        )
