"""Functional audio ops backed by soxr."""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["resample"]


def _to_numpy(waveform):
    if isinstance(waveform, torch.Tensor):
        return waveform.detach().to("cpu").numpy(), True
    return np.asarray(waveform), False


def resample(
    waveform,
    orig_freq: int,
    new_freq: int,
    lowpass_filter_width: int = 6,
    rolloff: float = 0.99,
    resampling_method: str = "sinc_interp_hann",
    beta: float | None = None,
    *,
    dtype=None,
):
    """Resample a waveform along its last dimension.

    Signature mirrors ``torchaudio.functional.resample`` for the arguments used
    by the speech-to-speech pipeline. The actual filtering is delegated to soxr
    (VHQ quality), which is a well-behaved, dependency-light resampler.
    """
    if orig_freq == new_freq:
        if dtype is not None and isinstance(waveform, torch.Tensor):
            return waveform.to(dtype)
        return waveform

    import soxr

    array, was_tensor = _to_numpy(waveform)
    device = waveform.device if was_tensor else None
    orig_dtype = waveform.dtype if was_tensor else None

    arr = np.ascontiguousarray(array, dtype=np.float32)
    out = soxr.resample(arr, orig_freq, new_freq, quality="VHQ")
    out = np.ascontiguousarray(out, dtype=np.float32)

    if was_tensor:
        target_dtype = dtype if dtype is not None else orig_dtype
        return torch.from_numpy(out).to(device=device, dtype=target_dtype)
    return out


def amplitude_to_DB(x, multiplier, amin, db_multiplier, top_db=None, **kwargs):
    raise NotImplementedError("amplitude_to_DB is not provided by the jetson shim")


def spectrogram(*args, **kwargs):
    raise NotImplementedError(
        "spectrogram is not provided by the jetson shim; use torch.stft directly"
    )
