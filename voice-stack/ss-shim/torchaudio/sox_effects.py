"""Very small subset of ``torchaudio.sox_effects``.

silero-vad imports this module at module scope but only reaches it when a caller
explicitly passes sox effects. The pipeline never does, so unsupported effects
degrade to a warning plus a plain file load rather than an exception.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch

from . import functional, io

__all__ = ["apply_effects_file", "apply_effects_tensor", "effect_names"]


def effect_names():
    return ["rate", "trim"]


def _apply(effects, tensor, sample_rate):
    """Apply the tiny effect subset we support; ignore the rest with a warning."""
    for effect in effects or []:
        name = str(effect[0]).lower() if effect else ""
        if name == "rate":
            new_freq = int(effect[1])
            tensor = functional.resample(tensor, sample_rate, new_freq)
            sample_rate = new_freq
        elif name == "trim":
            # trim only makes sense on a whole file; not attempted here
            warnings.warn(
                "torchaudio shim: 'trim' effect is not implemented; ignored",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            warnings.warn(
                f"torchaudio shim: sox effect {name!r} is not implemented; ignored",
                RuntimeWarning,
                stacklevel=2,
            )
    return tensor, sample_rate


def apply_effects_file(path, effects=None, normalize: bool = True, channels_first: bool = True):
    tensor, sample_rate = io.load(
        path, normalize=normalize, channels_first=channels_first
    )
    tensor, sample_rate = _apply(effects, tensor, sample_rate)
    return tensor, sample_rate


def apply_effects_tensor(waveform, sample_rate: int, effects=None, channels_first: bool = True):
    tensor = waveform
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.from_numpy(np.asarray(tensor))
    if not channels_first:
        tensor = tensor.transpose(0, 1)
    tensor, sample_rate = _apply(effects, tensor, sample_rate)
    if not channels_first:
        tensor = tensor.transpose(0, 1)
    return tensor, sample_rate
