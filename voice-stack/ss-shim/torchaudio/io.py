"""File I/O backed by soundfile (libsndfile)."""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["load", "save"]


def load(
    filepath,
    frame_offset: int = 0,
    num_frames: int = -1,
    normalize: bool = True,
    channels_first: bool = True,
):
    """Return ``(tensor, sample_rate)``.

    Always yields float32; ``normalize=False`` also yields float32 because
    libsndfile hands back normalized floats for float formats and integers
    otherwise. The pipeline only ever feeds float audio downstream, so keeping
    a single dtype is both simpler and safer.
    """
    import soundfile as sf

    start = int(frame_offset) if frame_offset else 0
    frames = int(num_frames) if num_frames and num_frames > 0 else -1

    data, sample_rate = sf.read(
        str(filepath),
        start=start,
        frames=frames,
        dtype="float32",
        always_2d=True,
    )
    tensor = torch.from_numpy(np.ascontiguousarray(data))
    if channels_first:
        tensor = tensor.transpose(0, 1).contiguous()
    return tensor, sample_rate


def save(filepath, src, sample_rate: int, channels_first: bool = True, **kwargs):
    """Write a tensor to an audio file."""
    import soundfile as sf

    tensor = src
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().to("cpu").numpy()

    array = np.asarray(tensor, dtype=np.float32)
    if channels_first and array.ndim == 2:
        array = array.T
    if array.ndim == 1:
        array = array[:, None]

    sf.write(str(filepath), np.ascontiguousarray(array), int(sample_rate))
    return None
