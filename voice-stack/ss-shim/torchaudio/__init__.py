"""Minimal torchaudio compatibility shim for NVIDIA Jetson (L4T / aarch64).

Why this exists
---------------
The aarch64 ``torchaudio`` wheel published on PyPI is built against upstream
PyTorch and is ABI-incompatible with the NVIDIA Jetson ``torch`` build:

    OSError: libtorchaudio.so: undefined symbol:
        _ZNK5torch8autograd4Node4nameEv   (torch::autograd::Node::name() const)

There is no matching torchaudio wheel available for JetPack 5.x / CUDA 11.4
(NVIDIA's redist only ships ``torch`` there), and building torchaudio from
source on the device is impractical.

This shim provides the small API surface that the HuggingFace
``speech-to-speech`` pipeline and ``silero-vad`` actually touch, implemented on
top of ``soxr`` (high quality resampling) and ``soundfile`` (file I/O), both of
which work natively on this device.

Load it ahead of site-packages via ``PYTHONPATH`` so the real (broken) package
is never imported.
"""

from __future__ import annotations

__version__ = "2.2.0+jetson.shim"

from . import functional, transforms, sox_effects  # noqa: E402,F401
from . import io  # noqa: E402,F401

__all__ = [
    "__version__",
    "functional",
    "transforms",
    "sox_effects",
    "io",
    "load",
    "save",
    "info",
    "list_audio_backends",
    "set_audio_backend",
]


def load(
    filepath,
    frame_offset: int = 0,
    num_frames: int = -1,
    normalize: bool = True,
    channels_first: bool = True,
    format=None,  # noqa: A002 - kept for signature compatibility
    backend=None,
    **kwargs,
):
    """Load an audio file into a ``(channels, frames)`` float tensor."""
    return io.load(
        filepath,
        frame_offset=frame_offset,
        num_frames=num_frames,
        normalize=normalize,
        channels_first=channels_first,
    )


def save(filepath, src, sample_rate: int, channels_first: bool = True, **kwargs):
    """Save a ``(channels, frames)`` tensor to disk."""
    return io.save(filepath, src, sample_rate, channels_first=channels_first)


def info(filepath, format=None, backend=None):  # noqa: A002
    raise NotImplementedError(
        "torchaudio.info() is not provided by the jetson shim; use soundfile.info() instead"
    )


def list_audio_backends():
    return ["soundfile"]


def set_audio_backend(backend):
    """Deprecated upstream; kept as a no-op so old call sites keep working."""
    return None
