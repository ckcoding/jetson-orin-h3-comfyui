"""Minimal pure-torch torchaudio shim for the custom torch-2.2.0 ARM64 build.

The official torchaudio 2.2.0 wheel ships a compiled extension that is
ABI-incompatible with the locally-built torch 2.2.0 on this device
(``undefined symbol: torch::autograd::Node::name()`` at CDLL load time).
This shim re-implements the small API surface used by ComfyUI on this box:

  * functional.resample          (windowed-sinc, hann / kaiser)
  * functional.bass_biquad       (RBJ low shelf via scipy.signal.lfilter)
  * functional.equalizer_biquad  (RBJ peaking EQ)
  * functional.treble_biquad     (RBJ high shelf)
  * transforms.MelSpectrogram    (torch.stft + slaney/htk mel filterbank)
  * load / save / info           (backed by soundfile)

Version string keeps the 2.2.0 prefix so version checks keep passing.
"""

__version__ = "2.2.0+shim.puretorch"

from . import functional
from . import transforms
from .functional import (
    bass_biquad,
    biquad,
    equalizer_biquad,
    highpass_biquad,
    lowpass_biquad,
    resample,
    treble_biquad,
)
from .transforms import AmplitudeToDB, MelSpectrogram, Spectrogram

__all__ = [
    "__version__",
    "functional",
    "transforms",
    "resample",
    "biquad",
    "bass_biquad",
    "treble_biquad",
    "equalizer_biquad",
    "lowpass_biquad",
    "highpass_biquad",
    "Spectrogram",
    "MelSpectrogram",
    "AmplitudeToDB",
    "load",
    "save",
    "info",
]


def load(filepath, *args, **kwargs):
    """Load audio via soundfile. Returns (Tensor[C, T], sample_rate)."""
    import soundfile as sf
    import torch

    data, sr = sf.read(filepath, dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), int(sr)


def save(filepath, src, sample_rate, *args, **kwargs):
    """Save a tensor (channels-first, (*, T)) to file via soundfile."""
    import numpy as np
    import soundfile as sf

    if isinstance(src, torch.Tensor):  # noqa: F821 - torch imported lazily below
        pass
    try:
        import torch

        if isinstance(src, torch.Tensor):
            arr = src.detach().cpu().numpy()
        else:
            arr = np.asarray(src)
    except ImportError:  # pragma: no cover
        arr = np.asarray(src)
    if arr.ndim == 1:
        sf.write(filepath, arr, int(sample_rate))
    else:
        if arr.shape[0] < arr.shape[1]:
            arr = arr.T  # (C, T) -> (T, C)
        sf.write(filepath, arr, int(sample_rate))
    return None


class _AudioInfo(object):
    def __init__(self, sample_rate, num_frames, num_channels):
        self.sample_rate = sample_rate
        self.num_frames = num_frames
        self.num_channels = num_channels

    def __repr__(self):
        return "AudioFileInfo(sample_rate=%s, num_frames=%s, num_channels=%s)" % (
            self.sample_rate,
            self.num_frames,
            self.num_channels,
        )


def info(filepath):
    import soundfile as sf

    i = sf.info(filepath)
    return _AudioInfo(int(i.samplerate), int(i.frames), int(i.channels))
