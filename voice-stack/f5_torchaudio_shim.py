"""Fake torchaudio shim.

The ComfyUI NVIDIA torch 2.2.0 on this box is ABI-incompatible with PyPI
torchaudio (both 2.2.0 and 2.11.0 fail with an undefined-symbol error), but
f5_tts + vocos import torchaudio at module load AND use torchaudio's
MelSpectrogram as an actual nn.Module inside the F5 conditioning path.

This module registers a minimal fake `torchaudio` package:
  - top-level load/save (soundfile)
  - transforms.Resample / MelSpectrogram (nn.Module) / Spectrogram
  - functional.functional._hz_to_mel / _mel_to_hz (vocos.heads import)
Real audio I/O / mel computation is done with soundfile + librosa.
"""
import sys
import types
import math

import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
import librosa


# --------------------------------------------------------------------------
# top-level load / save
# --------------------------------------------------------------------------
def load(path, frame_offset=0, num_frames=-1, normalize=True, channels_first=True,
         format=None):
    data, sr = sf.read(path)
    if data.ndim == 1:
        data = data[None, :]
    elif data.ndim == 2 and data.shape[0] > 2:
        data = data.T
    return torch.from_numpy(data).float(), sr


def save(path, waveform, sample_rate):
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().cpu().numpy()
    if waveform.ndim == 1:
        waveform = waveform[:, None]
    sf.write(path, waveform, sample_rate)


# --------------------------------------------------------------------------
# transforms
# --------------------------------------------------------------------------
class Resample:
    def __init__(self, orig_freq, new_freq, *args, **kwargs):
        self.orig = orig_freq
        self.new = new_freq

    def __call__(self, waveform):
        x = waveform.detach().cpu().numpy() if isinstance(waveform, torch.Tensor) else waveform
        if x.ndim == 1:
            x = x[None, :]
        elif x.ndim == 2 and x.shape[0] > 2:
            x = x.T
        y = np.stack([librosa.resample(x[c], orig_sr=self.orig, target_sr=self.new)
                      for c in range(x.shape[0])])
        t = torch.from_numpy(y).float()
        return t if t.dim() == 2 else t.unsqueeze(0)


class MelSpectrogram(nn.Module):
    """nn.Module drop-in for torchaudio.transforms.MelSpectrogram.

    Returns magnitude mel (power applied) so the caller's
    ``.clamp(min=1e-5).log()`` matches torchaudio behaviour.
    Computed on CPU (torch.stft is robust there) and returned on the input
    device so it composes with ``module.to(device)``.
    """

    def __init__(self, sample_rate=16000, n_fft=400, win_length=None, hop_length=None,
                 f_min=0.0, f_max=None, n_mels=128, norm=None, mel_scale="htk",
                 center=True, power=2.0, normalized=False, **kwargs):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sample_rate // 2
        self.n_mels = n_mels
        self.norm = norm
        self.mel_scale = mel_scale
        self.center = center
        self.power = power
        self.normalized = normalized
        fb = librosa.filters.mel(
            sr=sample_rate, n_fft=n_fft, n_mels=n_mels,
            fmin=f_min, fmax=self.f_max, htk=mel_scale == "htk", norm=norm,
        )
        self.register_buffer("fb", torch.from_numpy(fb).float())

    def forward(self, waveform):
        dev = waveform.device if isinstance(waveform, torch.Tensor) else torch.device("cpu")
        if isinstance(waveform, torch.Tensor):
            x = waveform.detach().to("cpu").float()
        else:
            x = torch.from_numpy(np.asarray(waveform, dtype=np.float32)).float()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        mels = []
        for c in range(x.shape[0]):
            window = torch.hann_window(self.win_length, device=x.device)
            S = torch.stft(
                x[c], n_fft=self.n_fft, hop_length=self.hop_length,
                win_length=self.win_length, center=self.center,
                window=window, return_complex=True,
            ).abs() ** self.power
            fb = self.fb.to(S.device)
            mels.append(torch.matmul(fb, S))
        return torch.stack(mels).to(dev)


class Spectrogram:
    def __init__(self, n_fft=400, win_length=None, hop_length=None, power=2.0, **kwargs):
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        self.power = power

    def __call__(self, waveform):
        x = waveform.detach().cpu().numpy() if isinstance(waveform, torch.Tensor) else waveform
        if x.ndim == 1:
            x = x[None, :]
        specs = []
        for c in range(x.shape[0]):
            S = np.abs(librosa.stft(x[c], n_fft=self.n_fft, hop_length=self.hop_length,
                                    win_length=self.win_length)) ** self.power
            specs.append(S)
        out = np.stack(specs)
        return torch.from_numpy(out).float()


# --------------------------------------------------------------------------
# functional.functional  (_hz_to_mel / _mel_to_hz used by vocos.heads)
# --------------------------------------------------------------------------
def _hz_to_mel(freq, mel_scale="htk"):
    if mel_scale == "htk":
        return 2595.0 * math.log10(1.0 + freq / 700.0)
    elif mel_scale == "slaney":
        f_min, f_sp = 0.0, 200.0 / 3
        freq_arr = np.asarray(freq, dtype=float)
        mels = (freq_arr - f_min) / f_sp
        min_log_hz, min_log_mel, logstep = 1000.0, (1000.0 - f_min) / f_sp, math.log(6.4) / 27.0
        mask = freq_arr >= min_log_hz
        mels[mask] = min_log_mel + np.log(freq_arr[mask] / min_log_hz) / logstep
        return mels if isinstance(freq, (np.ndarray, list, torch.Tensor)) else float(mels)
    raise ValueError('mel_scale should be either "htk" or "slaney".')


def _mel_to_hz(mels, mel_scale="htk"):
    if mel_scale == "htk":
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    elif mel_scale == "slaney":
        f_min, f_sp = 0.0, 200.0 / 3
        mels_arr = np.asarray(mels, dtype=float)
        freqs = f_min + f_sp * mels_arr
        min_log_hz, min_log_mel, logstep = 1000.0, (1000.0 - f_min) / f_sp, math.log(6.4) / 27.0
        mask = mels_arr >= min_log_mel
        freqs[mask] = min_log_hz * np.exp(logstep * 27.0 * (mels_arr[mask] - min_log_mel))
        return freqs if isinstance(mels, (np.ndarray, list, torch.Tensor)) else float(freqs)
    raise ValueError('mel_scale should be either "htk" or "slaney".')


# --------------------------------------------------------------------------
# assemble the fake package tree
# --------------------------------------------------------------------------
_this = sys.modules[__name__]

_funcfunc = types.ModuleType("torchaudio.functional.functional")
_funcfunc._hz_to_mel = _hz_to_mel
_funcfunc._mel_to_hz = _mel_to_hz

_functional = types.ModuleType("torchaudio.functional")
_functional.functional = _funcfunc

_transforms = types.ModuleType("torchaudio.transforms")
_transforms.Resample = Resample
_transforms.MelSpectrogram = MelSpectrogram
_transforms.Spectrogram = Spectrogram

_this.functional = _functional
_this.transforms = _transforms
_this.load = load
_this.save = save

sys.modules["torchaudio"] = _this
sys.modules["torchaudio.functional"] = _functional
sys.modules["torchaudio.functional.functional"] = _funcfunc
sys.modules["torchaudio.transforms"] = _transforms
