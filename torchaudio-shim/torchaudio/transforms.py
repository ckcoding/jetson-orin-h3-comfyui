"""Pure-torch implementations of torchaudio.transforms (MelSpectrogram & co)."""

import math

import torch

__all__ = ["Spectrogram", "MelSpectrogram", "AmplitudeToDB"]


def _hz_to_mel(freq, mel_scale="htk"):
    freq = torch.as_tensor(freq, dtype=torch.float64)
    if mel_scale == "htk":
        return 2595.0 * torch.log10(1.0 + freq / 700.0)
    # slaney
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    return torch.where(
        freq < min_log_hz,
        freq / f_sp,
        min_log_mel + torch.log(freq / min_log_hz) / logstep,
    )


def _mel_to_hz(mel, mel_scale="htk"):
    mel = torch.as_tensor(mel, dtype=torch.float64)
    if mel_scale == "htk":
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    # slaney
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    return torch.where(
        mel < min_log_mel,
        mel * f_sp,
        min_log_hz * torch.exp(logstep * (mel - min_log_mel)),
    )


def _mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max, norm, mel_scale):
    n_freqs = n_fft // 2 + 1
    all_freqs = torch.linspace(0, float(sample_rate) / 2.0, n_freqs, dtype=torch.float64)
    m_min = _hz_to_mel(float(f_min), mel_scale)
    m_max = _hz_to_mel(float(f_max) if f_max is not None else float(sample_rate) / 2.0, mel_scale)
    m_pts = torch.linspace(m_min.item(), m_max.item(), n_mels + 2, dtype=torch.float64)
    f_pts = _mel_to_hz(m_pts, mel_scale)

    fb = torch.zeros(n_mels, n_freqs, dtype=torch.float64)
    for k in range(n_mels):
        f_left, f_center, f_right = f_pts[k], f_pts[k + 1], f_pts[k + 2]
        rising = (all_freqs - f_left) / max(float(f_center - f_left), 1e-12)
        falling = (f_right - all_freqs) / max(float(f_right - f_center), 1e-12)
        fb[k] = torch.clamp(torch.minimum(rising, falling), min=0.0)

    if norm == "slaney":
        enorm = 2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels])
        fb *= enorm.unsqueeze(1)
    return fb


class Spectrogram(torch.nn.Module):
    def __init__(
        self,
        n_fft=400,
        win_length=None,
        hop_length=None,
        pad=0,
        window_fn=torch.hann_window,
        power=2.0,
        normalized=False,
        wkwargs=None,
        center=True,
        pad_mode="reflect",
        onesided=True,
    ):
        super().__init__()
        self.n_fft = int(n_fft)
        self.win_length = int(win_length) if win_length is not None else int(n_fft)
        self.hop_length = int(hop_length) if hop_length is not None else self.win_length // 2
        self.pad = int(pad)
        self.power = power
        self.normalized = normalized
        self.center = center
        self.pad_mode = pad_mode
        self.onesided = onesided
        window = window_fn(self.win_length, **(wkwargs or {}))
        self.register_buffer("window", window, persistent=False)

    def forward(self, waveform):
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(waveform.dtype),
            center=self.center,
            pad_mode=self.pad_mode,
            normalized=self.normalized,
            onesided=self.onesided,
            return_complex=True,
        )
        if self.power is None:
            return spec
        return spec.abs() ** self.power


class MelSpectrogram(Spectrogram):
    def __init__(
        self,
        sample_rate=16000,
        n_fft=400,
        win_length=None,
        hop_length=None,
        f_min=0.0,
        f_max=None,
        pad=0,
        n_mels=128,
        window_fn=torch.hann_window,
        power=2.0,
        normalized=False,
        wkwargs=None,
        center=True,
        pad_mode="reflect",
        norm=None,
        mel_scale="htk",
        **_ignored,
    ):
        super().__init__(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            pad=pad,
            window_fn=window_fn,
            power=power,
            normalized=normalized,
            wkwargs=wkwargs,
            center=center,
            pad_mode=pad_mode,
        )
        self.sample_rate = int(sample_rate)
        self.f_min = float(f_min)
        self.f_max = float(f_max) if f_max is not None else None
        self.n_mels = int(n_mels)
        self.norm = norm
        self.mel_scale = mel_scale
        fb = _mel_filterbank(
            self.sample_rate,
            self.n_fft,
            self.n_mels,
            self.f_min,
            self.f_max,
            self.norm,
            self.mel_scale,
        )
        self.register_buffer("mel_fb", fb, persistent=False)

    def forward(self, waveform):
        spec = super().forward(waveform)  # (..., n_freq, T)
        return torch.matmul(self.mel_fb.to(spec.dtype), spec)


class AmplitudeToDB(torch.nn.Module):
    def __init__(self, stype="power", top_db=None):
        super().__init__()
        self.multiplier = 10.0 if stype == "power" else 20.0
        self.top_db = top_db

    def forward(self, x):
        if hasattr(x, "dtype") and x.dtype.is_complex:
            raise TypeError("AmplitudeToDB: complex input not supported")
        x = x.clamp_min(torch.finfo(x.dtype).tiny if x.dtype.is_floating_point else 1.0)
        db = self.multiplier * torch.log10(x)
        if self.top_db is not None:
            ref = db.amax(dim=tuple(range(db.ndim - 1)), keepdim=True)
            db = torch.maximum(db, ref - float(self.top_db))
        return db
