"""Pure-torch (+scipy) implementations of torchaudio.functional APIs used here."""

import math

import torch

try:
    import scipy.signal as _scipy_signal
except Exception:  # pragma: no cover
    _scipy_signal = None

__all__ = [
    "resample",
    "biquad",
    "bass_biquad",
    "treble_biquad",
    "equalizer_biquad",
    "lowpass_biquad",
    "highpass_biquad",
]

_KAISER_DEFAULT_BETA = 12.9866004


def _sinc(x):
    out = torch.where(
        x.abs() < 1e-8, torch.ones_like(x), torch.sin(math.pi * x) / (math.pi * x)
    )
    return out


def resample(
    waveform,
    orig_freq,
    new_freq,
    lowpass_filter_width=6,
    rolloff=0.99,
    resampling_method="sinc_interp_hann",
    beta=None,
):
    """Windowed-sinc resample of a (*, L) tensor to ``new_freq``.

    Supports the torchaudio parameter surface (lowpass_filter_width, rolloff,
    resampling_method in {sinc_interp_hann, sinc_interp_kaiser}, beta). Other
    torchaudio methods (nearest/linear/kaldi_*) fall back to interpolation.
    """
    if int(orig_freq) == int(new_freq):
        return waveform
    if orig_freq <= 0 or new_freq <= 0:
        raise ValueError("resample: sample rates must be positive")
    if resampling_method not in ("sinc_interp_hann", "sinc_interp_kaiser"):
        out_len = int(math.ceil(waveform.shape[-1] * float(new_freq) / float(orig_freq)))
        mode = "nearest" if resampling_method == "nearest" else "linear"
        shape = waveform.shape
        flat = waveform.reshape(-1, 1, shape[-1]).float()
        y = torch.nn.functional.interpolate(
            flat, size=out_len, mode=mode, align_corners=False
        )
        return y.reshape(*shape[:-1], out_len).to(waveform.dtype)

    g = math.gcd(int(orig_freq), int(new_freq))
    orig_freq = int(orig_freq) // g
    new_freq = int(new_freq) // g

    L = waveform.shape[-1]
    n_out = (L * new_freq + orig_freq - 1) // orig_freq  # ceil division
    device = waveform.device
    dtype = waveform.dtype

    width = max(int(lowpass_filter_width), 1)
    # Anti-alias cutoff in cycles per *input* sample.
    cutoff = 0.5 * float(rolloff) * min(1.0, float(new_freq) / float(orig_freq))

    pos = (
        torch.arange(n_out, device=device, dtype=torch.float64)
        * (float(orig_freq) / float(new_freq))
    )
    base = torch.floor(pos)
    frac = pos - base
    taps = torch.arange(
        -(width - 1), width + 1, device=device, dtype=torch.float64
    )
    arg = taps.unsqueeze(0) - frac.unsqueeze(1)  # (n_out, taps) input-sample offsets

    if resampling_method == "sinc_interp_kaiser":
        b = float(beta) if beta is not None else _KAISER_DEFAULT_BETA
        ratio = torch.clamp(arg / float(width), -1.0, 1.0)
        win = torch.i0(b * torch.sqrt(torch.clamp(1.0 - ratio * ratio, min=0.0))) / torch.i0(
            torch.tensor(b, dtype=torch.float64, device=device)
        )
    else:  # hann
        win = 0.5 * (1.0 + torch.cos(math.pi * torch.clamp(arg / float(width), -1.0, 1.0)))

    weights = _sinc(arg * cutoff) * win * cutoff  # analytic DC normalization
    weights = weights.to(dtype)

    idx = (base.long().unsqueeze(1) + taps.long().unsqueeze(0))  # (n_out, taps)
    flat = waveform.reshape(-1, L)
    pad_l = width - 1
    padded = torch.nn.functional.pad(flat, (pad_l, width))
    idx2 = (idx + pad_l).clamp(0, padded.shape[-1] - 1)
    gathered = padded[:, idx2]  # (C, n_out, taps)
    out = (gathered.to(weights.dtype) * weights.unsqueeze(0)).sum(dim=-1)
    return out.reshape(*waveform.shape[:-1], n_out)


# ---------------------------------------------------------------------------
# Biquad filters (RBJ cookbook), applied with scipy.signal.lfilter.
# Waveform convention: (*, L), filtering along the last (time) axis.
# ---------------------------------------------------------------------------


def _lfilter_biquad(waveform, b, a):
    if _scipy_signal is None:
        raise RuntimeError(
            "torchaudio shim: scipy is required for biquad filters"
        )
    import numpy as np

    x = waveform.detach().cpu().numpy()
    dtype = x.dtype
    y = _scipy_signal.lfilter(b, a, x.astype("float64"), axis=-1)
    return torch.from_numpy(y.astype(dtype)).to(waveform.device)


def biquad(waveform, b0, b1, b2, a0, a1, a2):
    return _lfilter_biquad(waveform, [b0, b1, b2], [a0, a1, a2])


def _shelf(waveform, sample_rate, gain, central_freq, Q, kind):
    A = 10.0 ** (float(gain) / 40.0)
    w0 = 2.0 * math.pi * float(central_freq) / float(sample_rate)
    alpha = math.sin(w0) / (2.0 * float(Q))
    cw = math.cos(w0)
    sqA = math.sqrt(A)
    if kind == "low":
        b0 = A * ((A + 1) - (A - 1) * cw + 2 * sqA * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cw)
        b2 = A * ((A + 1) - (A - 1) * cw - 2 * sqA * alpha)
        a0 = (A + 1) + (A - 1) * cw + 2 * sqA * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cw)
        a2 = (A + 1) + (A - 1) * cw - 2 * sqA * alpha
    else:  # high shelf
        b0 = A * ((A + 1) + (A - 1) * cw + 2 * sqA * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cw)
        b2 = A * ((A + 1) + (A - 1) * cw - 2 * sqA * alpha)
        a0 = (A + 1) - (A - 1) * cw + 2 * sqA * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cw)
        a2 = (A + 1) - (A - 1) * cw - 2 * sqA * alpha
    return _lfilter_biquad(
        waveform, [b0 / a0, b1 / a0, b2 / a0], [1.0, a1 / a0, a2 / a0]
    )


def bass_biquad(waveform, sample_rate, gain, central_freq=100, Q=0.707):
    return _shelf(waveform, sample_rate, gain, central_freq, Q, "low")


def treble_biquad(waveform, sample_rate, gain, central_freq=3000, Q=0.707):
    return _shelf(waveform, sample_rate, gain, central_freq, Q, "high")


def equalizer_biquad(waveform, sample_rate, center_freq, gain, Q=0.707):
    A = 10.0 ** (float(gain) / 40.0)
    w0 = 2.0 * math.pi * float(center_freq) / float(sample_rate)
    alpha = math.sin(w0) / (2.0 * float(Q))
    cw = math.cos(w0)
    b0 = 1 + alpha * A
    b1 = -2 * cw
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * cw
    a2 = 1 - alpha / A
    return _lfilter_biquad(
        waveform, [b0 / a0, b1 / a0, b2 / a0], [1.0, a1 / a0, a2 / a0]
    )


def lowpass_biquad(waveform, sample_rate, cutoff_freq, Q=0.707):
    w0 = 2.0 * math.pi * float(cutoff_freq) / float(sample_rate)
    alpha = math.sin(w0) / (2.0 * float(Q))
    cw = math.cos(w0)
    b0 = (1 - cw) / 2
    b1 = 1 - cw
    b2 = (1 - cw) / 2
    a0 = 1 + alpha
    a1 = -2 * cw
    a2 = 1 - alpha
    return _lfilter_biquad(
        waveform, [b0 / a0, b1 / a0, b2 / a0], [1.0, a1 / a0, a2 / a0]
    )


def highpass_biquad(waveform, sample_rate, cutoff_freq, Q=0.707):
    w0 = 2.0 * math.pi * float(cutoff_freq) / float(sample_rate)
    alpha = math.sin(w0) / (2.0 * float(Q))
    cw = math.cos(w0)
    b0 = (1 + cw) / 2
    b1 = -(1 + cw)
    b2 = (1 + cw) / 2
    a0 = 1 + alpha
    a1 = -2 * cw
    a2 = 1 - alpha
    return _lfilter_biquad(
        waveform, [b0 / a0, b1 / a0, b2 / a0], [1.0, a1 / a0, a2 / a0]
    )
