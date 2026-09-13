"""用 librosa 替代 torchaudio.compliance.kaldi.fbank"""
import numpy as np
import torch
import librosa


def fbank(waveform, num_mel_bins=80, frame_length=25, frame_shift=10,
          sample_frequency=16000, sample_rate=None, dither=0, **kwargs):
    sample_rate = sample_rate or sample_frequency
    """
    模拟 torchaudio.compliance.kaldi.fbank
    返回 (frames, num_mel_bins) 的 tensor
    """
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()

    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim > 1:
        waveform = waveform[0]  # [1, time] -> [time]

    # kaldi 参数转换
    frame_length_samples = int(frame_length * sample_rate / 1000)
    frame_shift_samples = int(frame_shift * sample_rate / 1000)

    # 提取 mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=frame_length_samples,
        hop_length=frame_shift_samples,
        win_length=frame_length_samples,
        n_mels=num_mel_bins,
        window='hamming',
        center=False,
    )

    # 转 log mel (kaldi 用 log)
    mel_spec = np.log(mel_spec + 1e-10)

    return torch.from_numpy(mel_spec.T)
