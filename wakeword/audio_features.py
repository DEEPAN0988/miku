"""
Acoustic Feature Extractor for Wake Word & ASR.
Calculates MFCC and filterbank features using pure NumPy & SciPy.
No external black-box models or third-party cloud dependencies.
"""

import numpy as np
from scipy.fftpack import dct


def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700.0)


def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595.0) - 1)


def get_filterbanks(num_filters: int = 26, nfft: int = 512, sample_rate: int = 16000, low_freq: float = 0, high_freq: float = None):
    high_freq = high_freq or sample_rate / 2
    low_mel = hz_to_mel(low_freq)
    high_mel = hz_to_mel(high_freq)
    mel_points = np.linspace(low_mel, high_mel, num_filters + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((nfft + 1) * hz_points / sample_rate).astype(int)

    fbank = np.zeros((num_filters, int(np.floor(nfft / 2 + 1))))
    for m in range(1, num_filters + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]

        for k in range(f_m_minus, f_m):
            if f_m != f_m_minus:
                fbank[m - 1, k] = (k - bin_points[m - 1]) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus != f_m:
                fbank[m - 1, k] = (bin_points[m + 1] - k) / (f_m_plus - f_m)
    return fbank


def extract_mfcc(signal: np.ndarray, sample_rate: int = 16000, num_cepstral: int = 13, num_filters: int = 26, frame_len: float = 0.025, frame_step: float = 0.010, target_frames: int = 99) -> np.ndarray:
    """
    Extract MFCC features from 1D audio signal array.
    Returns array of shape (target_frames, num_cepstral).
    """
    # Pre-emphasis filter
    emphasized = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])
    
    # Framing
    frame_length = int(round(frame_len * sample_rate))
    frame_step_samples = int(round(frame_step * sample_rate))
    signal_length = len(emphasized)
    
    if signal_length <= frame_length:
        num_frames = 1
    else:
        num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step_samples)) + 1

    pad_signal_length = (num_frames - 1) * frame_step_samples + frame_length
    pad = np.zeros((pad_signal_length - signal_length,))
    pad_signal = np.append(emphasized, pad)

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(
        np.arange(0, num_frames * frame_step_samples, frame_step_samples), (frame_length, 1)
    ).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    
    # Hamming window
    frames *= np.hamming(frame_length)

    # FFT & Power Spectrum
    nfft = 512
    mag_frames = np.absolute(np.fft.rfft(frames, nfft))
    pow_frames = ((1.0 / nfft) * ((mag_frames) ** 2))

    # Filterbanks
    fbanks = get_filterbanks(num_filters, nfft, sample_rate)
    filter_banks = np.dot(pow_frames, fbanks.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)

    # DCT to obtain MFCCs
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, :num_cepstral]

    # Normalize frames to target_frames (e.g. 99 frames = 1 second)
    if mfcc.shape[0] < target_frames:
        pad_width = ((0, target_frames - mfcc.shape[0]), (0, 0))
        mfcc = np.pad(mfcc, pad_width, mode='constant')
    else:
        mfcc = mfcc[:target_frames, :]

    # Standardize
    mean = np.mean(mfcc)
    std = np.std(mfcc) + 1e-8
    return (mfcc - mean) / std
