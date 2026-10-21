"""Waveform augmentations that imitate phone-quality, noisy and reverberant audio."""

from __future__ import annotations

import random

import torch
import torchaudio.functional as audio_functional

from .config import SAMPLE_RATE, AugmentationConfig

TELEPHONE_SAMPLE_RATE = 8_000
TELEPHONE_BAND_HZ = (300.0, 3400.0)
MU_LAW_CHANNELS = 256
SIGNAL_EPSILON = 1e-8


def apply_telephone_filter(waveform: torch.Tensor) -> torch.Tensor:
    """Band-limit a 16 kHz waveform to the 300-3400 Hz telephone band."""
    narrowband = audio_functional.resample(
        waveform, orig_freq=SAMPLE_RATE, new_freq=TELEPHONE_SAMPLE_RATE
    )
    narrowband = audio_functional.highpass_biquad(
        narrowband, sample_rate=TELEPHONE_SAMPLE_RATE, cutoff_freq=TELEPHONE_BAND_HZ[0]
    )
    narrowband = audio_functional.lowpass_biquad(
        narrowband, sample_rate=TELEPHONE_SAMPLE_RATE, cutoff_freq=TELEPHONE_BAND_HZ[1]
    )
    return audio_functional.resample(
        narrowband, orig_freq=TELEPHONE_SAMPLE_RATE, new_freq=SAMPLE_RATE
    )


def apply_mu_law(waveform: torch.Tensor) -> torch.Tensor:
    """Quantise a waveform with 8-bit mu-law companding, as in G.711 telephony."""
    peak = waveform.abs().max().clamp_min(SIGNAL_EPSILON)
    encoded = audio_functional.mu_law_encoding(
        waveform / peak, quantization_channels=MU_LAW_CHANNELS
    )
    return audio_functional.mu_law_decoding(
        encoded, quantization_channels=MU_LAW_CHANNELS
    )


def make_white_noise(length: int) -> torch.Tensor:
    """Gaussian white noise with the given number of samples."""
    return torch.randn(length)


def make_pink_noise(length: int) -> torch.Tensor:
    """Noise whose power falls off as 1/f, with the given number of samples."""
    spectrum = torch.fft.rfft(torch.randn(length))
    frequency_bins = torch.arange(1, spectrum.numel() + 1, dtype=torch.float32)
    return torch.fft.irfft(spectrum / frequency_bins.sqrt(), n=length)


def mix_at_snr(
    waveform: torch.Tensor, noise: torch.Tensor, snr_db: float
) -> torch.Tensor:
    """Add noise to a waveform at the requested signal-to-noise ratio.

    :param waveform: Clean signal.
    :param noise: Noise with the same number of samples as the signal.
    :param snr_db: Target signal-to-noise ratio in decibels.
    """
    signal_power = waveform.pow(2).mean() + SIGNAL_EPSILON
    noise_power = noise.pow(2).mean() + SIGNAL_EPSILON
    noise_gain = torch.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10)))
    return waveform + noise_gain * noise


def add_reverb(waveform: torch.Tensor, decay_seconds: float) -> torch.Tensor:
    """Convolve a waveform with an exponentially decaying noise impulse response.

    :param waveform: Input signal.
    :param decay_seconds: Time for the reverberation to decay by 60 dB.
    """
    impulse_length = int(SAMPLE_RATE * decay_seconds)
    time_axis = torch.arange(impulse_length, dtype=torch.float32)
    decay_envelope = torch.exp(-time_axis / (SAMPLE_RATE * decay_seconds / 6.9))
    impulse_response = torch.randn(impulse_length) * decay_envelope
    impulse_response[0] = 1.0
    impulse_response = impulse_response / impulse_response.abs().sum()
    return audio_functional.fftconvolve(waveform, impulse_response)[: waveform.numel()]


def change_speed(waveform: torch.Tensor, factor: float) -> torch.Tensor:
    """Speed a waveform up (factor > 1) or slow it down, shifting pitch with it."""
    return audio_functional.resample(
        waveform, orig_freq=int(SAMPLE_RATE * factor), new_freq=SAMPLE_RATE
    )


class Augmenter:
    """
    Randomly applies the configured augmentations to a 16 kHz mono waveform.

    Randomness comes from Python's global `random` module, which PyTorch reseeds in
    every `DataLoader` worker, so workers draw different augmentations.
    """

    def __init__(self, config: AugmentationConfig) -> None:
        """
        :param config: Probabilities and ranges for each augmentation.
        """
        self.config = config

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Return an augmented copy of a 1-D float waveform."""
        config = self.config
        if random.random() < config.speed_probability:
            waveform = change_speed(
                waveform, factor=random.choice(config.speed_factors)
            )
        if random.random() < config.reverb_probability:
            waveform = add_reverb(
                waveform, decay_seconds=random.uniform(*config.reverb_decay_seconds)
            )
        if random.random() < config.telephone_probability:
            waveform = apply_telephone_filter(waveform)
            if random.random() < config.mu_law_probability:
                waveform = apply_mu_law(waveform)
        if random.random() < config.noise_probability:
            noise_factory = random.choice((make_white_noise, make_pink_noise))
            waveform = mix_at_snr(
                waveform,
                noise=noise_factory(waveform.numel()),
                snr_db=random.uniform(*config.noise_snr_db),
            )
        return waveform
