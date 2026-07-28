"""Frequency-domain (FFT band energy) features for one vibration channel's
0.1s snapshot.

Approach: generic equal-width frequency bins spanning 0 to the Nyquist
frequency (half the 25.6kHz sampling rate), rather than bins targeted at
specific bearing fault frequencies (BPFO/BPFI/BSF/FTF) — the latter needs
the exact PRONOSTIA bearing geometry verified against the dataset spec
sheet, which we haven't done. Equal-width binning needs no geometry lookup
and still captures shifts in spectral shape as degradation progresses.
See docs/data_notes.md for the tradeoff.
"""

import numpy as np

from femto_rul.config import ACCELEROMETER_SAMPLING_RATE_HZ

DEFAULT_N_BANDS = 8


def band_edges_hz(sampling_rate_hz: float = ACCELEROMETER_SAMPLING_RATE_HZ, n_bands: int = DEFAULT_N_BANDS) -> np.ndarray:
    nyquist_hz = sampling_rate_hz / 2
    return np.linspace(0, nyquist_hz, n_bands + 1)


def fft_band_energy(
    x: np.ndarray,
    sampling_rate_hz: float = ACCELEROMETER_SAMPLING_RATE_HZ,
    n_bands: int = DEFAULT_N_BANDS,
) -> dict[str, float]:
    """Power spectrum energy summed within each of n_bands equal-width bins
    from 0Hz to Nyquist. Keys are "fft_band_{i}" in increasing-frequency order."""
    n = len(x)
    power = np.abs(np.fft.rfft(x)) ** 2 / n
    freqs = np.fft.rfftfreq(n, d=1 / sampling_rate_hz)

    edges = band_edges_hz(sampling_rate_hz, n_bands)
    features = {}
    for i in range(n_bands):
        lo, hi = edges[i], edges[i + 1]
        in_band = (freqs >= lo) & (freqs < hi) if i < n_bands - 1 else (freqs >= lo) & (freqs <= hi)
        features[f"fft_band_{i}"] = float(power[in_band].sum())
    return features


def fft_band_feature_names(n_bands: int = DEFAULT_N_BANDS) -> list[str]:
    return [f"fft_band_{i}" for i in range(n_bands)]
