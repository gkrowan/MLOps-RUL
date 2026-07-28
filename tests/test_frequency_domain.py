import numpy as np

from femto_rul.features.frequency_domain import band_edges_hz, fft_band_energy


def test_band_edges_span_zero_to_nyquist():
    edges = band_edges_hz(sampling_rate_hz=25_600, n_bands=8)
    assert edges[0] == 0
    assert edges[-1] == 12_800
    assert len(edges) == 9
    assert np.all(np.diff(edges) > 0)


def test_fft_band_energy_returns_one_value_per_band():
    x = np.random.default_rng(0).normal(size=2560)
    features = fft_band_energy(x, sampling_rate_hz=25_600, n_bands=8)
    assert list(features.keys()) == [f"fft_band_{i}" for i in range(8)]
    assert all(v >= 0 for v in features.values())


def test_pure_tone_energy_falls_in_expected_band():
    # a pure tone at 2000Hz should show up almost entirely in the band
    # covering [1600, 3200) Hz, given 8 equal bands over 0-12800Hz
    sampling_rate_hz = 25_600
    n = 2560
    t = np.arange(n) / sampling_rate_hz
    x = np.sin(2 * np.pi * 2000 * t)
    features = fft_band_energy(x, sampling_rate_hz=sampling_rate_hz, n_bands=8)

    edges = band_edges_hz(sampling_rate_hz, n_bands=8)
    expected_band = next(i for i in range(8) if edges[i] <= 2000 < edges[i + 1])

    total_energy = sum(features.values())
    assert features[f"fft_band_{expected_band}"] / total_energy > 0.9
