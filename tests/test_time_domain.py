import numpy as np

from femto_rul.features.time_domain import crest_factor, kurtosis, rms, skewness


def test_rms_constant_signal():
    x = np.full(1000, 3.0)
    assert rms(x) == 3.0


def test_rms_sine_wave():
    # RMS of a sine wave with amplitude A is A/sqrt(2)
    t = np.linspace(0, 1, 10_000, endpoint=False)
    x = 2.0 * np.sin(2 * np.pi * 50 * t)
    assert abs(rms(x) - 2.0 / np.sqrt(2)) < 1e-3


def test_kurtosis_gaussian_is_near_zero_excess():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200_000)
    assert abs(kurtosis(x)) < 0.1


def test_skewness_symmetric_signal_is_near_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200_000)
    assert abs(skewness(x)) < 0.05


def test_crest_factor_single_spike():
    x = np.zeros(1000)
    x[500] = 10.0
    # rms = sqrt(100/1000) = sqrt(0.1); crest factor = 10 / sqrt(0.1)
    assert abs(crest_factor(x) - 10.0 / np.sqrt(0.1)) < 1e-6


def test_crest_factor_zero_signal_does_not_divide_by_zero():
    assert crest_factor(np.zeros(100)) == 0.0
