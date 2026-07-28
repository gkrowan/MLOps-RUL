"""Time-domain summary features for one vibration channel's 0.1s snapshot.

Each function takes a 1D array of raw accelerometer samples (one channel,
one acc_*.csv file) and returns a scalar. Kurtosis uses scipy's default
(excess/Fisher kurtosis, so a Gaussian signal scores ~0, not ~3).
"""

import numpy as np
from scipy import stats

TIME_DOMAIN_FEATURE_NAMES = ["rms", "kurtosis", "skewness", "crest_factor"]


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def kurtosis(x: np.ndarray) -> float:
    return float(stats.kurtosis(x, fisher=True))


def skewness(x: np.ndarray) -> float:
    return float(stats.skew(x))


def crest_factor(x: np.ndarray) -> float:
    signal_rms = rms(x)
    if signal_rms == 0:
        return 0.0
    return float(np.max(np.abs(x)) / signal_rms)


def time_domain_features(x: np.ndarray) -> dict[str, float]:
    """All time-domain features for one channel's snapshot, as a dict keyed
    by name in TIME_DOMAIN_FEATURE_NAMES."""
    return {
        "rms": rms(x),
        "kurtosis": kurtosis(x),
        "skewness": skewness(x),
        "crest_factor": crest_factor(x),
    }
