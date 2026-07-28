"""Shared constants for the FEMTO/PRONOSTIA dataset layout.

Values here were confirmed against the actual downloaded dataset by
scripts/verify_data.py — see docs/data_notes.md for the full verification
report and rationale.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

TRAINING_SET_DIR = DATA_DIR / "Training_set"
VALIDATION_SET_DIR = DATA_DIR / "Validation_Set"  # full run-to-failure ground truth for test bearings
TEST_SET_DIR = DATA_DIR / "Test_set"  # truncated bearings; true RUL is held out in VALIDATION_SET_DIR

ACCELEROMETER_SAMPLING_RATE_HZ = 25_600
ACC_SAMPLES_PER_FILE = 2560  # 0.1s snapshot
ACC_COLUMNS = ["hour", "minute", "second", "microsecond", "horiz_accel_g", "vert_accel_g"]

TEMP_SAMPLING_RATE_HZ = 10
TEMP_SAMPLES_PER_FILE = 600  # nominal; the last file of a run is often a shorter partial chunk
TEMP_COLUMNS = ["hour", "minute", "second", "hundredth_second", "temperature_c"]

FILE_INTERVAL_SECONDS = 10  # a new acc/temp file is written roughly every 10s of run time

# operating condition -> (rotation speed rpm, radial load N), per FEMTO/PHM12 spec
CONDITIONS = {
    1: {"rotation_speed_rpm": 1800, "radial_load_n": 4000},
    2: {"rotation_speed_rpm": 1650, "radial_load_n": 4200},
    3: {"rotation_speed_rpm": 1500, "radial_load_n": 5000},
}
