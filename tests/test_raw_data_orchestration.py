from pathlib import Path
from zipfile import ZipFile

import pytest

from femto_rul.orchestration.raw_data import extract_and_normalize_archives


def _archive(path: Path, entries: dict[str, str]) -> Path:
    with ZipFile(path, "w") as zip_file:
        for name, content in entries.items():
            zip_file.writestr(name, content)
    return path


def test_extracts_and_normalizes_release_split_names(tmp_path: Path) -> None:
    archives = [
        _archive(tmp_path / "training_set.zip", {"Learning_set/Bearing1_1/a": "1"}),
        _archive(
            tmp_path / "validation_set.zip", {"Full_Test_Set/Bearing1_3/a": "2"}
        ),
        _archive(tmp_path / "test_set.zip", {"Test_set/Bearing1_3/a": "3"}),
    ]

    output = extract_and_normalize_archives(archives, tmp_path / "output")

    assert (output / "Training_set/Bearing1_1/a").read_text() == "1"
    assert (output / "Validation_Set/Bearing1_3/a").read_text() == "2"
    assert (output / "Test_set/Bearing1_3/a").read_text() == "3"


def test_rejects_zip_path_traversal(tmp_path: Path) -> None:
    archives = [
        _archive(tmp_path / "training_set.zip", {"../escape": "bad"}),
        _archive(tmp_path / "validation_set.zip", {"Full_Test_Set/a": "2"}),
        _archive(tmp_path / "test_set.zip", {"Test_set/a": "3"}),
    ]

    with pytest.raises(ValueError, match="Unsafe path"):
        extract_and_normalize_archives(archives, tmp_path / "output")
