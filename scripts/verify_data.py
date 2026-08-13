"""Verify integrity of the FEMTO/PRONOSTIA bearing dataset.

The verifier distinguishes between:

CRITICAL issues
---------------
- missing required dataset split directories
- no bearing directories
- malformed / short / ragged acceleration snapshots
- acceleration file-numbering gaps
- invalid bearing directory names
- Test_set / Validation_Set truncation mismatches

WARNINGS
--------
- temperature file-numbering gaps
- partial / short temperature chunks
- stray non-dataset files

Temperature is warning-only by default because Feature Set V1 is vibration-only.
Use --strict-temperature if temperature is later promoted to a model feature.

Delimiter differences are informational because the released dataset legitimately
contains both comma- and semicolon-delimited files.

Usage:
    python scripts/verify_data.py
    python scripts/verify_data.py --data-dir data/raw
    python scripts/verify_data.py --strict-temperature
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from femto_rul.config import (
    ACC_COLUMNS,
    ACC_SAMPLES_PER_FILE,
    EXTRACTED_DATA_DIR,
    FILE_INTERVAL_SECONDS,
    TEMP_COLUMNS,
    TEMP_SAMPLES_PER_FILE,
)

ACC_EXPECTED_ROWS = ACC_SAMPLES_PER_FILE
ACC_EXPECTED_COLS = len(ACC_COLUMNS)

TEMP_EXPECTED_ROWS = TEMP_SAMPLES_PER_FILE
TEMP_EXPECTED_COLS = len(TEMP_COLUMNS)

BEARING_DIR_RE = re.compile(r"^Bearing(\d+)_(\d+)$")
FILE_RE = re.compile(r"^(acc|temp)_(\d+)\.csv$")


@dataclass
class FileStat:
    name: str
    delimiter: str
    row_count: int
    col_counts: set[int]


@dataclass
class BearingReport:
    split: str
    name: str
    condition: int
    acc_files: list[FileStat] = field(default_factory=list)
    temp_files: list[FileStat] = field(default_factory=list)
    acc_gaps: list[int] = field(default_factory=list)
    temp_gaps: list[int] = field(default_factory=list)


def find_gaps(numbers: list[int]) -> list[int]:
    if not numbers:
        return []
    numbers = sorted(numbers)
    expected = set(range(1, numbers[-1] + 1))
    return sorted(expected - set(numbers))


def read_lines(path: Path) -> list[bytes]:
    data = path.read_bytes()
    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    return lines


def stat_file(path: Path) -> FileStat:
    lines = read_lines(path)
    row_count = len(lines)
    if row_count == 0:
        return FileStat(
            path.name,
            delimiter="?",
            row_count=0,
            col_counts=set(),
        )

    first = lines[0]
    delimiter = ";" if b";" in first else ","
    sample_idx = {0, row_count // 2, row_count - 1}
    col_counts = {
        lines[i].count(delimiter.encode()) + 1
        for i in sample_idx
    }

    return FileStat(
        path.name,
        delimiter=delimiter,
        row_count=row_count,
        col_counts=col_counts,
    )


def verify_bearing_dir(
    split: str,
    bearing_dir: Path,
) -> tuple[BearingReport, list[str]]:
    match = BEARING_DIR_RE.match(bearing_dir.name)
    condition = int(match.group(1)) if match else -1

    report = BearingReport(
        split=split,
        name=bearing_dir.name,
        condition=condition,
    )

    stray_files: list[str] = []
    acc_nums: list[int] = []
    temp_nums: list[int] = []

    for file_path in sorted(bearing_dir.iterdir()):
        if not file_path.is_file():
            continue

        file_match = FILE_RE.match(file_path.name)
        if not file_match:
            stray_files.append(str(file_path))
            continue

        kind = file_match.group(1)
        number = int(file_match.group(2))
        stat = stat_file(file_path)

        if kind == "acc":
            acc_nums.append(number)
            report.acc_files.append(stat)
        else:
            temp_nums.append(number)
            report.temp_files.append(stat)

    report.acc_gaps = find_gaps(acc_nums)
    report.temp_gaps = find_gaps(temp_nums)

    return report, stray_files


def summarize_shape_issues(
    files: list[FileStat],
    expected_rows: int,
    expected_cols: int,
) -> list[str]:
    issues: list[str] = []

    for file_stat in files:
        if len(file_stat.col_counts) > 1:
            issues.append(
                f"{file_stat.name}: ragged — column counts differ "
                f"within file {sorted(file_stat.col_counts)}"
            )
            continue

        cols = (
            next(iter(file_stat.col_counts))
            if file_stat.col_counts
            else 0
        )

        if (
            file_stat.row_count != expected_rows
            or cols != expected_cols
        ):
            issues.append(
                f"{file_stat.name}: shape "
                f"{file_stat.row_count}x{cols} "
                f"(expected {expected_rows}x{expected_cols})"
            )

    return issues


def delimiter_summary(files: list[FileStat]) -> str | None:
    if not files:
        return None

    delimiters = {file_stat.delimiter for file_stat in files}

    if delimiters == {","}:
        return None

    counts = defaultdict(int)

    for file_stat in files:
        counts[file_stat.delimiter] += 1

    return ", ".join(
        f"{count} file(s) use '{delimiter}'"
        for delimiter, count in sorted(counts.items())
    )


def parse_row(line: bytes, delimiter: str) -> list[float]:
    return [
        float(value)
        for value in line.split(delimiter.encode())
    ]


def check_test_is_prefix_of_validation(
    data_dir: Path,
    tolerance: float = 1e-6,
) -> tuple[list[str], list[str]]:
    """Return (informational notes, critical mismatch messages)."""

    notes: list[str] = []
    critical: list[str] = []

    test_dir = data_dir / "Test_set"
    validation_dir = data_dir / "Validation_Set"

    if not (
        test_dir.is_dir()
        and validation_dir.is_dir()
    ):
        return notes, critical

    test_bearings = sorted(
        directory.name
        for directory in test_dir.iterdir()
        if directory.is_dir()
    )

    for bearing in test_bearings:
        test_bearing = test_dir / bearing
        validation_bearing = validation_dir / bearing

        if not validation_bearing.is_dir():
            message = (
                f"{bearing}: in Test_set but no matching "
                "Validation_Set directory"
            )
            notes.append(f"{message} [MISMATCH]")
            critical.append(message)
            continue

        test_acc = sorted(test_bearing.glob("acc_*.csv"))
        validation_acc = sorted(
            validation_bearing.glob("acc_*.csv")
        )

        if not test_acc or not validation_acc:
            message = (
                f"{bearing}: missing acceleration snapshots "
                "in Test_set or Validation_Set"
            )
            notes.append(f"{message} [MISMATCH]")
            critical.append(message)
            continue

        if len(test_acc) >= len(validation_acc):
            message = (
                f"{bearing}: Test_set has {len(test_acc)} acc files, "
                f"Validation_Set has {len(validation_acc)} — "
                "expected Test_set to be strictly shorter"
            )
            notes.append(f"{message} [MISMATCH]")
            critical.append(message)
            continue

        def matches(
            test_path: Path,
            validation_path: Path,
        ) -> bool:
            test_lines = read_lines(test_path)
            validation_lines = read_lines(validation_path)

            if len(test_lines) != len(validation_lines):
                return False

            if not test_lines or not validation_lines:
                return False

            test_delimiter = (
                ";" if b";" in test_lines[0] else ","
            )
            validation_delimiter = (
                ";" if b";" in validation_lines[0] else ","
            )

            stride = max(len(test_lines) // 5, 1)

            for test_line, validation_line in zip(
                test_lines[::stride],
                validation_lines[::stride],
            ):
                test_values = parse_row(
                    test_line,
                    test_delimiter,
                )
                validation_values = parse_row(
                    validation_line,
                    validation_delimiter,
                )

                if len(test_values) != len(validation_values):
                    return False

                if any(
                    abs(test_value - validation_value) > tolerance
                    for test_value, validation_value in zip(
                        test_values,
                        validation_values,
                    )
                ):
                    return False

            return True

        first_match = matches(
            test_acc[0],
            validation_acc[0],
        )

        last_shared_index = len(test_acc) - 1

        last_match = matches(
            test_acc[last_shared_index],
            validation_acc[last_shared_index],
        )

        is_match = first_match and last_match

        hidden_files = (
            len(validation_acc) - len(test_acc)
        )

        held_out_seconds = (
            hidden_files * FILE_INTERVAL_SECONDS
        )

        status = "OK" if is_match else "MISMATCH"

        notes.append(
            f"{bearing}: truncated at "
            f"{len(test_acc)}/{len(validation_acc)} acc files "
            f"({hidden_files} files / "
            f"~{held_out_seconds}s of run held out "
            f"as RUL ground truth) [{status}]"
        )

        if not is_match:
            critical.append(
                f"{bearing}: Test_set is not a numeric prefix "
                "of Validation_Set"
            )

    return notes, critical


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify integrity of extracted FEMTO/PRONOSTIA data."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=EXTRACTED_DATA_DIR,
        help=(
            "Parent directory containing Training_set, "
            "Validation_Set and Test_set. "
            f"Default from config: {EXTRACTED_DATA_DIR}"
        ),
    )

    parser.add_argument(
        "--strict-temperature",
        action="store_true",
        help=(
            "Treat temperature gaps/shape differences as critical. "
            "By default they are warnings because Feature Set V1 "
            "does not use temperature."
        ),
    )

    args = parser.parse_args()
    data_dir = args.data_dir.resolve()

    if not data_dir.is_dir():
        raise SystemExit(
            f"Configured data directory does not exist: {data_dir}"
        )

    splits = [
        "Training_set",
        "Validation_Set",
        "Test_set",
    ]

    missing_splits = [
        split
        for split in splits
        if not (data_dir / split).is_dir()
    ]

    if missing_splits:
        print("=" * 70)
        print("FEMTO data verification FAILED")
        print("=" * 70)
        print(f"Configured data directory: {data_dir}")
        print()
        print("Missing required split directories:")

        for split in missing_splits:
            print(f"  - {data_dir / split}")

        raise SystemExit(1)

    all_reports: list[BearingReport] = []
    all_strays: list[str] = []

    for split in splits:
        split_dir = data_dir / split

        for bearing_dir in sorted(split_dir.iterdir()):
            if not bearing_dir.is_dir():
                continue

            report, strays = verify_bearing_dir(
                split,
                bearing_dir,
            )

            all_reports.append(report)
            all_strays.extend(strays)

    if not all_reports:
        raise SystemExit(
            f"No bearing directories found under {data_dir}"
        )

    print("=" * 70)
    print("FEMTO data verification report")
    print("=" * 70)
    print(f"Data directory: {data_dir}")

    by_split = defaultdict(list)

    for report in all_reports:
        by_split[report.split].append(report)

    critical_issues: list[str] = []
    warning_issues: list[str] = []
    delimiter_notes: list[str] = []

    for split in splits:
        reports = by_split.get(split, [])

        if not reports:
            critical_issues.append(
                f"{split}: no bearing directories found"
            )
            continue

        print(f"\n{split}: {len(reports)} bearings")

        for report in reports:
            acc_issues = summarize_shape_issues(
                report.acc_files,
                ACC_EXPECTED_ROWS,
                ACC_EXPECTED_COLS,
            )

            temp_issues = summarize_shape_issues(
                report.temp_files,
                TEMP_EXPECTED_ROWS,
                TEMP_EXPECTED_COLS,
            )

            bearing_critical: list[str] = []
            bearing_warnings: list[str] = []

            if report.condition not in {1, 2, 3}:
                bearing_critical.append(
                    "invalid bearing directory name / condition"
                )

            if not report.acc_files:
                bearing_critical.append(
                    "no acceleration files found"
                )

            if report.acc_gaps:
                bearing_critical.append(
                    "acc file numbering gaps at "
                    f"{report.acc_gaps[:10]}"
                )

            if acc_issues:
                bearing_critical.append(
                    f"{len(acc_issues)} acc shape issue(s): "
                    f"{acc_issues[:3]}"
                )

            if report.temp_gaps:
                message = (
                    "temp file numbering gaps at "
                    f"{report.temp_gaps[:10]}"
                )

                if args.strict_temperature:
                    bearing_critical.append(message)
                else:
                    bearing_warnings.append(message)

            if temp_issues:
                message = (
                    f"{len(temp_issues)} temp shape issue(s): "
                    f"{temp_issues[:3]}"
                )

                if args.strict_temperature:
                    bearing_critical.append(message)
                else:
                    bearing_warnings.append(message)

            acc_delimiter = delimiter_summary(
                report.acc_files
            )

            temp_delimiter = delimiter_summary(
                report.temp_files
            )

            if acc_delimiter:
                delimiter_notes.append(
                    f"{split}/{report.name} acc_*.csv: "
                    f"{acc_delimiter}"
                )

            if temp_delimiter:
                delimiter_notes.append(
                    f"{split}/{report.name} temp_*.csv: "
                    f"{temp_delimiter}"
                )

            if bearing_critical:
                status = "  [CRITICAL]"
            elif bearing_warnings:
                status = "  [WARN]"
            else:
                status = ""

            print(
                f"  {report.name} "
                f"(condition {report.condition}): "
                f"{len(report.acc_files)} acc files, "
                f"{len(report.temp_files)} temp files"
                f"{status}"
            )

            for issue in bearing_critical:
                message = (
                    f"{split}/{report.name}: {issue}"
                )
                critical_issues.append(message)
                print(f"      - CRITICAL: {issue}")

            for issue in bearing_warnings:
                message = (
                    f"{split}/{report.name}: {issue}"
                )
                warning_issues.append(message)
                print(f"      - WARNING: {issue}")

    if all_strays:
        print(
            "\nStray files "
            "(not acc_*/temp_*; ignored): "
            f"{len(all_strays)}"
        )

        for stray in all_strays:
            print(f"  - {stray}")
            warning_issues.append(
                f"stray file: {stray}"
            )

    print("\n" + "=" * 70)
    print(
        "Delimiter irregularities "
        "(informational; supported by ingestion)"
    )
    print("=" * 70)

    if delimiter_notes:
        for note in delimiter_notes:
            print(f"  {note}")
    else:
        print("  none — all files comma-delimited")

    print("\n" + "=" * 70)
    print(
        "Test_set vs Validation_Set "
        "(truncation / RUL ground-truth check)"
    )
    print("=" * 70)

    truncation_notes, truncation_critical = (
        check_test_is_prefix_of_validation(data_dir)
    )

    for note in truncation_notes:
        print(f"  {note}")

    critical_issues.extend(truncation_critical)

    print("\n" + "=" * 70)

    if critical_issues:
        print(
            "FAILED — "
            f"{len(critical_issues)} critical issue(s), "
            f"{len(warning_issues)} warning(s), "
            f"{len(all_reports)} bearing directories checked."
        )
        print("=" * 70)

        print("\nCritical issues:")
        for issue in critical_issues:
            print(f"  - {issue}")

        raise SystemExit(1)

    if warning_issues:
        print(
            "PASS WITH WARNINGS — "
            f"0 critical issues, "
            f"{len(warning_issues)} warning(s), "
            f"{len(all_reports)} bearing directories checked."
        )
        print(
            "Temperature warnings are non-blocking because "
            "Feature Set V1 is vibration-only."
        )
    else:
        print(
            "PASS — "
            f"{len(all_reports)} bearing directories checked; "
            "no critical issues or warnings found."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()