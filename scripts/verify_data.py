"""Verify integrity of the raw FEMTO/PRONOSTIA bearing dataset under data/.

Checks, per bearing directory:
  - acc_*.csv / temp_*.csv files are sequentially numbered starting at 1 (no gaps/dupes)
  - every acc_*.csv has the expected row count (2560) and column count (6)
  - every temp_*.csv has the expected row count (600) and column count (5)
    (a handful of temp files are genuinely short — the sensor's first read of a
    run sometimes captures fewer than 600 samples; these are reported, not
    treated as corruption, since Test_set and Validation_Set agree on them)
  - no unreadable / empty files
  - flags any stray files that aren't acc_*.csv or temp_*.csv

Known upstream quirk this script accounts for: the delimiter is NOT uniform
across the dataset. Most files are comma-delimited, but some bearings ship
their acc_*.csv and/or temp_*.csv as semicolon-delimited instead — and the
choice can differ between Test_set and Validation_Set copies of the *same*
bearing (e.g. Bearing1_4). It is consistent within a single (bearing, file
type) though: we never observed a mid-run switch. The delimiter is
autodetected per file rather than assumed, and usage is reported per bearing
so the ingestion loader knows to do the same.

Also cross-checks Test_set bearings against Validation_Set: Test_set is
expected to be a truncated PREFIX of Validation_Set (the full run-to-failure
ground truth released after the PHM12 challenge for scoring), so this reports
the truncation point per bearing — i.e. the RUL held out for that test
bearing.

Usage: python scripts/verify_data.py [--data-dir data]
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ACC_EXPECTED_ROWS = 2560
ACC_EXPECTED_COLS = 6
TEMP_EXPECTED_ROWS = 600
TEMP_EXPECTED_COLS = 5

BEARING_DIR_RE = re.compile(r"^Bearing(\d+)_(\d+)$")
FILE_RE = re.compile(r"^(acc|temp)_(\d+)\.csv$")


@dataclass
class FileStat:
    name: str
    delimiter: str
    row_count: int
    col_counts: set[int]  # distinct column counts seen (first/middle/last line)


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
        return FileStat(path.name, delimiter="?", row_count=0, col_counts=set())

    first = lines[0]
    delimiter = ";" if b";" in first else ","
    sample_idx = {0, row_count // 2, row_count - 1}
    col_counts = {lines[i].count(delimiter.encode()) + 1 for i in sample_idx}
    return FileStat(path.name, delimiter=delimiter, row_count=row_count, col_counts=col_counts)


def verify_bearing_dir(split: str, bearing_dir: Path) -> tuple[BearingReport, list[str]]:
    m = BEARING_DIR_RE.match(bearing_dir.name)
    condition = int(m.group(1)) if m else -1
    report = BearingReport(split=split, name=bearing_dir.name, condition=condition)
    stray_files: list[str] = []

    acc_nums, temp_nums = [], []
    for f in sorted(bearing_dir.iterdir()):
        if not f.is_file():
            continue
        fm = FILE_RE.match(f.name)
        if not fm:
            stray_files.append(str(f))
            continue
        kind, num = fm.group(1), int(fm.group(2))
        stat = stat_file(f)
        if kind == "acc":
            acc_nums.append(num)
            report.acc_files.append(stat)
        else:
            temp_nums.append(num)
            report.temp_files.append(stat)

    report.acc_gaps = find_gaps(acc_nums)
    report.temp_gaps = find_gaps(temp_nums)
    return report, stray_files


def summarize_shape_issues(files: list[FileStat], expected_rows: int, expected_cols: int) -> list[str]:
    issues = []
    for f in files:
        if len(f.col_counts) > 1:
            issues.append(f"{f.name}: ragged — column counts differ within file {sorted(f.col_counts)}")
            continue
        cols = next(iter(f.col_counts)) if f.col_counts else 0
        if f.row_count != expected_rows or cols != expected_cols:
            issues.append(f"{f.name}: shape {f.row_count}x{cols} (expected {expected_rows}x{expected_cols})")
    return issues


def delimiter_summary(files: list[FileStat]) -> str | None:
    if not files:
        return None
    delims = {f.delimiter for f in files}
    if delims == {","}:
        return None
    counts = defaultdict(int)
    for f in files:
        counts[f.delimiter] += 1
    return ", ".join(f"{n} file(s) use '{d}'" for d, n in sorted(counts.items()))


def parse_row(line: bytes, delimiter: str) -> list[float]:
    return [float(x) for x in line.split(delimiter.encode())]


def check_test_is_prefix_of_validation(data_dir: Path, tolerance: float = 1e-6) -> list[str]:
    """Test_set bearings should be a truncated PREFIX of the matching
    Validation_Set bearing (the released ground-truth full run). Compares
    parsed numeric values rather than raw bytes, since the two releases can
    use different delimiters for the same bearing (e.g. Bearing1_4)."""
    notes = []
    test_dir = data_dir / "Test_set"
    val_dir = data_dir / "Validation_Set"
    if not (test_dir.is_dir() and val_dir.is_dir()):
        return notes

    for bearing in sorted(d.name for d in test_dir.iterdir() if d.is_dir()):
        val_bearing = val_dir / bearing
        test_bearing = test_dir / bearing
        if not val_bearing.is_dir():
            notes.append(f"{bearing}: in Test_set but no matching Validation_Set dir")
            continue

        test_acc = sorted(test_bearing.glob("acc_*.csv"))
        val_acc = sorted(val_bearing.glob("acc_*.csv"))
        if len(test_acc) >= len(val_acc):
            notes.append(
                f"{bearing}: Test_set has {len(test_acc)} acc files, "
                f"Validation_Set has {len(val_acc)} — expected Test_set to be strictly shorter"
            )
            continue

        def matches(test_path: Path, val_path: Path) -> bool:
            t_lines, v_lines = read_lines(test_path), read_lines(val_path)
            if len(t_lines) != len(v_lines):
                return False
            t_delim = ";" if b";" in t_lines[0] else ","
            v_delim = ";" if b";" in v_lines[0] else ","
            for t_line, v_line in zip(t_lines[::len(t_lines) // 5 or 1], v_lines[::len(v_lines) // 5 or 1]):
                if any(
                    abs(a - b) > tolerance
                    for a, b in zip(parse_row(t_line, t_delim), parse_row(v_line, v_delim))
                ):
                    return False
            return True

        first_match = matches(test_acc[0], val_acc[0])
        last_shared_idx = len(test_acc) - 1
        last_match = matches(test_acc[last_shared_idx], val_acc[last_shared_idx])
        status = "OK" if (first_match and last_match) else "MISMATCH"
        hidden = len(val_acc) - len(test_acc)
        notes.append(
            f"{bearing}: truncated at {len(test_acc)}/{len(val_acc)} acc files "
            f"({hidden} files / ~{hidden * 10}s of run held out as RUL ground truth) [{status}]"
        )
    return notes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"data dir not found: {data_dir}")

    splits = ["Training_set", "Validation_Set", "Test_set"]
    all_reports: list[BearingReport] = []
    all_strays: list[str] = []

    for split in splits:
        split_dir = data_dir / split
        if not split_dir.is_dir():
            print(f"[!] missing expected split dir: {split}")
            continue
        for bearing_dir in sorted(split_dir.iterdir()):
            if not bearing_dir.is_dir():
                continue
            report, strays = verify_bearing_dir(split, bearing_dir)
            all_reports.append(report)
            all_strays.extend(strays)

    print("=" * 70)
    print("FEMTO data verification report")
    print("=" * 70)

    by_split = defaultdict(list)
    for r in all_reports:
        by_split[r.split].append(r)

    total_shape_issues = 0
    delimiter_notes = []
    for split in splits:
        reports = by_split.get(split, [])
        if not reports:
            continue
        print(f"\n{split}: {len(reports)} bearings")
        for r in reports:
            acc_issues = summarize_shape_issues(r.acc_files, ACC_EXPECTED_ROWS, ACC_EXPECTED_COLS)
            temp_issues = summarize_shape_issues(r.temp_files, TEMP_EXPECTED_ROWS, TEMP_EXPECTED_COLS)
            issues = []
            if r.acc_gaps:
                issues.append(f"acc file numbering gaps at {r.acc_gaps[:10]}")
            if r.temp_gaps:
                issues.append(f"temp file numbering gaps at {r.temp_gaps[:10]}")
            if acc_issues:
                issues.append(f"{len(acc_issues)} acc shape issue(s): {acc_issues[:3]}")
            if temp_issues:
                issues.append(f"{len(temp_issues)} temp shape issue(s): {temp_issues[:3]}")
            total_shape_issues += len(acc_issues) + len(temp_issues)

            acc_delim = delimiter_summary(r.acc_files)
            temp_delim = delimiter_summary(r.temp_files)
            if acc_delim:
                delimiter_notes.append(f"{split}/{r.name} acc_*.csv: {acc_delim}")
            if temp_delim:
                delimiter_notes.append(f"{split}/{r.name} temp_*.csv: {temp_delim}")

            flag = "  [ISSUES]" if issues else ""
            print(
                f"  {r.name} (condition {r.condition}): "
                f"{len(r.acc_files)} acc files, {len(r.temp_files)} temp files{flag}"
            )
            for issue in issues:
                print(f"      - {issue}")

    if all_strays:
        print(f"\nStray files (not acc_*/temp_*, ignored — e.g. .DS_Store): {len(all_strays)}")
        for s in all_strays:
            print(f"  - {s}")

    print("\n" + "=" * 70)
    print("Delimiter irregularities (not corruption — dataset ships inconsistently;")
    print("ingestion code must autodetect delimiter per file, not assume comma)")
    print("=" * 70)
    if delimiter_notes:
        for note in delimiter_notes:
            print(f"  {note}")
    else:
        print("  none — all files comma-delimited")

    print("\n" + "=" * 70)
    print("Test_set vs Validation_Set (truncation / RUL ground-truth check)")
    print("=" * 70)
    for note in check_test_is_prefix_of_validation(data_dir):
        print(f"  {note}")

    print("\n" + "=" * 70)
    if total_shape_issues == 0:
        print(f"PASS — {len(all_reports)} bearing directories, all files checked, no shape/gap issues found.")
    else:
        print(f"FOUND {total_shape_issues} shape issue(s) across {len(all_reports)} bearing directories — see above.")
    print("=" * 70)


if __name__ == "__main__":
    main()
