"""Download and safely extract the released FEMTO archives from S3/MinIO."""

from __future__ import annotations

import shutil
import stat
from pathlib import Path, PurePosixPath
from zipfile import ZipFile, ZipInfo


ARCHIVE_SPLITS = {
    "training_set.zip": ("Learning_set", "Training_set"),
    "validation_set.zip": ("Full_Test_Set", "Validation_Set"),
    "test_set.zip": ("Test_set", "Test_set"),
}

REQUIRED_SPLITS = tuple(canonical for _, canonical in ARCHIVE_SPLITS.values())


def download_raw_archives(
    *,
    bucket: str,
    destination: Path,
    endpoint_url: str,
    prefix: str = "",
) -> list[Path]:
    """Download the three expected release archives from a MinIO bucket."""
    import boto3

    destination.mkdir(parents=True, exist_ok=True)
    client = boto3.client("s3", endpoint_url=endpoint_url)
    paginator = client.get_paginator("list_objects_v2")
    found: dict[str, str] = {}

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            archive_name = Path(key).name.lower()
            if archive_name not in ARCHIVE_SPLITS:
                continue
            if archive_name in found:
                raise RuntimeError(
                    f"Multiple MinIO objects match {archive_name!r}: "
                    f"{found[archive_name]!r} and {key!r}"
                )
            found[archive_name] = key

    missing = sorted(set(ARCHIVE_SPLITS) - set(found))
    if missing:
        raise FileNotFoundError(
            f"Bucket {bucket!r} is missing required archive(s) under "
            f"prefix {prefix!r}: {', '.join(missing)}"
        )

    downloaded: list[Path] = []
    for archive_name in ARCHIVE_SPLITS:
        local_path = destination / archive_name
        client.download_file(bucket, found[archive_name], str(local_path))
        downloaded.append(local_path)
    return downloaded


def _safe_member_path(destination: Path, member: ZipInfo) -> Path:
    normalized = member.filename.replace("\\", "/")
    relative = PurePosixPath(normalized)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe path in ZIP archive: {member.filename!r}")
    if stat.S_ISLNK(member.external_attr >> 16):
        raise ValueError(f"Symbolic links are not allowed in ZIPs: {member.filename!r}")
    return destination.joinpath(*relative.parts)


def _extract_zip_safely(archive: Path, destination: Path) -> None:
    with ZipFile(archive) as zip_file:
        for member in zip_file.infolist():
            output_path = _safe_member_path(destination, member)
            if member.is_dir():
                output_path.mkdir(parents=True, exist_ok=True)
                continue
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with zip_file.open(member) as source, output_path.open("wb") as target:
                shutil.copyfileobj(source, target)


def extract_and_normalize_archives(
    archives: list[Path], destination: Path
) -> Path:
    """Extract archives and normalize released split names used by the pipeline."""
    archive_lookup = {archive.name.lower(): archive for archive in archives}
    missing = sorted(set(ARCHIVE_SPLITS) - set(archive_lookup))
    if missing:
        raise FileNotFoundError(f"Missing downloaded archive(s): {', '.join(missing)}")

    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    for archive_name in ARCHIVE_SPLITS:
        _extract_zip_safely(archive_lookup[archive_name], destination)

    for source_name, canonical_name in ARCHIVE_SPLITS.values():
        source = destination / source_name
        canonical = destination / canonical_name
        if source_name != canonical_name:
            if canonical.exists():
                raise FileExistsError(
                    f"Both {source_name!r} and {canonical_name!r} were extracted"
                )
            if not source.is_dir():
                raise FileNotFoundError(
                    f"Expected directory {source_name!r} was not found after extraction"
                )
            source.rename(canonical)

    missing_splits = [name for name in REQUIRED_SPLITS if not (destination / name).is_dir()]
    if missing_splits:
        raise FileNotFoundError(
            f"Missing extracted split directories: {', '.join(missing_splits)}"
        )
    return destination
