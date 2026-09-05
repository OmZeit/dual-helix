#!/usr/bin/env python3
"""Move selected experiment directories into an archive and optionally remove checkpoints."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def _within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def archive_runs(
    experiments_root: Path,
    *,
    archive_name: str,
    includes: list[str],
    delete_checkpoints: bool,
) -> dict:
    root = experiments_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Experiments root does not exist: {root}")
    if root.name != "experiments":
        raise ValueError(f"Refusing to archive from a directory not named 'experiments': {root}")
    if not archive_name or archive_name in {".", ".."} or Path(archive_name).name != archive_name:
        raise ValueError("archive_name must be one plain directory name")

    archive_parent = (root / "archive").resolve()
    archive_dir = (archive_parent / archive_name).resolve()
    if not _within(archive_dir, root):
        raise ValueError(f"Archive target escapes the experiment root: {archive_dir}")
    if archive_dir.exists():
        raise FileExistsError(f"Archive already exists: {archive_dir}")

    requested = list(dict.fromkeys(includes))
    if not requested:
        requested = sorted(
            child.name
            for child in root.iterdir()
            if child.is_dir() and not child.is_symlink() and child.name != "archive"
        )
    if not requested:
        raise ValueError(f"No run directories found under {root}")

    sources: list[Path] = []
    for name in requested:
        if Path(name).name != name or name in {".", "..", "archive"}:
            raise ValueError(f"Invalid run directory name: {name!r}")
        source = (root / name).resolve()
        if not _within(source, root) or source.parent != root:
            raise ValueError(f"Run target escapes the experiment root: {source}")
        if not source.is_dir() or source.is_symlink():
            raise FileNotFoundError(f"Run directory does not exist or is not a real directory: {source}")
        sources.append(source)

    archive_dir.mkdir(parents=True)
    moved: list[str] = []
    for source in sources:
        destination = archive_dir / source.name
        if destination.exists():
            raise FileExistsError(f"Archive destination already exists: {destination}")
        source.rename(destination)
        moved.append(source.name)

    checkpoints = sorted(path for path in archive_dir.rglob("*.pt") if path.is_file())
    deleted = [{"path": path.relative_to(archive_dir).as_posix(), "bytes": path.stat().st_size} for path in checkpoints]
    if delete_checkpoints:
        for path in checkpoints:
            if not _within(path, archive_dir):
                raise RuntimeError(f"Checkpoint resolved outside archive: {path}")
            path.unlink()
    remaining_checkpoints = sorted(path for path in archive_dir.rglob("*.pt") if path.is_file())
    if delete_checkpoints and remaining_checkpoints:
        raise RuntimeError(f"Checkpoint deletion incomplete: {remaining_checkpoints[0]}")

    json_files = sorted(path for path in archive_dir.rglob("*.json") if path.is_file())
    manifest = {
        "schema": "true-dna-run-archive-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_experiments_root": str(root),
        "archive": str(archive_dir),
        "archived_directories": moved,
        "checkpoint_policy": "deleted" if delete_checkpoints else "preserved",
        "deleted_checkpoints": deleted if delete_checkpoints else [],
        "deleted_checkpoint_count": len(deleted) if delete_checkpoints else 0,
        "deleted_checkpoint_bytes": sum(item["bytes"] for item in deleted) if delete_checkpoints else 0,
        "preserved_json_count": len(json_files),
        "preserved_json_files": [path.relative_to(archive_dir).as_posix() for path in json_files],
    }
    manifest_path = archive_dir / "archive_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-root", type=Path, required=True)
    parser.add_argument(
        "--archive-name",
        default=f"previous_runs_{datetime.now(timezone.utc):%Y%m%d}",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Top-level run directory to archive; repeat to select multiple directories",
    )
    parser.add_argument(
        "--delete-checkpoints",
        action="store_true",
        help="Permanently delete only files ending in .pt after moving runs into the archive",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = archive_runs(
        args.experiments_root,
        archive_name=args.archive_name,
        includes=args.include,
        delete_checkpoints=args.delete_checkpoints,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
