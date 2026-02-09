#!/usr/bin/env python3
"""
Reformat sliced surgical dataset episodes for training.

This script combines two normalization steps into a single, maintainable pipeline:

1. Rename tissue subfolders
   Example:
       20251217-150034-281409 -> 1_cholecystectomy
       
2. Timestamp normalization
   Converts absolute nanosecond timestamps in `ee_csv.csv` to relative seconds
   starting at 0.0 for each episode.

   Formula:
       (current_timestamp_ns - first_timestamp_ns) / 1e9

2. Frame renaming
   Renames images in camera subdirectories to a consistent sequential format:
       frame000000_left.jpg
       frame000001_left.jpg
       ...

Both operations are:
    - In-place
    - Parallelized per episode
    - Idempotent (safe to re-run)
    - Collision-safe

Episode definition:
    Any directory containing an `ee_csv.csv` file.

Dataset structure example:
    dataset_sliced/tissue_N/action_subdir/episode_XXX/
        ee_csv.csv
        left_img_dir/
        right_img_dir/
        endo_psm1/
        endo_psm2/

Typical CLI usage:
    python reformat_data.py --base-dir dataset_sliced
    python reformat_data.py --base-dir dataset_sliced --rename-folders --new-name cholecystectomy
    python reformat_data.py --base-dir dataset_sliced --workers 12
    python reformat_data.py --timestamps-only
    python reformat_data.py --frames-only

Programmatic usage (recommended for pipelines):

    from reformat_data import run_reformat_data

    run_reformat_data(
        base_dir=Path("dataset_sliced"),
        workers=8,
        normalize_timestamps=True,
        normalize_frames=True,
        rename_folders=True,
        new_name="cholecystectomy",
    )
"""
import argparse
import csv
import os
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from surpass_data_collection.logger_config import get_logger


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

IMAGE_DIRS: dict[str, str] = {
    "left_img_dir": "_left",
    "right_img_dir": "_right",
    "endo_psm1": "_psm1",
    "endo_psm2": "_psm2",
}

VALID_EXTENSIONS: tuple[str, ...] = (".jpg", ".png")

logger = get_logger(__name__)

# ---------------------------------------------------------------------
# Folder Renaming
# ---------------------------------------------------------------------


def rename_tissue_subfolders(base_dir: Path, new_name: str) -> int:
    """
    Rename timestamp-style tissue subfolders sequentially.

    Args:
        base_dir: Directory containing tissue_# folders.
        new_name: Suffix name appended after index.

    Returns:
        Number of folders successfully renamed.

    Notes:
        - Deterministic ordering via lexicographic sort.
        - Skips collisions safely.
    """
    renamed: int = 0

    for tissue_dir in base_dir.iterdir():
        if not (tissue_dir.is_dir() and tissue_dir.name.startswith("tissue_")):
            continue

        timestamp_dirs: List[Path] = [
            p
            for p in tissue_dir.iterdir()
            if p.is_dir() and p.name.startswith("202") and "-" in p.name
        ]

        timestamp_dirs.sort(key=lambda p: p.name)

        for idx, src in enumerate(timestamp_dirs, start=1):
            dst = tissue_dir / f"{idx}_{new_name}"

            if dst.exists():
                logger.warning("Skipping existing folder: %s", dst)
                continue

            src.rename(dst)
            renamed += 1

    logger.info("Renamed %d tissue subfolders", renamed)
    return renamed

# ---------------------------------------------------------------------
# Episode discovery
# ---------------------------------------------------------------------


def find_episodes(root: Path) -> List[Path]:
    """
    Discover all episode directories under a dataset root.

    An episode is defined as any directory containing `ee_csv.csv`.

    Args:
        root: Root dataset directory.

    Returns:
        Sorted list of episode directory paths.
    """
    episodes: List[Path] = []

    for dirpath, _, files in os.walk(root):
        if "ee_csv.csv" in files:
            episodes.append(Path(dirpath))

    return sorted(episodes)


# ---------------------------------------------------------------------
# Timestamp normalization
# ---------------------------------------------------------------------


def normalize_episode_timestamps(episode_path: Path) -> int:
    """
    Normalize timestamps inside a single episode's `ee_csv.csv`.

    Converts nanosecond timestamps to relative seconds starting at 0.

    Args:
        episode_path: Episode directory containing ee_csv.csv.

    Returns:
        rows_processed:Number of rows successfully normalized.

    Notes:
        - Preserves header row automatically
        - Uses streaming CSV read/write
        - Writes to temp file then atomic replace
        - Invalid rows are skipped with warning
    """
    csv_path: Path = episode_path / "ee_csv.csv"

    if not csv_path.exists():
        return 0

    temp_path: Path = csv_path.with_suffix(".tmp")

    rows_processed: int = 0

    with csv_path.open("r", newline="") as fin, temp_path.open(
        "w", newline=""
    ) as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)

        try:
            first_row: List[str] = next(reader)
        except StopIteration:
            return 0

        header: Optional[List[str]] = None
        first_timestamp_ns: Optional[int] = None

        # Detect header by attempting integer conversion
        try:
            first_timestamp_ns = int(first_row[0])
            first_data_row = first_row
        except (ValueError, IndexError):
            header = first_row
            writer.writerow(header)

            first_data_row = next(reader)
            first_timestamp_ns = int(first_data_row[0])

        # First row always becomes 0.0
        first_data_row[0] = "0.0"
        writer.writerow(first_data_row)
        rows_processed += 1

        # Stream remaining rows
        for row in reader:
            try:
                ts_ns: int = int(row[0])
                relative_sec: float = (ts_ns - first_timestamp_ns) / 1e9
                row[0] = f"{relative_sec:.9f}"
                writer.writerow(row)
                rows_processed += 1
            except Exception:
                logger.warning("Skipping invalid timestamp row in %s", csv_path)

    temp_path.replace(csv_path)

    return rows_processed


# ---------------------------------------------------------------------
# Frame renaming
# ---------------------------------------------------------------------


def _sorted_image_files(directory: Path, sort_by: str) -> List[Path]:
    """
    Collect and sort image files deterministically.

    Sorting is necessary to ensure stable frame ordering.

    Args:
        directory: Image directory.
        sort_by: 'name' or 'mtime'.

    Returns:
        files: Sorted list of image paths.
    """
    files: List[Path] = [
        p for p in directory.iterdir() if p.suffix.lower() in VALID_EXTENSIONS
    ]

    if sort_by == "mtime":
        files.sort(key=lambda p: p.stat().st_mtime)
    else:
        files.sort(key=lambda p: p.name)

    return files


def normalize_episode_frames(episode_path: Path, sort_by: str = "name") -> int:
    """
    Rename image frames sequentially within one episode.

    Uses a two-pass rename strategy:
        Pass 1: rename -> temporary UUID names
        Pass 2: rename -> final normalized names

    This avoids collisions when filenames already partially match target pattern.

    Args:
        episode_path: Episode directory.
        sort_by: 'name' or 'mtime'.

    Returns:
        images_renamed: Number of images renamed.
    """
    images_renamed: int = 0

    for subdir, suffix in IMAGE_DIRS.items():
        img_dir: Path = episode_path / subdir

        if not img_dir.is_dir():
            continue

        files: List[Path] = _sorted_image_files(img_dir, sort_by)

        if not files:
            continue

        unique_id: str = uuid.uuid4().hex[:8]
        temp_files: List[Tuple[Path, int, str]] = []

        # Pass 1 — avoid name collisions
        for idx, src in enumerate(files):
            ext: str = src.suffix.lower()
            tmp = img_dir / f"tmp_{unique_id}_{idx:06d}{ext}"
            os.rename(src, tmp)
            temp_files.append((tmp, idx, ext))

        # Pass 2 — final names
        for tmp, idx, ext in temp_files:
            final = img_dir / f"frame{idx:06d}{suffix}{ext}"
            os.replace(tmp, final)
            images_renamed += 1

    return images_renamed


# ---------------------------------------------------------------------
# Per-episode worker
# ---------------------------------------------------------------------

def process_episode(
    episode_path: Path,
    normalize_timestamps: bool,
    normalize_frames: bool,
    sort_by: str,
) -> Tuple[int, int]:
    """
    Process a single episode: normalize timestamps and/or rename frames.
    
    This function is designed to be called by ProcessPoolExecutor for
    parallel processing of multiple episodes.
    
    Args:
        episode_path: Path to the episode directory.
        normalize_timestamps: If True, normalize timestamps in ee_csv.csv.
        normalize_frames: If True, rename image frames sequentially.
        sort_by: Sorting method for frames ('name' or 'mtime').
    
    Returns:
        Tuple of (rows_normalized, images_renamed).
    """
    rows: int = 0
    images: int = 0
    
    if normalize_timestamps:
        rows = normalize_episode_timestamps(episode_path)
    
    if normalize_frames:
        images = normalize_episode_frames(episode_path, sort_by=sort_by)
    
    return rows, images

def run_reformat_data(
    *,
    base_dir: Path,
    workers: Optional[int] = None,
    normalize_timestamps: bool = True,
    normalize_frames: bool = True,
    rename_folders: bool = False,
    new_name: str = "cholecystectomy",
    sort_by: str = "name",
) -> Tuple[int, int]:
    """
    Run reformatting pipeline programmatically across episodes.

    It will optionally rename tissue subfolders, then discover
    episodes and process them in parallel using :func:`process_episode`.

    Args:
        base_dir: Root dataset directory containing tissue_* folders.
        workers: Number of parallel worker processes. If None, defaults to CPU count.
        normalize_timestamps: If True, run timestamp normalization for each episode.
        normalize_frames: If True, run frame renaming for each episode.
        rename_folders: If True, perform tissue subfolder renaming before processing.
        new_name: Suffix to use when renaming timestamp subfolders (e.g. "cholecystectomy").
        sort_by: Sorting key for frames: 'name' or 'mtime'.

    Returns:
        A tuple (rows_normalized, images_renamed).

    Raises:
        ValueError: If base_dir does not exist.
        RuntimeError: If unexpected failures occur during parallel processing.
    """
    # Validate inputs early
    if not base_dir.exists():
        logger.error("Base directory not found: %s", base_dir)
        raise ValueError(f"Base directory not found: {base_dir}")

    # Optionally rename tissue_* timestamp subfolders first
    if rename_folders:
        logger.info("Renaming tissue subfolders under %s using suffix '%s'", base_dir, new_name)
        renamed_count = rename_tissue_subfolders(base_dir, new_name)
        logger.info("Renamed %d subfolders", renamed_count)

    # Discover episodes (directories that contain ee_csv.csv)
    logger.info("Discovering episodes under %s", base_dir)
    episodes: List[Path] = find_episodes(base_dir)
    logger.info("Found %d episodes", len(episodes))

    if not episodes:
        return 0, 0

    # Determine worker count (ensure at least 1)
    worker_count: int = workers or (os.cpu_count() or 4)
    worker_count = max(1, int(worker_count))
    logger.info("Processing with %d workers (normalize_ts=%s, normalize_frames=%s)",
                worker_count, normalize_timestamps, normalize_frames)

    total_rows: int = 0
    total_images: int = 0
    errors: int = 0

    # Submit per-episode tasks using the top-level process_episode (pickleable)
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                process_episode,
                ep,
                normalize_timestamps,
                normalize_frames,
                sort_by,
            ): ep
            for ep in episodes
        }

        # Collect results as they finish
        for fut in as_completed(futures):
            ep = futures[fut]
            try:
                rows, imgs = fut.result()
                total_rows += int(rows)
                total_images += int(imgs)
            except Exception as exc:
                # Log and continue: one episode's failure shouldn't stop the batch
                errors += 1
                logger.exception("Episode %s failed: %s", ep, exc)

    if errors:
        logger.warning("Completed with %d episode errors", errors)

    logger.info("Reformatting complete: %d rows normalized, %d images renamed",
                total_rows, total_images)

    return total_rows, total_images


def main() -> int:
    """
    CLI entrypoint that delegates to :func:`run_reformat_data`.

    This function parses CLI args, performs validation/dry-run handling,
    and calls :func:`run_reformat_data` to do the heavy lifting.
    """
    parser = argparse.ArgumentParser(
        description="Normalize timestamps and frame names for dataset episodes."
    )

    parser.add_argument("--data-path", required=True, type=Path, help="Dataset root (base_dir)")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--sort-by", choices=("name", "mtime"), default="name")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--timestamps-only", action="store_true", help="Only normalize timestamps")
    group.add_argument("--frames-only", action="store_true", help="Only rename frames")

    parser.add_argument("--rename-folders", action="store_true", help="Rename tissue timestamp folders")
    parser.add_argument("--new-name", default="cholecystectomy", help="Suffix used when renaming folders")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without modifying files")

    args = parser.parse_args()

    # Validate dataset path
    if not args.data_path.exists():
        logger.error("Dataset path not found: %s", args.data_path)
        return 1

    # Resolve flags
    normalize_ts = not args.frames_only
    normalize_frames_flag = not args.timestamps_only

    # Dry-run: only display what we would do
    if args.dry_run:
        logger.info("DRY RUN: would run reformat on %s", args.data_path)
        logger.info("  rename_folders=%s, new_name=%s", args.rename_folders, args.new_name)
        logger.info("  normalize_timestamps=%s, normalize_frames=%s", normalize_ts, normalize_frames_flag)
        return 0

    # Run the programmatic entry point and handle returned stats
    try:
        rows, imgs = run_reformat_data(
            base_dir=args.data_path,
            workers=args.workers,
            normalize_timestamps=normalize_ts,
            normalize_frames=normalize_frames_flag,
            rename_folders=args.rename_folders,
            new_name=args.new_name,
            sort_by=args.sort_by,
        )
        logger.info("Finished: rows_normalized=%d, images_renamed=%d", rows, imgs)
        return 0
    except ValueError as ve:
        logger.error("Invalid input: %s", ve)
        return 2
    except Exception as e:
        logger.exception("Unhandled error during reformat: %s", e)
        return 3

if __name__ == "__main__":
    sys.exit(main())