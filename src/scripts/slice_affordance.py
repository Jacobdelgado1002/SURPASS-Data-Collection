#!/usr/bin/env python3
"""
Slice original session data into per-action episodes using affordance_range.

This script reads JSON annotations to identify action intervals and slices the
corresponding data from a source dataset into organized episodes. It ensures
semantic consistency by mapping frame indices to timestamps.

The output is structured as:
dataset_sliced/
  tissue_N/
    <subtask_action>/
      episode_XXX/
        left_img_dir/
        right_img_dir/
        endo_psm1/
        endo_psm2/
        ee_csv.csv
"""

import argparse
import csv
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Dict, Any, Union

import numpy as np
from tqdm import tqdm

# Configure top-level logger
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s",
)
logger = logging.getLogger("slice_affordance")

# --- Constants ---------------------------------------------------------------

FRAME_RE = re.compile(r"frame(\d+)")

# Mapping from raw action names in JSON to output subdirectory names
ACTION_SUBDIRS = {
    "grasp": "1_grasp",
    "dissect": "2_dissect",
}


# --- Utility Helpers ---------------------------------------------------------

def ensure_dir(path: Path) -> None:
    """Creates a directory and its parents if they do not exist.

    Args:
        path: The directory path to create.
    """
    path.mkdir(parents=True, exist_ok=True)


def extract_timestamp(filename: str) -> int:
    """Extracts the nanosecond timestamp from a frame filename.

    Args:
        filename: Image filename, e.g., 'frame1756826516968031906_left.jpg'.

    Returns:
        The extracted timestamp as an integer.

    Raises:
        ValueError: If the filename does not contain a timestamp pattern.
    """
    match = FRAME_RE.search(filename)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not extract timestamp from filename: {filename}")


def _frame_key(filename: str) -> Union[int, str]:
    """Returns a sorting key for filenames.

    Prioritizes the numeric frame index or timestamp extracted from the filename.

    Args:
        filename: The filename to generate a key for.

    Returns:
        An integer key if a number is found, otherwise the filename string.
    """
    match = FRAME_RE.search(filename)
    if match:
        return int(match.group(1))
    return filename


def list_sorted_frames(src_dir: Path, suffix: str) -> List[str]:
    """Lists and sorts filenames in a directory by frame number/timestamp.

    Args:
        src_dir: The directory to search for files.
        suffix: The suffix to filter files by (e.g., '_left.jpg').

    Returns:
        A sorted list of filenames. Returns an empty list if directory is missing.
    """
    if not src_dir.is_dir():
        return []
    files = [
        f.name for f in src_dir.iterdir() if f.is_file() and f.name.endswith(suffix)
    ]
    files.sort(key=_frame_key)
    return files


def find_sessions(post_process_dir: Path) -> Iterator[Tuple[Path, int, str]]:
    """Discovers post-process session directories.

    Args:
        post_process_dir: The root directory containing post-process output.

    Yields:
        Tuples containing (directory_path, tissue_number, session_name).
    """
    if not post_process_dir.is_dir():
        return
    for entry in post_process_dir.iterdir():
        if not entry.is_dir():
            continue
        # Expected format: cautery_tissue<N>_<session_name>_left_video
        match = re.match(r"cautery_tissue(\d+)_(.+)_left_video$", entry.name)
        if not match:
            continue
        tissue = int(match.group(1))
        session = match.group(2)
        yield entry, tissue, session


def read_annotation_jsons(annotation_dir: Path) -> Iterator[Path]:
    """Recursively yields all JSON annotation files in a directory.

    Args:
        annotation_dir: The directory to search for .json files.

    Yields:
        Paths to discovered JSON files.
    """
    if not annotation_dir.is_dir():
        return
    for root, _, files in os.walk(annotation_dir):
        for filename in files:
            if filename.lower().endswith(".json"):
                yield Path(root) / filename


# --- CSV Slicing -------------------------------------------------------------

def slice_ee_csv(
    src_csv: Path, out_csv: Path, start_idx: int, end_idx: int
) -> Tuple[int, str]:
    """Slices a CSV file to include only rows within the specified range.

    This function attempts to detect a header. If the first cell converts to an
    integer, it assumes no header exists.

    Args:
        src_csv: Path to the source CSV file.
        out_csv: Path to the destination CSV file.
        start_idx: The starting row index (0-indexed, inclusive).
        end_idx: The ending row index (0-indexed, inclusive).

    Returns:
        A tuple of (rows_written, status). Status is one of 'ok', 'missing',
        'empty', or 'error'.
    """
    if not src_csv.exists():
        return 0, "missing"

    start = max(0, start_idx)
    end = max(0, end_idx)
    rows_written = 0

    try:
        with src_csv.open("r", newline="") as fin:
            reader = csv.reader(fin)
            try:
                first_row = next(reader)
            except StopIteration:
                return 0, "empty"

            # Check for header
            header = None
            try:
                int(first_row[0])
                # First row is data
                data_iter = (r for r in ([first_row] + list(reader)))
            except (ValueError, IndexError):
                # First row is likely a header
                header = first_row
                data_iter = reader

            ensure_dir(out_csv.parent)
            with out_csv.open("w", newline="") as fout:
                writer = csv.writer(fout)
                if header is not None:
                    writer.writerow(header)

                for i, row in enumerate(data_iter):
                    if i < start:
                        continue
                    if i > end:
                        break
                    writer.writerow(row)
                    rows_written += 1

    except Exception as e:
        logger.exception("Failed to slice CSV %s -> %s: %s", src_csv, out_csv, e)
        return rows_written, "error"

    return rows_written, "ok"


# --- File Operations ---------------------------------------------------------

def copy_or_link(
    src: Path, dst: Path, use_hardlink: bool = False, overwrite: bool = False
) -> bool:
    """Copies or hardlinks a file from source to destination.

    Args:
        src: Source file path.
        dst: Destination file path.
        use_hardlink: If True, attempts to create a hardlink first.
        overwrite: If True, overwrites the destination if it exists.

    Returns:
        True if the operation succeeded, False otherwise.
    """
    try:
        if dst.exists() and not overwrite:
            return True

        ensure_dir(dst.parent)

        if use_hardlink:
            try:
                if dst.exists():
                    dst.unlink()
                os.link(src, dst)
                return True
            except OSError:
                # Fallback to copy if link fails (e.g., cross-device)
                pass

        if dst.exists() and overwrite:
            dst.unlink()

        shutil.copy2(src, dst)
        return True

    except Exception:
        logger.debug("Failed to copy/link %s -> %s", src, dst, exc_info=True)
        return False


# --- Core Logic -------------------------------------------------------------

def plan_episodes(
    post_dir: Path,
    cautery_dir: Path,
    out_dir: Path,
    source_dataset_dir: Optional[Path] = None,
) -> List[Tuple[Path, Path, Path, Path, int, int]]:
    """Scans annotations and plans the episodes to create.

    Args:
        post_dir: Root directory of post-process annotations.
        cautery_dir: Root directory of raw cautery data (reference).
        out_dir: Output directory for the sliced dataset.
        source_dataset_dir: Root directory of the dataset to slice.
                            If None, defaults to cautery_dir.

    Returns:
        A list of planning tuples:
        (annotation_path, ref_session_dir, src_session_dir, dst_base_dir, start_frame, end_frame)
    """
    planned_episodes = []
    
    # Track episode counts per (tissue, subtask) to generate IDs like episode_001
    episode_counters: Dict[Tuple[int, str], int] = {}

    if source_dataset_dir is None:
        source_dataset_dir = cautery_dir

    for post_path, tissue_num, session_name in find_sessions(post_dir):
        annotation_dir = post_path / "annotation"
        if not annotation_dir.is_dir():
            continue

        ref_session_dir = cautery_dir / f"cautery_tissue#{tissue_num}" / session_name

        # Resolve source directory
        if source_dataset_dir == cautery_dir:
            src_session_dir = ref_session_dir
        else:
            # Assume converted dataset structure: tissue_N/session_name
            src_session_dir = (
                source_dataset_dir / f"tissue_{tissue_num}" / session_name
            )
            if not src_session_dir.exists():
                # Fallback to raw structure
                src_session_dir = (
                    source_dataset_dir / f"cautery_tissue#{tissue_num}" / session_name
                )

        if not ref_session_dir.is_dir():
            logger.warning(
                "Reference session dir not found: %s", ref_session_dir
            )
            continue
        if not src_session_dir.is_dir():
            logger.warning("Source session dir not found: %s", src_session_dir)
            continue

        for ann_path in read_annotation_jsons(annotation_dir):
            try:
                content = ann_path.read_text(encoding="utf-8")
                ann_data = json.loads(content)
            except Exception as e:
                logger.warning("Failed to read JSON %s: %s", ann_path, e)
                continue

            # Extract range
            affordance = ann_data.get("affordance_range") or ann_data.get(
                "afforadance_range"
            )
            if not affordance:
                logger.warning("No affordance_range found in %s", ann_path)
                continue

            try:
                start = int(affordance.get("start"))
                end = int(affordance.get("end"))
            except (ValueError, TypeError):
                logger.warning("Invalid affordance range in %s", ann_path)
                continue

            # Extract action and determine subdirectory
            action = ann_data.get("action", "unknown").lower().strip()
            
            # Use mapped name or fallback to action name itself
            # If not in mapping, we might want to put it in a generic folder or just use the name
            # For now, we use the name directly if not mapped, potentially prefixed if needed.
            # But the requirement suggests specific structure for grasp/dissect.
            subtask_dir_name = ACTION_SUBDIRS.get(action, action)
            
            # Counters are keyed by (tissue, subtask) to give sequential IDs per subtask folder
            counter_key = (tissue_num, subtask_dir_name)
            idx = episode_counters.get(counter_key, 0) + 1
            episode_counters[counter_key] = idx
            
            episode_name = f"episode_{idx:03d}"
            
            # Destination: dataset_sliced/tissue_N/<subtask>/episode_XXX
            dst_base = (
                out_dir
                / f"tissue_{tissue_num}"
                / subtask_dir_name
                / episode_name
            )

            planned_episodes.append(
                (ann_path, ref_session_dir, src_session_dir, dst_base, start, end)
            )

    return planned_episodes


def process_episode(
    ref_session_dir: Path,
    src_session_dir: Path,
    dst_base: Path,
    start_idx: int,
    end_idx: int,
    workers: int,
    use_hardlink: bool,
    overwrite: bool,
) -> Tuple[int, int, int]:
    """Processes a single episode: aligns timestamps and copies data.

    Args:
        ref_session_dir: Raw session directory (for timestamp reference).
        src_session_dir: Target session directory (source of files).
        dst_base: Destination directory for the episode.
        start_idx: Start frame index (in reference timeline).
        end_idx: End frame index (in reference timeline).
        workers: Number of threads for file copying.
        use_hardlink: Whether to use hardlinks.
        overwrite: Whether to overwrite existing files.

    Returns:
        A tuple of (files_copied, files_missing, csv_rows_written).
    """
    # 1. Resolve Timestamps from Reference
    ref_left_dir = ref_session_dir / "left_img_dir"
    ref_files = list_sorted_frames(ref_left_dir, "_left.jpg")

    if not ref_files:
        logger.error("Reference directory empty: %s", ref_left_dir)
        return 0, 0, 0

    s_idx_clamped = max(0, min(start_idx, len(ref_files) - 1))
    e_idx_clamped = max(0, min(end_idx, len(ref_files) - 1))

    try:
        t_start = extract_timestamp(ref_files[s_idx_clamped])
        t_end = extract_timestamp(ref_files[e_idx_clamped])
    except ValueError as e:
        logger.error("Timestamp extraction failed: %s", e)
        return 0, 0, 0

    # 2. Map to Source Indices
    src_left_dir = src_session_dir / "left_img_dir"
    src_files = list_sorted_frames(src_left_dir, "_left.jpg")

    if not src_files:
        logger.error("Source directory empty: %s", src_left_dir)
        return 0, 0, 0

    try:
        src_timestamps = np.array([extract_timestamp(f) for f in src_files])
    except ValueError:
        logger.error("Invalid timestamp format in source: %s", src_left_dir)
        return 0, 0, 0

    # Find closest frames in source corresponding to reference timestamps
    new_start_idx = int(np.searchsorted(src_timestamps, t_start, side="left"))
    new_end_idx = int(np.searchsorted(src_timestamps, t_end, side="right")) - 1

    new_start_idx = max(0, min(new_start_idx, len(src_files) - 1))
    new_end_idx = max(0, min(new_end_idx, len(src_files) - 1))

    # 3. Setup File Lists
    right_src_dir = src_session_dir / "right_img_dir"
    psm1_src_dir = src_session_dir / "endo_psm1"
    psm2_src_dir = src_session_dir / "endo_psm2"
    ee_csv_src = src_session_dir / "ee_csv.csv"

    # Destination paths
    dst_dirs = {
        "left": dst_base / "left_img_dir",
        "right": dst_base / "right_img_dir",
        "psm1": dst_base / "endo_psm1",
        "psm2": dst_base / "endo_psm2",
    }
    ee_csv_dst = dst_base / "ee_csv.csv"

    # Get file lists
    file_lists = {
        "left": src_files,
        "right": list_sorted_frames(right_src_dir, "_right.jpg"),
        "psm1": list_sorted_frames(psm1_src_dir, "_psm1.jpg"),
        "psm2": list_sorted_frames(psm2_src_dir, "_psm2.jpg"),
    }
    
    src_dirs = {
        "left": src_left_dir,
        "right": right_src_dir,
        "psm1": psm1_src_dir,
        "psm2": psm2_src_dir,
    }

    indices = range(new_start_idx, new_end_idx + 1)

    # 4. Execute Copy
    copy_tasks = []
    
    for key, file_list in file_lists.items():
        if not file_list:
            continue
            
        target_dir = dst_dirs[key]
        source_dir = src_dirs[key]
        ensure_dir(target_dir)

        for i in indices:
            if i < len(file_list):
                fname = file_list[i]
                copy_tasks.append((source_dir / fname, target_dir / fname))

    copied_count = 0
    missing_count = 0

    if copy_tasks:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_task = {
                executor.submit(
                    copy_or_link, src, dst, use_hardlink, overwrite
                ): (src, dst)
                for src, dst in copy_tasks
            }
            
            for future in as_completed(future_to_task):
                try:
                    success = future.result()
                    if success:
                        copied_count += 1
                    else:
                        missing_count += 1
                except Exception:
                    missing_count += 1

    # 5. Slice CSV
    ee_rows = 0
    if not ee_csv_dst.exists() or overwrite:
        ee_rows, status = slice_ee_csv(
            ee_csv_src, ee_csv_dst, new_start_idx, new_end_idx
        )

    logger.debug(
        "Episode %s: Copied %d files, %d missing, %d CSV rows.",
        dst_base.name, copied_count, missing_count, ee_rows
    )

    return copied_count, missing_count, ee_rows


def main() -> None:
    """Main entry point for command-line execution."""
    parser = argparse.ArgumentParser(
        description="Slice dataset into action-based episodes."
    )
    parser.add_argument(
        "--post_process_dir",
        default="post_process",
        help="Root directory for post-process annotations (default: post_process)",
    )
    parser.add_argument(
        "--cautery_dir",
        default="cautery",
        help="Reference raw directory for timestamp retrieval (default: cautery)",
    )
    parser.add_argument(
        "--source_dataset_dir",
        default=None,
        help="Source dataset to slice (defaults to cautery_dir if not matched)",
    )
    parser.add_argument(
        "--out_dir",
        default="dataset_sliced",
        help="Output directory for sliced episodes (default: dataset_sliced)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Perform a dry run (list plans without executing)",
    )
    parser.add_argument(
        "--execute",
        dest="dry_run",
        action="store_false",
        help="Execute the plan (opposite of --dry_run)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of file copy threads per episode (default: 8)",
    )
    parser.add_argument(
        "--episode-workers",
        type=int,
        default=None,
        help="Number of parallel episodes (default: auto)",
    )
    parser.add_argument(
        "--hardlink",
        action="store_true",
        help="Use hardlinks instead of copying where possible",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Automatically normalize frame names after slicing (calls normalize_frame_names.py)",
    )

    args = parser.parse_args()

    post_dir = Path(args.post_process_dir)
    cautery_dir = Path(args.cautery_dir)
    source_dir = Path(args.source_dataset_dir) if args.source_dataset_dir else None
    out_dir = Path(args.out_dir)

    planned = plan_episodes(
        post_dir, cautery_dir, out_dir, source_dataset_dir=source_dir
    )

    logger.info("Planned %d episodes.", len(planned))

    if args.dry_run:
        for item in planned:
            ann, ref, src, dst, s, e = item
            print(f"PLAN: {dst}\n  Ref: {ref}\n  Src: {src}\n  Range: {s}-{e}")
        return

    # Execution
    episode_workers = (
        args.episode_workers if args.episode_workers else min(4, os.cpu_count() or 4)
    )
    # Distribute workers: mostly rely on threads since IO bound, but process per episode
    # helps with the CPU heavy searchsorted/csv parts if many.
    inner_workers = max(1, args.workers // episode_workers)

    logger.info(
        "Executing with %d episode workers, %d threads/episode",
        episode_workers,
        inner_workers,
    )

    total_copied = 0
    total_missing = 0

    with ProcessPoolExecutor(max_workers=episode_workers) as executor:
        futures = {}
        for item in planned:
            ann, ref, src, dst, s, e = item
            
            if dst.exists() and not args.overwrite:
                continue

            # Heuristic check for hardlink feasibility on source drive
            use_hardlink = args.hardlink
            
            future = executor.submit(
                process_episode,
                ref,
                src,
                dst,
                s,
                e,
                inner_workers,
                use_hardlink,
                args.overwrite,
            )
            futures[future] = dst

        with tqdm(total=len(futures), desc="Processing") as pbar:
            for future in as_completed(futures):
                dst_path = futures[future]
                try:
                    c, m, _ = future.result()
                    total_copied += c
                    total_missing += m
                except Exception as e:
                    logger.error("Episode %s failed: %s", dst_path.name, e)
                
                pbar.update(1)
                pbar.set_postfix({"copied": total_copied})

    logger.info("Complete. Total copied: %d, Total missing: %d", total_copied, total_missing)

    # Post-processing: Normalize frame names if requested
    if args.normalize:
        logger.info("Starting frame name normalization...")
        
        # Locate normalize_frame_names.py relative to this script
        script_dir = Path(__file__).parent
        normalize_script = script_dir / "normalize_frame_names.py"
        
        if not normalize_script.exists():
            logger.error(
                "normalize_frame_names.py not found at %s. Skipping normalization.",
                normalize_script
            )
        else:
            try:
                # Build command with appropriate arguments
                normalize_cmd = [
                    sys.executable,
                    str(normalize_script),
                    "--data-path", str(out_dir),
                    "--workers", str(args.workers),
                ]
                
                logger.info("Running: %s", " ".join(normalize_cmd))
                
                # Run normalization script as subprocess
                result = subprocess.run(
                    normalize_cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                
                # Log output
                if result.stdout:
                    for line in result.stdout.strip().split("\n"):
                        logger.info("[normalize] %s", line)
                
                logger.info("Frame normalization completed successfully.")
                
            except subprocess.CalledProcessError as e:
                logger.error(
                    "Frame normalization failed with exit code %d: %s",
                    e.returncode,
                    e.stderr,
                )
            except Exception as e:
                logger.error("Unexpected error during normalization: %s", e)


if __name__ == "__main__":
    main()
