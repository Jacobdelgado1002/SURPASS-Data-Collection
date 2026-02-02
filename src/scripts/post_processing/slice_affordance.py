#!/usr/bin/env python3
"""
slice_affordance.py

Slice surgical robot session data into action-based episodes using JSON annotations.

This script processes post-processed annotation data to extract action segments
from full surgical sessions. It uses timestamp-based alignment to ensure semantic
consistency when slicing data from different dataset versions (raw vs. filtered).

The script handles:
    - JSON annotation parsing with affordance_range extraction
    - Timestamp-based frame alignment between reference and source datasets
    - Multi-modal data slicing (4 camera views + kinematic CSV)
    - Parallel processing for efficiency
    - Optional frame name normalization
    - Hardlink support for fast copying

Data Structure Expected:
    post_process/
        cautery_tissue1_session_name_left_video/
            annotation/
                action_001.json  # Contains affordance_range
                action_002.json
                ...
    
    cautery/  # Reference dataset (for timestamps)
        cautery_tissue#1/
            session_name/
                left_img_dir/
                    frame1756826516968031906_left.jpg
                    ...
                right_img_dir/
                endo_psm1/
                endo_psm2/
                ee_csv.csv
    
    source_dataset/  # Dataset to slice (may differ from reference)
        tissue_1/
            session_name/
                left_img_dir/  # Same timestamps as reference
                right_img_dir/
                endo_psm1/
                endo_psm2/
                ee_csv.csv

Output Structure:
    dataset_sliced/
        tissue_1/
            1_grasp/  # Mapped from "grasp" action
                episode_001/
                    left_img_dir/
                    right_img_dir/
                    endo_psm1/
                    endo_psm2/
                    ee_csv.csv
                episode_002/
                    ...
            2_dissect/  # Mapped from "dissect" action
                episode_001/
                    ...

Processing Pipeline:
    1. Discover post-process annotation directories
    2. Parse JSON files to extract affordance_range (start/end frame indices)
    3. Map frame indices to timestamps using reference dataset
    4. Find corresponding frames in source dataset via binary search
    5. Copy sliced data (images + CSV) to organized output structure
    6. Optionally normalize frame names for consistency

Usage:
    # Basic slicing from raw cautery data
    python3 slice_affordance.py

    # Slice from filtered dataset with custom paths
    python3 slice_affordance.py --source_dataset_dir filtered_data \
        --out_dir sliced_episodes

    # Dry run to preview planned episodes
    python3 slice_affordance.py --dry_run

    # Use hardlinks for fast copying (same filesystem)
    python3 slice_affordance.py --hardlink

    # Parallel processing with 8 workers
    python3 slice_affordance.py --episode-workers 8 --workers 16

    # Automatically normalize frame names after slicing
    python3 slice_affordance.py --normalize

Notes:
    - Timestamp-based alignment ensures consistency across dataset versions
    - Reference dataset provides ground truth timestamps
    - Source dataset can be raw, filtered, or any processed version
    - Hardlink support requires source and destination on same filesystem
    - CSV slicing preserves headers and handles index-based extraction
    - Action names are mapped to numbered subdirectories (1_grasp, 2_dissect)
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

from logger_config import get_logger
from reformat_data import run_reformat_data


# ---------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------

# Regular expression for extracting frame numbers/timestamps
FRAME_RE = re.compile(r"frame(\d+)")

# Mapping from raw action names to numbered output subdirectories
# This ensures consistent ordering in the output structure
ACTION_SUBDIRS: Dict[str, str] = {
    "grasp": "1_grasp",
    "dissect": "2_dissect",
}

# Default number of threads for file copying per episode
DEFAULT_COPY_WORKERS: int = 8

# Default number of parallel episodes to process
DEFAULT_EPISODE_WORKERS: int = 4

# Post-process directory naming pattern
POST_PROCESS_PATTERN: str = r"cautery_tissue(\d+)_(.+)_left_video$"

# Camera modality configurations
CAMERA_MODALITIES: List[Tuple[str, str, str]] = [
    ("left_img_dir", "_left.jpg", "left"),
    ("right_img_dir", "_right.jpg", "right"),
    ("endo_psm1", "_psm1.jpg", "psm1"),
    ("endo_psm2", "_psm2.jpg", "psm2"),
]

# Kinematic CSV filename
KINEMATIC_CSV_NAME: str = "ee_csv.csv"

# Default output directory
DEFAULT_OUTPUT_DIR: str = "dataset_sliced"

# Default post-process directory
DEFAULT_POST_PROCESS_DIR: str = "post_process"

# Default cautery directory (reference)
DEFAULT_CAUTERY_DIR: str = "cautery"

# Initialize module logger
logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Directory and File Utilities
# ---------------------------------------------------------------------


def ensure_dir(path: Path) -> None:
    """
    Create a directory and its parents if they do not exist.

    This is a safe wrapper around mkdir that handles race conditions
    when multiple processes attempt to create the same directory.

    Args:
        path: Directory path to create.

    Returns:
        None. Directory is created on disk.

    Notes:
        - Thread-safe (exist_ok=True)
        - Process-safe when used with exist_ok
        - Creates parent directories as needed
    """
    path.mkdir(parents=True, exist_ok=True)


def extract_timestamp(filename: str) -> int:
    """
    Extract nanosecond timestamp from frame filename.

    Parses filenames following the pattern 'frame{timestamp}_{camera}.jpg'
    where timestamp is a Unix epoch timestamp in nanoseconds.

    Args:
        filename: Image filename to parse. Expected format like
            'frame1756826516968031906_left.jpg'.

    Returns:
        Extracted timestamp in nanoseconds as integer.

    Raises:
        ValueError: If filename does not contain a timestamp pattern.

    Notes:
        - Uses regex pattern to find frame number
        - Assumes timestamp is the numeric part after 'frame'
        - Case-sensitive matching
    """
    match = FRAME_RE.search(filename)
    if match:
        timestamp: int = int(match.group(1))
        logger.debug(f"Extracted timestamp {timestamp} from {filename}")
        return timestamp
    raise ValueError(f"Could not extract timestamp from filename: {filename}")


def _frame_key(filename: str) -> Union[int, str]:
    """
    Generate sorting key for frame filenames.

    Extracts numeric frame index/timestamp for proper chronological sorting.
    Falls back to filename string if no number found.

    Args:
        filename: Filename to generate sort key for.

    Returns:
        Integer key if number found, otherwise original filename string.

    Notes:
        - Ensures natural sorting (frame2 before frame10)
        - Used internally by list_sorted_frames()
        - Returns string fallback for non-frame files
    """
    match = FRAME_RE.search(filename)
    if match:
        return int(match.group(1))
    return filename


def list_sorted_frames(src_dir: Path, suffix: str) -> List[str]:
    """
    List and sort frame filenames in a directory.

    Scans directory for files with specified suffix and sorts them by
    frame number/timestamp for chronological processing.

    Args:
        src_dir: Directory to search for frame files.
        suffix: File suffix to filter by (e.g., '_left.jpg', '_psm1.jpg').

    Returns:
        Sorted list of filenames. Empty list if directory doesn't exist
        or contains no matching files.

    Notes:
        - Returns empty list (not error) if directory missing
        - Only includes files (ignores subdirectories)
        - Sorts using natural ordering via _frame_key()
        - Case-sensitive suffix matching
    """
    if not src_dir.is_dir():
        logger.debug(f"Directory not found: {src_dir}")
        return []

    files: List[str] = [
        f.name for f in src_dir.iterdir() if f.is_file() and f.name.endswith(suffix)
    ]

    files.sort(key=_frame_key)
    logger.debug(f"Found {len(files)} files in {src_dir} with suffix {suffix}")

    return files


# ---------------------------------------------------------------------
# Discovery Functions
# ---------------------------------------------------------------------


def find_sessions(post_process_dir: Path) -> Iterator[Tuple[Path, int, str]]:
    """
    Discover post-process session directories containing annotations.

    Scans the post-process directory for session folders matching the
    expected naming pattern and yields their metadata.

    Args:
        post_process_dir: Root directory containing post-process output
            with annotation subdirectories.

    Yields:
        Tuples of (directory_path, tissue_number, session_name) for each
        discovered session. Empty iterator if directory doesn't exist.

    Session Directory Pattern:
        cautery_tissue{N}_{session_name}_left_video
        - N: Tissue number (integer)
        - session_name: Arbitrary session identifier
        - Must end with '_left_video' suffix

    Notes:
        - Only yields directories matching the pattern
        - Skips files and non-matching directories
        - Tissue number extracted as integer
        - Session name includes all text between tissue# and _left_video
    """
    if not post_process_dir.is_dir():
        logger.warning(f"Post-process directory not found: {post_process_dir}")
        return

    logger.debug(f"Scanning for sessions in: {post_process_dir}")

    for entry in post_process_dir.iterdir():
        if not entry.is_dir():
            continue

        # Match pattern: cautery_tissue<N>_<session_name>_left_video
        match = re.match(POST_PROCESS_PATTERN, entry.name)
        if not match:
            logger.debug(f"Skipping non-matching directory: {entry.name}")
            continue

        tissue: int = int(match.group(1))
        session: str = match.group(2)

        logger.debug(f"Found session: tissue={tissue}, session={session}")
        yield entry, tissue, session


def read_annotation_jsons(annotation_dir: Path) -> Iterator[Path]:
    """
    Recursively find all JSON annotation files in a directory.

    Walks the directory tree and yields paths to all files ending in .json.

    Args:
        annotation_dir: Root directory to search for JSON files.

    Yields:
        Path objects for each discovered JSON file. Empty iterator if
        directory doesn't exist.

    Notes:
        - Recursive search (includes subdirectories)
        - Case-insensitive extension matching
        - Returns empty iterator for missing directories
        - Does not validate JSON content
    """
    if not annotation_dir.is_dir():
        logger.debug(f"Annotation directory not found: {annotation_dir}")
        return

    logger.debug(f"Scanning for JSON files in: {annotation_dir}")
    json_count: int = 0

    for root, _, files in os.walk(annotation_dir):
        for filename in files:
            if filename.lower().endswith(".json"):
                json_count += 1
                yield Path(root) / filename

    logger.debug(f"Found {json_count} JSON files")


# ---------------------------------------------------------------------
# CSV Processing Functions
# ---------------------------------------------------------------------


def slice_ee_csv(
    src_csv: Path, out_csv: Path, start_idx: int, end_idx: int
) -> Tuple[int, str]:
    """
    Slice kinematic CSV to include only rows within specified range.

    Extracts a subset of rows from the CSV based on index range. Handles
    header detection and preserves it in the output. Uses 0-based indexing
    for data rows (excluding header).

    Args:
        src_csv: Path to source CSV file containing kinematic data.
        out_csv: Path where sliced CSV should be written.
        start_idx: Starting row index (0-based, inclusive).
        end_idx: Ending row index (0-based, inclusive).

    Returns:
        Tuple of (rows_written, status) where:
            - rows_written: Number of data rows written (excludes header)
            - status: One of 'ok', 'missing', 'empty', or 'error'

    Header Detection:
        - Attempts to parse first cell as integer
        - If successful: No header, first row is data
        - If fails: First row is header

    Notes:
        - Creates output directory if needed
        - Clamps indices to valid range [0, max]
        - Returns 'missing' if source CSV doesn't exist
        - Returns 'empty' if source CSV has no rows
        - Returns 'error' on exception (logged with stack trace)
    """
    logger.debug(
        f"Slicing CSV: {src_csv} -> {out_csv}, "
        f"range: {start_idx}-{end_idx}"
    )

    if not src_csv.exists():
        logger.warning(f"Source CSV not found: {src_csv}")
        return 0, "missing"

    # Clamp indices to valid range
    start: int = max(0, start_idx)
    end: int = max(0, end_idx)
    rows_written: int = 0

    try:
        with src_csv.open("r", newline="", encoding="utf-8") as fin:
            reader = csv.reader(fin)

            # Read first row
            try:
                first_row = next(reader)
            except StopIteration:
                logger.warning(f"Source CSV is empty: {src_csv}")
                return 0, "empty"

            # Detect header
            header: Optional[List[str]] = None
            try:
                int(first_row[0])
                # First row is data (no header)
                data_iter = (r for r in ([first_row] + list(reader)))
                logger.debug("CSV has no header (first row is data)")
            except (ValueError, IndexError):
                # First row is likely header
                header = first_row
                data_iter = reader
                logger.debug(f"CSV header detected: {header}")

            # Write sliced CSV
            ensure_dir(out_csv.parent)

            with out_csv.open("w", newline="", encoding="utf-8") as fout:
                writer = csv.writer(fout)

                # Write header if present
                if header is not None:
                    writer.writerow(header)

                # Write data rows in range
                for i, row in enumerate(data_iter):
                    if i < start:
                        continue
                    if i > end:
                        break
                    writer.writerow(row)
                    rows_written += 1

        logger.info(f"Sliced {rows_written} CSV rows to: {out_csv}")

    except Exception as e:
        logger.error(
            f"Failed to slice CSV {src_csv} -> {out_csv}: {e}",
            exc_info=True,
        )
        return rows_written, "error"

    return rows_written, "ok"


# ---------------------------------------------------------------------
# File Operations
# ---------------------------------------------------------------------


def copy_or_link(
    src: Path, dst: Path, use_hardlink: bool = False, overwrite: bool = False
) -> bool:
    """
    Copy or hardlink a file from source to destination.

    Attempts to create a hardlink first (if enabled), falling back to
    regular copy if hardlink fails. Handles existing destinations and
    creates parent directories as needed.

    Args:
        src: Source file path. Must exist and be a file.
        dst: Destination file path. Parent directory will be created.
        use_hardlink: If True, attempts hardlink before copying.
            Hardlinks only work on same filesystem.
        overwrite: If True, overwrites existing destination file.
            If False, returns success for existing files.

    Returns:
        True if operation succeeded, False if it failed.

    Notes:
        - Returns True for already-existing files when overwrite=False
        - Logs failures at debug level (non-critical)
    """
    try:
        # If file exists and overwrite not requested, consider success
        if dst.exists() and not overwrite:
            logger.debug(f"Destination exists, skipping: {dst}")
            return True

        # Ensure parent directory exists
        ensure_dir(dst.parent)

        # Attempt hardlink if requested
        if use_hardlink:
            try:
                if dst.exists():
                    dst.unlink()
                os.link(src, dst)
                logger.debug(f"Hardlinked: {src} -> {dst}")
                return True
            except OSError as e:
                # Fallback to copy (e.g., cross-device link error)
                logger.debug(f"Hardlink failed ({e}), falling back to copy")

        # Copy file (fallback or default)
        if dst.exists() and overwrite:
            dst.unlink()

        shutil.copy2(src, dst)
        logger.debug(f"Copied: {src} -> {dst}")
        return True

    except Exception as e:
        logger.debug(f"Failed to copy/link {src} -> {dst}: {e}")
        return False


# ---------------------------------------------------------------------
# Episode Planning Functions
# ---------------------------------------------------------------------


def plan_episodes(
    post_dir: Path,
    cautery_dir: Path,
    out_dir: Path,
    source_dataset_dir: Optional[Path] = None,
) -> List[Tuple[Path, Path, Path, Path, int, int]]:
    """
    Scan annotations and plan episode extraction.

    Discovers all annotation files, parses affordance ranges, and generates
    a plan for which data should be extracted into which output directories.

    Args:
        post_dir: Root directory of post-process annotations containing
            session subdirectories.
        cautery_dir: Root directory of raw cautery data. Used as reference
            for timestamp alignment.
        out_dir: Output root directory for sliced dataset.
        source_dataset_dir: Root directory of dataset to slice. If None,
            uses cautery_dir as source.

    Returns:
        List of planning tuples, each containing:
            (annotation_path, ref_session_dir, src_session_dir,
             dst_base_dir, start_frame, end_frame)

    Planning Logic:
        1. Find all post-process sessions
        2. For each session, find annotation JSONs
        3. Parse affordance_range from each JSON
        4. Map action name to output subdirectory
        5. Generate sequential episode IDs per (tissue, subtask)
        6. Build destination path preserving hierarchy

    Output Path Structure:
        out_dir/tissue_{N}/{action_subdir}/episode_{XXX}/
        - N: Tissue number
        - action_subdir: Mapped action name (1_grasp, 2_dissect, etc.)
        - XXX: Zero-padded sequential episode ID (001, 002, ...)

    Notes:
        - Episode IDs are sequential per (tissue, action) combination
        - Skips annotations with missing or invalid ranges
        - Logs warnings for problematic annotations
    """
    logger.info("Planning episode extraction")

    planned_episodes: List[Tuple[Path, Path, Path, Path, int, int]] = []

    # Track episode counts per (tissue, subtask) for sequential IDs
    episode_counters: Dict[Tuple[int, str], int] = {}

    # Default source to reference if not specified
    if source_dataset_dir is None:
        source_dataset_dir = cautery_dir
        logger.info("Using cautery_dir as source dataset")

    sessions_found: int = 0

    # Discover sessions
    for post_path, tissue_num, session_name in find_sessions(post_dir):
        sessions_found += 1
        annotation_dir: Path = post_path / "annotation"

        if not annotation_dir.is_dir():
            logger.warning(
                f"No annotation directory for session: {post_path.name}"
            )
            continue

        # Resolve reference session directory (raw cautery data)
        ref_session_dir: Path = (
            cautery_dir / f"cautery_tissue#{tissue_num}" / session_name
        )

        # Resolve source session directory
        if source_dataset_dir == cautery_dir:
            src_session_dir = ref_session_dir
        else:
            # Try converted structure: tissue_N/session_name
            src_session_dir = (
                source_dataset_dir / f"tissue_{tissue_num}" / session_name
            )
            if not src_session_dir.exists():
                # Fallback to raw structure
                src_session_dir = (
                    source_dataset_dir
                    / f"cautery_tissue#{tissue_num}"
                    / session_name
                )

        # Validate directories exist
        if not ref_session_dir.is_dir():
            logger.warning(f"Reference session not found: {ref_session_dir}")
            continue

        if not src_session_dir.is_dir():
            logger.warning(f"Source session not found: {src_session_dir}")
            continue

        # Process annotations
        annotations_found: int = 0

        for ann_path in read_annotation_jsons(annotation_dir):
            annotations_found += 1

            try:
                content: str = ann_path.read_text(encoding="utf-8")
                ann_data: Dict[str, Any] = json.loads(content)
            except Exception as e:
                logger.warning(f"Failed to read JSON {ann_path}: {e}")
                continue

            # Extract affordance range (handle typo in field name)
            affordance = ann_data.get("affordance_range")

            if not affordance:
                logger.warning(f"No affordance_range in {ann_path}")
                continue

            # Parse range values
            try:
                start: int = int(affordance.get("start"))
                end: int = int(affordance.get("end"))
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid affordance range in {ann_path}: {e}")
                continue

            # Extract and map action name
            action: str = ann_data.get("action", "unknown").lower().strip()
            subtask_dir_name: str = ACTION_SUBDIRS.get(action, action)

            # Generate sequential episode ID
            counter_key: Tuple[int, str] = (tissue_num, subtask_dir_name)
            idx: int = episode_counters.get(counter_key, 0) + 1
            episode_counters[counter_key] = idx

            episode_name: str = f"episode_{idx:03d}"

            # Build destination path
            dst_base: Path = (
                out_dir
                / f"tissue_{tissue_num}"
                / subtask_dir_name
                / episode_name
            )

            planned_episodes.append(
                (ann_path, ref_session_dir, src_session_dir, dst_base, start, end)
            )

            logger.debug(
                f"Planned: {dst_base.name} from {ann_path.name}, "
                f"range: {start}-{end}"
            )

    logger.info(
        f"Planning complete: {len(planned_episodes)} episodes from "
        f"{sessions_found} sessions"
    )

    return planned_episodes


# ---------------------------------------------------------------------
# Episode Processing Functions
# ---------------------------------------------------------------------


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
    """
    Process single episode: align timestamps and copy sliced data.

    This is the core processing function that handles:
        1. Timestamp extraction from reference dataset
        2. Binary search to find corresponding frames in source dataset
        3. Multi-threaded file copying for all modalities
        4. CSV slicing for kinematic data

    Args:
        ref_session_dir: Reference session directory (for timestamp alignment).
            Typically raw cautery data with original timestamps.
        src_session_dir: Source session directory (data to slice).
            May be raw, filtered, or processed dataset.
        dst_base: Destination base directory for this episode.
        start_idx: Starting frame index in reference timeline (0-based).
        end_idx: Ending frame index in reference timeline (0-based, inclusive).
        workers: Number of threads for parallel file copying.
        use_hardlink: Whether to attempt hardlinks instead of copying.
        overwrite: Whether to overwrite existing destination files.

    Returns:
        Tuple of (files_copied, files_missing, csv_rows_written).

    Processing Steps:
        1. Load reference frames and extract timestamps
        2. Clamp start/end indices to valid range
        3. Extract start and end timestamps
        4. Load source frames and build timestamp array
        5. Binary search to find source indices matching ref timestamps
        6. Build list of files to copy for all modalities
        7. Execute parallel copy operations
        8. Slice and save kinematic CSV

    Timestamp Alignment:
        - Reference dataset defines ground truth timestamps
        - Source dataset may have different frame counts
        - Binary search finds nearest frames in source
        - Ensures semantic consistency across datasets

    Notes:
        - Returns (0, 0, 0) if reference or source is empty
        - Handles missing modalities gracefully (skips them)
        - Uses ThreadPoolExecutor for I/O-bound copying
        - Clamps indices to prevent out-of-bounds access
    """
    logger.info(f"Processing episode: {dst_base.name}")

    # Step 1: Resolve timestamps from reference dataset
    ref_left_dir: Path = ref_session_dir / "left_img_dir"
    ref_files: List[str] = list_sorted_frames(ref_left_dir, "_left.jpg")

    if not ref_files:
        logger.error(f"Reference directory empty: {ref_left_dir}")
        return 0, 0, 0

    # Clamp indices to valid range
    s_idx_clamped: int = max(0, min(start_idx, len(ref_files) - 1))
    e_idx_clamped: int = max(0, min(end_idx, len(ref_files) - 1))

    logger.debug(
        f"Reference range: {start_idx}-{end_idx}, "
        f"clamped: {s_idx_clamped}-{e_idx_clamped}"
    )

    # Extract timestamps
    try:
        t_start: int = extract_timestamp(ref_files[s_idx_clamped])
        t_end: int = extract_timestamp(ref_files[e_idx_clamped])
    except ValueError as e:
        logger.error(f"Timestamp extraction failed: {e}")
        return 0, 0, 0

    logger.debug(f"Reference timestamps: {t_start} - {t_end}")

    # Step 2: Map to source dataset indices
    src_left_dir: Path = src_session_dir / "left_img_dir"
    src_files: List[str] = list_sorted_frames(src_left_dir, "_left.jpg")

    if not src_files:
        logger.error(f"Source directory empty: {src_left_dir}")
        return 0, 0, 0

    # Build timestamp array for binary search
    try:
        src_timestamps: np.ndarray = np.array(
            [extract_timestamp(f) for f in src_files]
        )
    except ValueError as e:
        logger.error(f"Invalid timestamp format in source: {e}")
        return 0, 0, 0

    # Binary search for corresponding frames in source
    new_start_idx: int = int(np.searchsorted(src_timestamps, t_start, side="left"))
    new_end_idx: int = int(np.searchsorted(src_timestamps, t_end, side="right")) - 1

    # Clamp to valid range
    new_start_idx = max(0, min(new_start_idx, len(src_files) - 1))
    new_end_idx = max(0, min(new_end_idx, len(src_files) - 1))

    logger.info(
        f"Mapped indices: {start_idx}-{end_idx} (ref) -> "
        f"{new_start_idx}-{new_end_idx} (src), "
        f"{new_end_idx - new_start_idx + 1} frames"
    )

    # Step 3: Setup file lists for all modalities
    file_lists: Dict[str, List[str]] = {}
    src_dirs: Dict[str, Path] = {}
    dst_dirs: Dict[str, Path] = {}

    for dir_name, suffix, key in CAMERA_MODALITIES:
        src_dir: Path = src_session_dir / dir_name
        dst_dir: Path = dst_base / dir_name

        src_dirs[key] = src_dir
        dst_dirs[key] = dst_dir

        # Get file list
        if key == "left":
            file_lists[key] = src_files
        else:
            file_lists[key] = list_sorted_frames(src_dir, suffix)

        # Create destination directory
        ensure_dir(dst_dir)

    # Step 4: Build copy task list
    copy_tasks: List[Tuple[Path, Path]] = []
    indices: range = range(new_start_idx, new_end_idx + 1)

    for key, file_list in file_lists.items():
        if not file_list:
            logger.debug(f"No files for modality: {key}")
            continue

        source_dir: Path = src_dirs[key]
        target_dir: Path = dst_dirs[key]

        for i in indices:
            if i < len(file_list):
                fname: str = file_list[i]
                copy_tasks.append((source_dir / fname, target_dir / fname))

    logger.debug(f"Prepared {len(copy_tasks)} copy tasks")

    # Step 5: Execute parallel copy
    copied_count: int = 0
    missing_count: int = 0

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
                    success: bool = future.result()
                    if success:
                        copied_count += 1
                    else:
                        missing_count += 1
                except Exception as e:
                    logger.debug(f"Copy task failed: {e}")
                    missing_count += 1

    # Step 6: Slice kinematic CSV
    ee_csv_src: Path = src_session_dir / KINEMATIC_CSV_NAME
    ee_csv_dst: Path = dst_base / KINEMATIC_CSV_NAME

    ee_rows: int = 0
    if not ee_csv_dst.exists() or overwrite:
        ee_rows, status = slice_ee_csv(
            ee_csv_src, ee_csv_dst, new_start_idx, new_end_idx
        )
        logger.debug(f"CSV slicing: {ee_rows} rows, status: {status}")

    logger.info(
        f"Episode {dst_base.name} complete: {copied_count} files copied, "
        f"{missing_count} missing, {ee_rows} CSV rows"
    )

    return copied_count, missing_count, ee_rows


# ---------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------


def main() -> int:
    """
    Main entry point for command-line execution.

    Parses arguments, plans episode extraction, executes processing in
    parallel, and optionally runs frame normalization post-processing.

    Command-line Arguments:
        --post_process_dir: Post-process annotations directory
        --cautery_dir: Reference raw directory for timestamps
        --source_dataset_dir: Source dataset to slice (optional)
        --out_dir: Output directory for sliced episodes
        --dry_run: Preview planned episodes without execution
        --execute: Execute processing (opposite of --dry_run)
        --workers: File copy threads per episode
        --episode-workers: Number of parallel episodes
        --hardlink: Use hardlinks instead of copying
        --overwrite: Overwrite existing output files
        --normalize: Run frame normalization after slicing

    Exit Codes:
        0: Success (all processing completed)
        1: Error (invalid arguments or processing failure)

    Examples:
        # Basic usage with defaults
        $ python3 slice_affordance.py

        # Custom paths and parallel processing
        $ python3 slice_affordance.py \
            --source_dataset_dir filtered_data \
            --out_dir episodes \
            --episode-workers 8

        # Dry run to preview
        $ python3 slice_affordance.py --dry_run

        # With normalization
        $ python3 slice_affordance.py --normalize

    Processing Flow:
        1. Parse command-line arguments
        2. Plan episodes by scanning annotations
        3. Display plan (dry run) or execute processing
        4. Process episodes in parallel with progress bar
        5. Optionally run frame normalization script

    Notes:
        - Defaults to 4 episode workers (parallel episodes)
        - Workers are distributed between episodes and file copying
        - Progress bar shows real-time copy statistics
        - All errors logged with full context
        - Normalization requires normalize_frame_names.py in same directory
    """
    parser = argparse.ArgumentParser(
        description="Slice dataset into action-based episodes using annotations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    %(prog)s
    %(prog)s --source_dataset_dir filtered_data --out_dir episodes
    %(prog)s --dry_run
    %(prog)s --hardlink --episode-workers 8 --workers 16
    %(prog)s --normalize

    This script uses timestamp-based alignment to slice surgical robot data
    into semantically meaningful action episodes.
            """,
        )

    parser.add_argument(
        "--post_process_dir",
        default=DEFAULT_POST_PROCESS_DIR,
        help=f"Post-process annotations directory (default: {DEFAULT_POST_PROCESS_DIR})",
    )

    parser.add_argument(
        "--cautery_dir",
        default=DEFAULT_CAUTERY_DIR,
        help=f"Reference raw directory for timestamps (default: {DEFAULT_CAUTERY_DIR})",
    )

    parser.add_argument(
        "--source_dataset_dir",
        default=None,
        help="Source dataset to slice (defaults to cautery_dir)",
    )

    parser.add_argument(
        "--out_dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for sliced episodes (default: {DEFAULT_OUTPUT_DIR})",
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Preview planned episodes without execution",
    )

    parser.add_argument(
        "--execute",
        dest="dry_run",
        action="store_false",
        help="Execute processing (opposite of --dry_run)",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_COPY_WORKERS,
        help=f"File copy threads per episode (default: {DEFAULT_COPY_WORKERS})",
    )

    parser.add_argument(
        "--episode-workers",
        type=int,
        default=None,
        help="Number of parallel episodes (default: auto, typically 4)",
    )

    parser.add_argument(
        "--hardlink",
        action="store_true",
        help="Use hardlinks instead of copying (faster, same filesystem)",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )

    parser.add_argument(
        "--reformat",
        action="store_true",
        help="Run reformat_data.py after slicing",
    )

    args = parser.parse_args()

    # Setup paths
    post_dir: Path = Path(args.post_process_dir)
    cautery_dir: Path = Path(args.cautery_dir)
    source_dir: Optional[Path] = (
        Path(args.source_dataset_dir) if args.source_dataset_dir else None
    )
    out_dir: Path = Path(args.out_dir)

    # Log configuration
    logger.info("=" * 70)
    logger.info("Starting episode slicing pipeline")
    logger.info(f"Post-process dir: {post_dir}")
    logger.info(f"Cautery dir (reference): {cautery_dir}")
    logger.info(f"Source dir: {source_dir or cautery_dir}")
    logger.info(f"Output dir: {out_dir}")
    logger.info(f"Workers per episode: {args.workers}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info(f"Hardlink: {args.hardlink}")
    logger.info(f"Overwrite: {args.overwrite}")
    logger.info("=" * 70)

    # Plan episodes
    logger.info("Planning episodes...")
    planned: List[Tuple[Path, Path, Path, Path, int, int]] = plan_episodes(
        post_dir, cautery_dir, out_dir, source_dataset_dir=source_dir
    )

    logger.info(f"Planned {len(planned)} episodes")

    # Dry run mode
    if args.dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN - PLANNED EPISODES")
        print("=" * 60)
        for item in planned:
            ann, ref, src, dst, s, e = item
            print(f"\nDestination: {dst}")
            print(f"  Reference: {ref}")
            print(f"  Source: {src}")
            print(f"  Range: {s}-{e}")
            print(f"  Annotation: {ann}")
        print("=" * 60)
        print(f"Total episodes planned: {len(planned)}")
        print("=" * 60)
        return 0

    # Execution mode
    if not planned:
        logger.warning("No episodes planned. Nothing to process.")
        return 0

    # Determine worker allocation
    episode_workers: int = (
        args.episode_workers
        if args.episode_workers
        else min(DEFAULT_EPISODE_WORKERS, os.cpu_count() or 4)
    )

    # Distribute workers between episodes and copying
    inner_workers: int = max(1, args.workers // episode_workers)

    logger.info(
        f"Executing with {episode_workers} episode workers, "
        f"{inner_workers} copy threads per episode"
    )

    # Track statistics
    total_copied: int = 0
    total_missing: int = 0
    total_csv_rows: int = 0
    episodes_processed: int = 0
    episodes_failed: int = 0

    # Process episodes in parallel
    with ProcessPoolExecutor(max_workers=episode_workers) as executor:
        futures = {}

        for item in planned:
            ann, ref, src, dst, s, e = item

            # Skip if exists and not overwriting
            if dst.exists() and not args.overwrite:
                logger.info(f"Skipping existing episode: {dst.name}")
                continue

            # Submit processing task
            future = executor.submit(
                process_episode,
                ref,
                src,
                dst,
                s,
                e,
                inner_workers,
                args.hardlink,
                args.overwrite,
            )
            futures[future] = dst

        # Process results with progress bar
        with tqdm(total=len(futures), desc="Processing episodes", unit="episode") as pbar:
            for future in as_completed(futures):
                dst_path: Path = futures[future]

                try:
                    copied, missing, csv_rows = future.result()
                    total_copied += copied
                    total_missing += missing
                    total_csv_rows += csv_rows
                    episodes_processed += 1

                    pbar.set_postfix(
                        {
                            "copied": total_copied,
                            "missing": total_missing,
                            "episodes": episodes_processed,
                        }
                    )

                except Exception as e:
                    episodes_failed += 1
                    logger.error(f"Episode {dst_path.name} failed: {e}", exc_info=True)
                    pbar.set_postfix(
                        {
                            "copied": total_copied,
                            "failed": episodes_failed,
                        }
                    )

                pbar.update(1)

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Episodes processed: {episodes_processed}")
    logger.info(f"Episodes failed: {episodes_failed}")
    logger.info(f"Total files copied: {total_copied}")
    logger.info(f"Total files missing: {total_missing}")
    logger.info(f"Total CSV rows: {total_csv_rows}")
    logger.info("=" * 70)

    # Post-processing: Normalize frame names if requested
    if args.reformat:
        logger.info("Starting reformat_data.py...")

        run_reformat_data(
            base_dir=out_dir,
            workers=args.workers,
            episode_workers=args.episode_workers,
            dry_run=args.dry_run,
            execute=args.execute,
            overwrite=args.overwrite,
        )

    return 0

if __name__ == "__main__":
    sys.exit(main())