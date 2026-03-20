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

- Timestamp-based alignment ensures consistency across dataset versions
    - Reference dataset provides ground truth timestamps
    - Source dataset can be raw, filtered, or any processed version
    - Action names are mapped to numbered subdirectories (1_grasp, 2_dissect)
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

from surpass_data_collection.logger_config import get_logger


# ---------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------

# Regular expression for extracting frame numbers/timestamps
FRAME_RE = re.compile(r"frame(\d+)")

# New-format regex: frame{seq}_{camera}_{seconds}_{nanoseconds}.jpg
NEW_FORMAT_RE = re.compile(r"frame\d+_(?:left|right|psm1|psm2)_(\d+)_(\d+)\.jpg")

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

# Post-process directory naming patterns
# Old: cautery_tissue{N}_{session}_left_video
POST_PROCESS_PATTERN_OLD: str = r"cautery_tissue(\d+)_(.+)_left_video$"
# New: {collector}_tissue{N}_{timestamp}
POST_PROCESS_PATTERN_NEW: str = r"(\w+)_tissue(\d+)_(.+)$"

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



def extract_timestamp(filename: str) -> int:
    """
    Extract nanosecond timestamp from frame filename.

    Supports two filename formats:
        Old: 'frame{nanosecond_ts}_{camera}.jpg'
        New: 'frame{seq}_{camera}_{seconds}_{nanoseconds}.jpg'

    Args:
        filename: Image filename to parse.

    Returns:
        Extracted timestamp in nanoseconds as integer.

    Raises:
        ValueError: If filename does not contain a valid timestamp pattern.
    """
    # Try new format first (more specific)
    new_match = NEW_FORMAT_RE.search(filename)
    if new_match:
        seconds = int(new_match.group(1))
        nanoseconds = int(new_match.group(2))
        timestamp = seconds * 1_000_000_000 + nanoseconds
        logger.debug(f"Extracted timestamp {timestamp} from {filename} (new format)")
        return timestamp

    # Fall back to old format
    match = FRAME_RE.search(filename)
    if match:
        timestamp = int(match.group(1))
        logger.debug(f"Extracted timestamp {timestamp} from {filename} (old format)")
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

    Supports both old and new filename formats by checking if the suffix
    string appears anywhere in the filename (not just at the end).

    Args:
        src_dir: Directory to search for frame files.
        suffix: File suffix to filter by (e.g., '_left.jpg', '_psm1.jpg').

    Returns:
        Sorted list of filenames. Empty list if directory doesn't exist.
    """
    if not src_dir.is_dir():
        logger.debug(f"Directory not found: {src_dir}")
        return []

    # Use 'suffix in name' instead of 'endswith(suffix)' to support
    # new format where camera suffix is mid-filename
    suffix_bare = suffix.replace(".jpg", "")  # e.g. "_left"
    files: List[str] = [
        f.name for f in src_dir.iterdir()
        if f.is_file() and f.name.endswith(".jpg") and suffix_bare in f.name
    ]

    files.sort(key=_frame_key)
    logger.debug(f"Found {len(files)} files in {src_dir} with suffix {suffix}")

    return files


# ---------------------------------------------------------------------
# Discovery Functions
# ---------------------------------------------------------------------


def find_sessions(
    post_process_dir: Path,
) -> Iterator[Tuple[Path, int, str, Optional[str]]]:
    """
    Discover annotation session directories.

    Supports two naming conventions:
        Old: cautery_tissue{N}_{session}_left_video
        New: {collector}_tissue{N}_{timestamp}

    Yields:
        (directory_path, tissue_number, session_name, collector)
        collector is None for old-format directories.
    """
    if not post_process_dir.is_dir():
        logger.warning(f"Post-process directory not found: {post_process_dir}")
        return

    logger.debug(f"Scanning for sessions in: {post_process_dir}")

    for entry in post_process_dir.iterdir():
        if not entry.is_dir():
            continue

        # Try old pattern first: cautery_tissue{N}_{session}_left_video
        match = re.match(POST_PROCESS_PATTERN_OLD, entry.name)
        if match:
            tissue = int(match.group(1))
            session = match.group(2)
            logger.debug(f"Found old-format session: tissue={tissue}, session={session}")
            yield entry, tissue, session, None
            continue

        # Try new pattern: {collector}_tissue{N}_{timestamp}
        match = re.match(POST_PROCESS_PATTERN_NEW, entry.name)
        if match:
            collector = match.group(1)
            tissue = int(match.group(2))
            session = match.group(3)
            logger.debug(
                f"Found new-format session: collector={collector}, "
                f"tissue={tissue}, session={session}"
            )
            yield entry, tissue, session, collector
            continue

        logger.debug(f"Skipping non-matching directory: {entry.name}")


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
            if filename.lower().endswith(".json") and not filename.lower().endswith("_prompt.json"):
                json_count += 1
                yield Path(root) / filename

    logger.debug(f"Found {json_count} JSON files")




# ---------------------------------------------------------------------
# Episode Planning Functions
# ---------------------------------------------------------------------


def plan_episodes(
    source_data_path: Path,
    annotations_path: Path,
    out_dir: Path,
    source_dataset_dir: Optional[Path] = None,
) -> List[Tuple[Path, Path, Path, Path, int, int]]:
    """
    Scan annotations and plan episode extraction.

    Supports both old (cautery_tissue#N/session) and new
    (collector/Tissue#N/session) data layouts.

    Returns:
        List of (annotation_path, ref_session_dir, src_session_dir,
                 dst_base_dir, start_frame, end_frame).
    """
    logger.info("Planning episode extraction")

    planned_episodes: List[Tuple[Path, Path, Path, Path, int, int]] = []
    episode_counters: Dict[Tuple[str, int, str], int] = {}

    if source_dataset_dir is None:
        source_dataset_dir = source_data_path
        logger.info("Using source_data_path as source dataset")

    sessions_found: int = 0

    for post_path, tissue_num, session_name, collector in find_sessions(annotations_path):
        sessions_found += 1
        annotation_dir: Path = post_path / "annotation"

        if not annotation_dir.is_dir():
            logger.warning(
                f"No annotation directory for session: {post_path.name}"
            )
            continue

        # Resolve reference session directory (raw data)
        if collector is not None:
            # New format: source_data_path / collector / Tissue#N / session
            ref_session_dir = (
                source_data_path / collector / f"Tissue#{tissue_num}" / session_name
            )
        else:
            # Old format: source_data_path / cautery_tissue#N / session
            ref_session_dir = (
                source_data_path / f"cautery_tissue#{tissue_num}" / session_name
            )

        # Resolve source session directory
        if source_dataset_dir == source_data_path:
            src_session_dir = ref_session_dir
        else:
            if collector is not None:
                # New: try collector_tissueN/session, then collector/Tissue#N/session
                src_session_dir = (
                    source_dataset_dir
                    / f"{collector}_tissue{tissue_num}"
                    / session_name
                )
                if not src_session_dir.exists():
                    src_session_dir = (
                        source_dataset_dir
                        / collector
                        / f"Tissue#{tissue_num}"
                        / session_name
                    )
            else:
                # Old: try tissue_N/session, then cautery_tissue#N/session
                src_session_dir = (
                    source_dataset_dir / f"tissue_{tissue_num}" / session_name
                )
                if not src_session_dir.exists():
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

            # Extract affordance range
            affordance = ann_data.get("affordance_range")

            if not affordance:
                logger.warning(f"No affordance_range in {ann_path}")
                continue

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
            counter_key = (collector or "", tissue_num, subtask_dir_name)
            idx: int = episode_counters.get(counter_key, 0) + 1
            episode_counters[counter_key] = idx

            episode_name: str = f"episode_{idx:03d}"

            # Build destination path
            if collector is not None:
                tissue_label = f"{collector}_tissue{tissue_num}"
            else:
                tissue_label = f"tissue_{tissue_num}"

            dst_base: Path = (
                out_dir / tissue_label / subtask_dir_name / episode_name
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

    # Sort deterministically by destination path
    planned_episodes.sort(key=lambda x: str(x[3]))

    return planned_episodes


