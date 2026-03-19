#!/usr/bin/env python3
"""
sync_image_kinematics.py

Synchronize image timestamps with kinematic data from surgical robot recordings.

This module provides functionality to match image frames with their corresponding
kinematic (end-effector position/velocity) data points based on timestamp proximity.
It handles timestamp extraction from filenames, nearest-neighbor matching, outlier
removal, and optional visualization of synchronization quality.

The module is designed as a library to be imported by other scripts in the data
processing pipeline for performing automated temporal alignment.

Data Structure Expected:
    episode_dir/
        left_img_dir/
            frame1756826516968031906_left.jpg
            frame1756826517968031906_left.jpg
            ...
        right_img_dir/
            frame1756826516968031906_right.jpg
            ...
        endo_psm1/
            frame1756826516968031906_psm1.jpg
            ...
        endo_psm2/
            frame1756826516968031906_psm2.jpg
            ...
        ee_csv.csv  # Kinematic data with timestamps

Processing Pipeline:
    1. Extract nanosecond timestamps from image filenames
    2. Load kinematic data with timestamp column
    3. Find nearest kinematic data point for each image
    4. Calculate time differences
    5. Remove outliers beyond threshold
    6. Return in-memory synchronization dictionary for downstream use

Usage:
    # Library usage
    from sync_image_kinematics import process_episode_sync
    result = process_episode_sync(
        episode_path="/data/episode_001",
        camera="left",
        max_time_diff_ms=30.0
    )
    if result['success']:
        valid_files = result['valid_filenames']

Notes:
    - Timestamps are in nanoseconds (Unix epoch format)
    - Uses efficient binary search (np.searchsorted) for matching
    - Handles missing timestamps gracefully (synthetic generation at 30Hz)
    - Outlier removal is bidirectional (both positive and negative differences)
    - Can process any of four camera views: left, right, psm1, psm2
"""

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from surpass_data_collection.logger_config import get_logger

# ---------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------

# Supported camera views and their directory/suffix mappings
CAMERA_CONFIGS: Dict[str, Tuple[str, str]] = {
    "left": ("left_img_dir", "_left"),
    "right": ("right_img_dir", "_right"),
    "psm1": ("endo_psm1", "_psm1"),
    "psm2": ("endo_psm2", "_psm2"),
}

# Default kinematic CSV filename
DEFAULT_CSV_FILENAME: str = "ee_csv.csv"

# Default synchronization threshold in milliseconds
DEFAULT_MAX_TIME_DIFF_MS: float = 30.0

# Synthetic timestamp generation frequency (Hz) when timestamps are missing
SYNTHETIC_TIMESTAMP_FREQ_HZ: int = 30

# Progress reporting interval for large datasets
PROGRESS_REPORT_INTERVAL: int = 1000

# Initialize module logger
logger = get_logger(__name__)


# Pre-compiled regex patterns for timestamp extraction
# New format: frame{seq}_{camera}_{seconds}_{nanoseconds}.jpg
_NEW_FORMAT_RE = re.compile(r"frame\d+_(?:left|right|psm1|psm2)_(\d+)_(\d+)\.jpg")
# Old format: frame{nanosecond_ts}_{camera}.jpg
_OLD_FORMAT_RE = re.compile(r"frame(\d+)_\w+\.jpg")


def extract_timestamp_from_filename(filename: str) -> int:
    """
    Extract nanosecond timestamp from image filename.

    Supports two filename formats:
        Old: 'frame{nanosecond_ts}_{camera}.jpg'
             e.g. 'frame1756826516968031906_left.jpg'
        New: 'frame{seq}_{camera}_{seconds}_{nanoseconds}.jpg'
             e.g. 'frame000000_left_1772838053_710332121.jpg'

    Args:
        filename: Image filename to parse.

    Returns:
        Extracted timestamp in nanoseconds as an integer.

    Raises:
        ValueError: If the filename does not match either expected pattern.
    """
    # Try new format first (more specific pattern)
    new_match = _NEW_FORMAT_RE.search(filename)
    if new_match:
        return int(new_match.group(1)) * 1_000_000_000 + int(new_match.group(2))

    # Fall back to old format
    old_match = _OLD_FORMAT_RE.search(filename)
    if old_match:
        return int(old_match.group(1))

    raise ValueError(f"Could not extract timestamp from filename: {filename}")


def load_image_timestamps(
    image_dir: Union[str, Path], camera_suffix: str = "_left"
) -> List[Tuple[str, int]]:
    """
    Load all image files from a directory and extract their timestamps.

    Uses os.scandir for fast directory traversal and supports both old
    and new filename formats.

    Args:
        image_dir: Directory containing image files.
        camera_suffix: Camera suffix to filter images (e.g. "_left").

    Returns:
        List of (filename, timestamp) tuples, sorted by timestamp.
    """
    image_dir_path: Path = Path(image_dir)

    if not image_dir_path.is_dir():
        logger.warning(f"Image directory not found: {image_dir_path}")
        return []

    image_timestamps: List[Tuple[str, int]] = []
    skipped_count: int = 0

    # os.scandir is faster than glob — single syscall per entry,
    # no full-path string construction needed.
    with os.scandir(image_dir_path) as entries:
        for entry in entries:
            if (
                entry.is_file()
                and entry.name.endswith(".jpg")
                and camera_suffix in entry.name
            ):
                try:
                    timestamp = extract_timestamp_from_filename(entry.name)
                    image_timestamps.append((entry.name, timestamp))
                except ValueError:
                    skipped_count += 1

    if not image_timestamps:
        logger.warning(f"No image files found with suffix '{camera_suffix}' in: {image_dir_path}")

    if skipped_count > 0:
        logger.warning(f"Skipped {skipped_count} files with invalid timestamp format")

    # Sort by timestamp
    image_timestamps.sort(key=lambda x: x[1])

    logger.info(f"Loaded {len(image_timestamps)} image timestamps from {image_dir_path}")
    return image_timestamps


# ---------------------------------------------------------------------
# Kinematic Data Functions
# ---------------------------------------------------------------------


def load_kinematics_data(csv_file: Union[str, Path]) -> pd.DataFrame:
    """
    Load kinematic data from CSV file and ensure timestamp column exists.

    Reads the kinematic CSV file and adds a standardized 'timestamp_ns' column.
    If timestamps are missing, generates synthetic timestamps at 30Hz.

    Args:
        csv_file: Path to CSV file containing kinematic data. Must exist
            and be readable.

    Returns:
        DataFrame with kinematic data and 'timestamp_ns' column containing
        timestamps in nanoseconds.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        pd.errors.EmptyDataError: If the CSV file is empty.
        pd.errors.ParserError: If the CSV file is malformed.

    Timestamp Detection Strategy:
        1. Search for existing timestamp column (contains "time" or "stamp")
        2. If found, use it as 'timestamp_ns'
        3. If not found, generate synthetic timestamps at 30Hz starting from 0

    Notes:
        - Synthetic timestamps assume constant 30Hz sampling rate
        - Synthetic timestamps start at Unix epoch 0
        - Original timestamp columns are preserved
        - Case-insensitive column name matching
    """
    csv_path: Path = Path(csv_file)

    logger.debug(f"Loading kinematics data from: {csv_path}")

    if not csv_path.exists():
        logger.error(f"Kinematics CSV file not found: {csv_path}")
        raise FileNotFoundError(f"Kinematics CSV file not found: {csv_path}")

    try:
        df: pd.DataFrame = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"Failed to read CSV file {csv_path}: {e}", exc_info=True)
        raise

    if df.empty:
        logger.warning(f"Kinematics CSV file is empty: {csv_path}")
        return df

    logger.debug(f"Loaded {len(df)} rows from kinematics CSV")

    # Check for existing timestamp column
    timestamp_cols: List[str] = [
        col
        for col in df.columns
        if "time" in col.lower() or "stamp" in col.lower()
    ]
    
    # Defragment after column addition (avoids PerformanceWarning on wide CSVs)
    df = df.copy()

    if timestamp_cols:
        # Use the first timestamp column found
        timestamp_col: str = timestamp_cols[0]
        df["timestamp_ns"] = df[timestamp_col]
        logger.info(f"Using existing timestamp column: '{timestamp_col}'")
    else:
        # Generate synthetic timestamps at 30Hz
        logger.warning(
            "No timestamp column found in CSV. Generating synthetic timestamps at 30Hz"
        )
        start_time: int = 0
        freq_ns: int = int(1e9 / SYNTHETIC_TIMESTAMP_FREQ_HZ)  # 30 Hz in nanoseconds
        df["timestamp_ns"] = start_time + np.arange(len(df)) * freq_ns
        logger.info(f"Generated {len(df)} synthetic timestamps")

    return df


# ---------------------------------------------------------------------
# Synchronization Functions
# ---------------------------------------------------------------------


def find_nearest_kinematics(
    image_timestamps: List[Tuple[str, int]], kinematics_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Find nearest kinematic data point for each image timestamp (vectorized).

    Uses bulk np.searchsorted to match ALL images to their nearest
    kinematic timestamp in a single vectorized pass — no Python loop.

    Args:
        image_timestamps: Sorted list of (filename, timestamp) tuples.
        kinematics_df: DataFrame with 'timestamp_ns' column.

    Returns:
        DataFrame with columns: image_filename, image_timestamp_ns,
        kinematics_idx, kinematics_timestamp_ns, time_diff_ns, time_diff_ms.
        Empty DataFrame if no kinematic data.
    """
    logger.info(
        f"Finding nearest kinematics for {len(image_timestamps)} images "
        f"in {len(kinematics_df)} kinematic points"
    )

    kin_ts: np.ndarray = kinematics_df["timestamp_ns"].values
    if len(kin_ts) == 0:
        logger.error("Kinematic data has no timestamps")
        return pd.DataFrame()

    # Unzip filenames and timestamps into separate arrays
    filenames = [ft[0] for ft in image_timestamps]
    img_ts = np.array([ft[1] for ft in image_timestamps], dtype=np.int64)

    # Vectorized nearest-neighbor search
    idx = np.searchsorted(kin_ts, img_ts)
    idx = np.clip(idx, 0, len(kin_ts) - 1)
    prev = np.clip(idx - 1, 0, len(kin_ts) - 1)

    diff_curr = np.abs(kin_ts[idx] - img_ts)
    diff_prev = np.abs(kin_ts[prev] - img_ts)
    use_prev = diff_prev < diff_curr

    best_idx = np.where(use_prev, prev, idx)
    best_ts = kin_ts[best_idx]
    time_diff_ns = (img_ts - best_ts).astype(np.int64)

    sync_df = pd.DataFrame({
        "image_filename": filenames,
        "image_timestamp_ns": img_ts,
        "kinematics_idx": best_idx,
        "kinematics_timestamp_ns": best_ts,
        "time_diff_ns": time_diff_ns,
        "time_diff_ms": time_diff_ns / 1e6,
    })

    logger.info(f"Completed synchronization for {len(sync_df)} images")
    return sync_df


def remove_outliers(
    sync_df: pd.DataFrame, max_time_diff_ms: float = DEFAULT_MAX_TIME_DIFF_MS
) -> Tuple[pd.DataFrame, int]:
    """
    Remove outliers where time difference exceeds threshold (vectorized).

    Args:
        sync_df: DataFrame from find_nearest_kinematics.
        max_time_diff_ms: Maximum allowed absolute time difference (ms).

    Returns:
        Tuple of (filtered_df, num_outliers_removed).
    """
    mask = sync_df["time_diff_ms"].abs() <= max_time_diff_ms
    n_outliers = int((~mask).sum())

    if n_outliers > 0:
        outlier_diffs = sync_df.loc[~mask, "time_diff_ms"].abs()
        logger.info(
            f"Removed {n_outliers} outliers "
            f"(range: {outlier_diffs.min():.2f} - {outlier_diffs.max():.2f} ms)"
        )
    else:
        logger.debug("No outliers found")

    filtered = sync_df.loc[mask].reset_index(drop=True)
    logger.info(f"Filtered results: {len(filtered)} kept, {n_outliers} removed")

    return filtered, n_outliers


def get_valid_image_filenames(sync_df: pd.DataFrame) -> List[str]:
    """
    Extract list of valid image filenames from sync DataFrame.

    Args:
        sync_df: Filtered sync DataFrame.

    Returns:
        List of image filenames.
    """
    filenames = sync_df["image_filename"].tolist()
    logger.debug(f"Extracted {len(filenames)} valid filenames")
    return filenames


# ---------------------------------------------------------------------
# High-Level Processing Functions
# ---------------------------------------------------------------------


def process_episode_sync(
    episode_path: Union[str, Path],
    camera: str = "left",
    csv_filename: str = DEFAULT_CSV_FILENAME,
    max_time_diff_ms: float = DEFAULT_MAX_TIME_DIFF_MS,
) -> Dict[str, Any]:
    """
    Process a single episode to synchronize images and kinematics.

    This is the main entry point for programmatic use. It encapsulates
    the complete synchronization pipeline, allowing other modules to
    import and use it without subprocess overhead.

    Args:
        episode_path: Path to episode directory containing image folders
            and kinematic CSV.
        camera: Camera view to analyze. Must be one of: "left", "right",
            "psm1", "psm2". Default is "left".
        csv_filename: Name of kinematic CSV file. Default is "ee_csv.csv".
        max_time_diff_ms: Maximum time difference threshold for outlier
            removal in milliseconds. Default is 30.0.

    Returns:
        Dictionary containing synchronization results:
            - success: bool - Whether processing completed successfully
            - valid_filenames: List[str] - Filenames passing sync threshold
            - sync_df: pd.DataFrame - Filtered synchronization results
            - sync_output_dir: Optional[Path] - Where results were saved
            - num_valid_images: int - Count of valid synchronized images
            - outliers_removed: int - Count of outliers filtered out
            - error: str - Error message if success is False

    Processing Pipeline:
        1. Validate camera configuration
        2. Load image timestamps
        3. Load kinematic data
        4. Synchronize via nearest-neighbor matching
        5. Remove outliers beyond threshold
        6. Optionally save results and generate plots

    Notes:
        - Designed for library import (no subprocess overhead)
        - Returns structured result dict for easy error handling
        - Gracefully handles missing data (returns success=False)
        - Can be used in batch processing with save_results=False
        - Thread-safe for parallel processing
    """
    episode_path = Path(episode_path)

    logger.info("=" * 70)
    logger.info(f"Processing episode: {episode_path.name}")
    logger.info(f"Camera: {camera}")
    logger.info(f"Max time diff: {max_time_diff_ms} ms")
    logger.info("=" * 70)

    # Validate camera configuration
    if camera not in CAMERA_CONFIGS:
        error_msg: str = (
            f"Unknown camera: '{camera}'. "
            f"Valid options: {list(CAMERA_CONFIGS.keys())}"
        )
        logger.error(error_msg)
        return {"success": False, "error": error_msg}

    # Get camera-specific paths
    image_dir_name, camera_suffix = CAMERA_CONFIGS[camera]
    image_dir: Path = episode_path / image_dir_name
    csv_file: Path = episode_path / csv_filename

    # Validate paths exist
    if not image_dir.exists():
        error_msg = f"Image directory not found: {image_dir}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}

    if not csv_file.exists():
        error_msg = f"Kinematic CSV not found: {csv_file}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}

    try:
        # Load image timestamps
        image_timestamps: List[Tuple[str, int]] = load_image_timestamps(
            image_dir, camera_suffix
        )

        if not image_timestamps:
            error_msg = "No images found with valid timestamps"
            logger.warning(error_msg)
            return {"success": False, "error": error_msg}

        # Load kinematic data
        kinematics_df: pd.DataFrame = load_kinematics_data(csv_file)

        if kinematics_df.empty:
            error_msg = "Kinematic data is empty"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}

        # Perform vectorized synchronization
        sync_df: pd.DataFrame = find_nearest_kinematics(
            image_timestamps, kinematics_df
        )

        # Remove outliers (vectorized mask)
        filtered_df, n_outliers = remove_outliers(sync_df, max_time_diff_ms)

        # Extraction in-memory valid filenames
        valid_filenames = get_valid_image_filenames(filtered_df)

        # Prepare return dictionary
        result: Dict[str, Any] = {
            "success": True,
            "valid_filenames": valid_filenames,
            "sync_df": filtered_df,
            "sync_output_dir": None,
            "num_valid_images": len(valid_filenames),
            "outliers_removed": n_outliers,
        }

        logger.info(f"Processing completed successfully")
        logger.info(f"Valid images: {result['num_valid_images']}")
        logger.info(f"Outliers removed: {result['outliers_removed']}")

        return result

    except Exception as e:
        error_msg: str = f"Processing failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"success": False, "error": error_msg}