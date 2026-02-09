#!/usr/bin/env python3
"""
sync_image_kinematics.py

Synchronize image timestamps with kinematic data from surgical robot recordings.

This module provides functionality to match image frames with their corresponding
kinematic (end-effector position/velocity) data points based on timestamp proximity.
It handles timestamp extraction from filenames, nearest-neighbor matching, outlier
removal, and optional visualization of synchronization quality.

The module can be used both as a library (imported by other scripts) and as a
standalone CLI tool for analyzing individual episodes.

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
    6. Optionally visualize and save results

Usage:
    # CLI usage for single episode
    python3 sync_image_kinematics.py /path/to/episode --camera left

    # With custom settings
    python3 sync_image_kinematics.py /path/to/episode --camera left \
        --max-time-diff 50.0 --output-dir results

    # Library usage (no subprocess overhead)
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

import argparse
import glob
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
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

# Output subdirectory name for synchronization results
SYNC_OUTPUT_SUBDIR: str = "sync_analysis"

# Plot settings
PLOT_DPI: int = 300
PLOT_FIGSIZE: Tuple[int, int] = (12, 10)

# Progress reporting interval for large datasets
PROGRESS_REPORT_INTERVAL: int = 1000

# Initialize module logger
logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Timestamp Extraction Functions
# ---------------------------------------------------------------------


def extract_timestamp_from_filename(filename: str) -> int:
    """
    Extract nanosecond timestamp from image filename.

    Parses filenames following the pattern 'frame{timestamp}_{camera}.jpg'
    where timestamp is a Unix epoch timestamp in nanoseconds.

    Args:
        filename: Image filename to parse. Should match pattern like
            'frame1756826516968031906_left.jpg'.

    Returns:
        Extracted timestamp in nanoseconds as an integer.

    Raises:
        ValueError: If the filename does not match the expected pattern
            or timestamp cannot be parsed as integer.

    Notes:
        - Case-sensitive matching
    """
    # Pattern to match frame{timestamp}_{camera}.jpg
    pattern: str = r"frame(\d+)_\w+\.jpg"
    match = re.search(pattern, filename)

    if match:
        try:
            timestamp: int = int(match.group(1))
            logger.debug(f"Extracted timestamp {timestamp} from {filename}")
            return timestamp
        except ValueError as e:
            logger.error(f"Failed to parse timestamp as integer from {filename}: {e}")
            raise ValueError(
                f"Could not parse timestamp from filename: {filename}"
            ) from e
    else:
        raise ValueError(f"Could not extract timestamp from filename: {filename}")


def load_image_timestamps(
    image_dir: Union[str, Path], camera_suffix: str = "_left"
) -> List[Tuple[str, int]]:
    """
    Load all image files from a directory and extract their timestamps.

    Scans the specified directory for image files matching the camera suffix,
    extracts timestamps, and returns them sorted chronologically.

    Args:
        image_dir: Directory containing image files. Must be a valid,
            readable directory path.
        camera_suffix: Camera suffix to filter images. Must be one of:
            "_left", "_right", "_psm1", "_psm2". Default is "_left".

    Returns:
        List of tuples (filename, timestamp), sorted by timestamp in
        ascending order. Returns empty list if no valid images found.

    Notes:
        - Only processes .jpg files
        - Invalid filenames are skipped with warning logged
        - Sorting ensures chronological processing
    """
    image_dir_path: Path = Path(image_dir)
    pattern: Path = image_dir_path / f"frame*{camera_suffix}.jpg"

    logger.debug(f"Scanning for images in: {image_dir_path} with suffix: {camera_suffix}")

    # Use glob to find matching files
    image_files: List[str] = glob.glob(str(pattern))

    if not image_files:
        logger.warning(f"No image files found matching pattern: {pattern}")

    image_timestamps: List[Tuple[str, int]] = []
    skipped_count: int = 0

    for img_file in image_files:
        filename: str = os.path.basename(img_file)
        try:
            timestamp: int = extract_timestamp_from_filename(filename)
            image_timestamps.append((filename, timestamp))
        except ValueError as e:
            # Skip files that don't match expected format
            logger.debug(f"Skipping file with invalid format: {filename}")
            skipped_count += 1
            continue

    if skipped_count > 0:
        logger.warning(f"Skipped {skipped_count} files with invalid timestamp format")

    # Sort by timestamp to ensure chronological order
    image_timestamps.sort(key=lambda x: x[1])

    logger.info(
        f"Loaded {len(image_timestamps)} image timestamps from {image_dir_path}"
    )

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
) -> List[Dict[str, Any]]:
    """
    Find the nearest kinematic data point for each image timestamp.

    Uses efficient binary search (np.searchsorted) to find the closest
    kinematic timestamp for each image. Checks both neighbors to ensure
    absolute minimum distance.

    Args:
        image_timestamps: List of (filename, timestamp) tuples from images.
            Should be sorted by timestamp for optimal performance.
        kinematics_df: DataFrame containing kinematic data with 'timestamp_ns'
            column. Should be sorted by timestamp.

    Returns:
        List of dictionaries, one per image, containing:
            - image_filename: str
            - image_timestamp_ns: int
            - kinematics_idx: int (index into kinematics_df)
            - kinematics_timestamp_ns: int
            - time_diff_ns: int (signed difference)
            - time_diff_ms: float (signed difference in milliseconds)

    Algorithm:
        - Uses np.searchsorted for search per image
        - Compares both neighbors (idx and idx-1) for absolute minimum
        - Handles edge cases (first/last kinematic points)

    Notes:
        - Assumes kinematics_timestamps are sorted (not validated)
        - Returns signed time differences (positive = image after kinematic)
        - All timestamps should be in same epoch reference
        - Memory efficient: processes one image at a time
    """
    logger.info(
        f"Finding nearest kinematics for {len(image_timestamps)} images "
        f"in {len(kinematics_df)} kinematic points"
    )

    sync_results: List[Dict[str, Any]] = []
    kinematics_timestamps: np.ndarray = kinematics_df["timestamp_ns"].values

    if len(kinematics_timestamps) == 0:
        logger.error("Kinematic data has no timestamps")
        return sync_results

    # Progress tracking for large datasets
    report_interval: int = PROGRESS_REPORT_INTERVAL
    processed: int = 0

    for filename, img_timestamp in image_timestamps:
        # Binary search to find insertion point
        # idx is such that: kinematics[idx-1] <= img_timestamp < kinematics[idx]
        idx: int = np.searchsorted(kinematics_timestamps, img_timestamp)

        # Determine the nearest neighbor by checking idx and idx-1
        best_idx: int = 0

        if idx == 0:
            # Image timestamp is before all kinematic points
            best_idx = 0
        elif idx == len(kinematics_timestamps):
            # Image timestamp is after all kinematic points
            best_idx = len(kinematics_timestamps) - 1
        else:
            # Compare distances to idx and idx-1
            diff_curr: int = abs(kinematics_timestamps[idx] - img_timestamp)
            diff_prev: int = abs(kinematics_timestamps[idx - 1] - img_timestamp)

            if diff_curr < diff_prev:
                best_idx = idx
            else:
                best_idx = idx - 1

        nearest_timestamp: int = int(kinematics_timestamps[best_idx])
        time_diff_ns: int = int(img_timestamp - nearest_timestamp)
        time_diff_ms: float = float(time_diff_ns / 1e6)

        sync_info: Dict[str, Any] = {
            "image_filename": filename,
            "image_timestamp_ns": int(img_timestamp),
            "kinematics_idx": int(best_idx),
            "kinematics_timestamp_ns": nearest_timestamp,
            "time_diff_ns": time_diff_ns,
            "time_diff_ms": time_diff_ms,
        }
        sync_results.append(sync_info)

        processed += 1
        if processed % report_interval == 0:
            logger.debug(f"Processed {processed}/{len(image_timestamps)} images")

    logger.info(f"Completed synchronization for {len(sync_results)} images")

    return sync_results


def remove_outliers(
    sync_results: List[Dict[str, Any]], max_time_diff_ms: float = DEFAULT_MAX_TIME_DIFF_MS
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Remove outliers where time difference exceeds threshold.

    Filters synchronization results to remove frames where the time difference
    between image and kinematic data exceeds the specified threshold. This
    helps remove poorly synchronized frames from the dataset.

    Args:
        sync_results: List of sync information dictionaries from
            find_nearest_kinematics().
        max_time_diff_ms: Maximum allowed absolute time difference in
            milliseconds. Frames exceeding this are considered outliers.
            Default is 30.0 ms.

    Returns:
        Tuple containing:
            - filtered_results: List of sync dicts within threshold
            - outliers_removed: List of sync dicts exceeding threshold

    Notes:
        - Uses absolute value (both positive and negative differences count)
        - Threshold is inclusive (values == max_time_diff_ms are kept)
        - Preserves original order of results
        - Logs statistics about outlier distribution
    """
    logger.debug(
        f"Removing outliers with threshold: {max_time_diff_ms} ms "
        f"from {len(sync_results)} results"
    )

    filtered_results: List[Dict[str, Any]] = []
    outliers_removed: List[Dict[str, Any]] = []

    for result in sync_results:
        abs_time_diff: float = abs(result["time_diff_ms"])
        if abs_time_diff <= max_time_diff_ms:
            filtered_results.append(result)
        else:
            outliers_removed.append(result)

    # Log outlier statistics
    if outliers_removed:
        outlier_diffs: List[float] = [abs(r["time_diff_ms"]) for r in outliers_removed]
        logger.info(
            f"Removed {len(outliers_removed)} outliers "
            f"(range: {min(outlier_diffs):.2f} - {max(outlier_diffs):.2f} ms)"
        )
    else:
        logger.debug("No outliers found")

    logger.info(
        f"Filtered results: {len(filtered_results)} kept, "
        f"{len(outliers_removed)} removed"
    )

    return filtered_results, outliers_removed


def get_valid_image_filenames(sync_results: List[Dict[str, Any]]) -> List[str]:
    """
    Extract list of valid image filenames from sync results.

    Convenience function to extract just the filenames from synchronization
    results, useful for subsequent filtering operations.

    Args:
        sync_results: List of sync information dictionaries.

    Returns:
        List of image filenames that passed synchronization checks.

    Notes:
        - Preserves original order from sync_results
    """
    filenames: List[str] = [result["image_filename"] for result in sync_results]
    logger.debug(f"Extracted {len(filenames)} valid filenames")
    return filenames


# ---------------------------------------------------------------------
# Visualization Functions
# ---------------------------------------------------------------------


def plot_time_differences(
    sync_results: List[Dict[str, Any]], output_dir: Optional[str] = None
) -> None:
    """
    Plot time differences between images and kinematic data.

    Creates a two-panel visualization:
        1. Time series plot showing temporal evolution of sync differences
        2. Histogram showing distribution of sync differences

    Includes statistical annotations (mean, std, max) and saves to file
    if output directory is specified.

    Args:
        sync_results: List of sync information dictionaries containing
            'time_diff_ms' values.
        output_dir: Directory to save plots. If None, plot is displayed
            but not saved. Directory will be created if it doesn't exist.

    Returns:
        None. Plot is saved to disk or displayed.

    Side Effects:
        - Creates output directory if it doesn't exist
        - Writes PNG file to output_dir/sync_analysis.png
        - Logs plot save location

    Plot Features:
        - Line plot with scatter overlay for temporal view
        - Statistics box with mean, std, max values
        - Histogram with mean line overlay
        - Grid for readability
        - High DPI (300) for publication quality
    """
    if not sync_results:
        logger.warning("No sync results to plot")
        return

    logger.info(f"Generating synchronization plot for {len(sync_results)} results")

    time_diffs_ms: List[float] = [result["time_diff_ms"] for result in sync_results]
    image_indices: List[int] = list(range(len(sync_results)))

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=PLOT_FIGSIZE)

    # Plot 1: Time differences over image sequence
    ax1.plot(image_indices, time_diffs_ms, "b-", alpha=0.7, linewidth=1)
    ax1.scatter(image_indices, time_diffs_ms, c="red", s=20, alpha=0.6)
    ax1.set_xlabel("Image Index")
    ax1.set_ylabel("Time Difference (ms)")
    ax1.set_title("Time Difference Between Images and Nearest Kinematics Data")
    ax1.grid(True, alpha=0.3)

    # Add statistics text box
    mean_diff: float = float(np.mean(time_diffs_ms))
    std_diff: float = float(np.std(time_diffs_ms))
    max_diff: float = float(np.max(np.abs(time_diffs_ms)))

    stats_text: str = (
        f"Mean: {mean_diff:.2f} ms\n"
        f"Std: {std_diff:.2f} ms\n"
        f"Max |diff|: {max_diff:.2f} ms"
    )

    ax1.text(
        0.02,
        0.98,
        stats_text,
        transform=ax1.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    logger.debug(
        f"Statistics - Mean: {mean_diff:.2f} ms, Std: {std_diff:.2f} ms, "
        f"Max: {max_diff:.2f} ms"
    )

    # Plot 2: Histogram of time differences
    ax2.hist(time_diffs_ms, bins=50, alpha=0.7, color="skyblue", edgecolor="black")
    ax2.axvline(
        mean_diff,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_diff:.2f} ms",
    )
    ax2.set_xlabel("Time Difference (ms)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Distribution of Time Differences")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path: str = os.path.join(output_dir, "sync_analysis.png")
        plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches="tight")
        logger.info(f"Plot saved to: {plot_path}")
        plt.close(fig)  # Close to free memory
    else:
        logger.debug("No output directory specified, plot not saved")
        plt.close(fig)


# ---------------------------------------------------------------------
# File I/O Functions
# ---------------------------------------------------------------------


def save_sync_results(sync_results: List[Dict[str, Any]], output_file: str) -> None:
    """
    Save synchronization results to CSV file.

    Converts sync results dictionary to DataFrame and writes to CSV with
    proper formatting.

    Args:
        sync_results: List of sync information dictionaries.
        output_file: Path where CSV should be written. Parent directories
            will be created if they don't exist.

    Returns:
        None. Results are written to disk.

    Raises:
        OSError: If file cannot be written (permissions, disk space).
        pd.errors.*: If DataFrame conversion or CSV writing fails.

    Side Effects:
        - Creates parent directories if needed
        - Overwrites existing file without warning
        - Writes CSV with header row

    Notes:
        - CSV includes all dictionary keys as columns
        - Index is not written to CSV
        - Uses pandas default CSV settings
    """
    logger.debug(f"Saving {len(sync_results)} sync results to: {output_file}")

    # Ensure parent directory exists
    output_path: Path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df: pd.DataFrame = pd.DataFrame(sync_results)
        df.to_csv(output_file, index=False)
        logger.info(f"Saved sync results to: {output_file}")
    except Exception as e:
        logger.error(f"Failed to save sync results to {output_file}: {e}", exc_info=True)
        raise


# ---------------------------------------------------------------------
# High-Level Processing Functions
# ---------------------------------------------------------------------


def process_episode_sync(
    episode_path: Union[str, Path],
    camera: str = "left",
    output_dir: Optional[str] = None,
    csv_filename: str = DEFAULT_CSV_FILENAME,
    max_time_diff_ms: float = DEFAULT_MAX_TIME_DIFF_MS,
    plot: bool = False,
    save_results: bool = True,
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
        output_dir: Directory to save results. If None, creates
            'sync_analysis' subdirectory in episode_path.
        csv_filename: Name of kinematic CSV file. Default is "ee_csv.csv".
        max_time_diff_ms: Maximum time difference threshold for outlier
            removal in milliseconds. Default is 30.0.
        plot: Whether to generate and save visualization plots.
            Default is False (skip plotting for batch processing speed).
        save_results: Whether to save CSV results and filenames to disk.
            Default is True. Set to False for in-memory processing only.

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

        # Perform synchronization
        sync_results: List[Dict[str, Any]] = find_nearest_kinematics(
            image_timestamps, kinematics_df
        )

        # Remove outliers
        filtered_sync_results, outliers = remove_outliers(
            sync_results, max_time_diff_ms
        )

        # Determine output directory
        if save_results:
            if output_dir:
                out_dir: Path = Path(output_dir)
            else:
                out_dir = episode_path / SYNC_OUTPUT_SUBDIR

            out_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory: {out_dir}")

            # Save all results
            save_sync_results(
                sync_results, str(out_dir / "sync_results_original.csv")
            )
            save_sync_results(
                filtered_sync_results, str(out_dir / "sync_results_filtered.csv")
            )

            if outliers:
                save_sync_results(outliers, str(out_dir / "sync_results_outliers.csv"))

            # Save valid filenames list
            valid_filenames: List[str] = get_valid_image_filenames(
                filtered_sync_results
            )
            filenames_path: Path = out_dir / "valid_image_filenames.txt"

            with open(filenames_path, "w") as f:
                for filename in valid_filenames:
                    f.write(f"{filename}\n")

            logger.info(f"Saved valid filenames to: {filenames_path}")

            # Generate plot if requested
            if plot:
                logger.info("Generating synchronization plot")
                plot_time_differences(filtered_sync_results, str(out_dir))
        else:
            # In-memory only mode
            out_dir = None
            valid_filenames = get_valid_image_filenames(filtered_sync_results)
            logger.debug("Skipping file saves (save_results=False)")

        # Prepare return dictionary
        result: Dict[str, Any] = {
            "success": True,
            "valid_filenames": valid_filenames,
            "sync_df": pd.DataFrame(filtered_sync_results),
            "sync_output_dir": out_dir,
            "num_valid_images": len(valid_filenames),
            "outliers_removed": len(outliers),
        }

        logger.info(f"Processing completed successfully")
        logger.info(f"Valid images: {result['num_valid_images']}")
        logger.info(f"Outliers removed: {result['outliers_removed']}")

        return result

    except Exception as e:
        error_msg: str = f"Processing failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"success": False, "error": error_msg}


# ---------------------------------------------------------------------
# Main Entry Point (CLI)
# ---------------------------------------------------------------------


def main() -> int:
    """
    Main entry point for command-line usage.

    Parses command-line arguments and delegates to process_episode_sync()
    for the actual processing. Provides user-friendly CLI interface with
    validation and helpful error messages.

    Command-line Arguments:
        episode_path: Positional argument specifying episode directory. Required.
        --camera: Camera view to analyze. Default: "left".
        --output-dir: Directory to save results. Default: episode/sync_analysis.
        --csv-file: Kinematic CSV filename. Default: "ee_csv.csv".
        --max-time-diff: Maximum time difference in ms. Default: 30.0.

    Exit Codes:
        0: Success (processing completed without errors)
        1: Failure (error during processing)

    Examples:
        # Basic usage
        $ python3 sync_image_kinematics.py /data/episode_001

        # Custom camera and threshold
        $ python3 sync_image_kinematics.py /data/episode_001 --camera right \
            --max-time-diff 50.0

        # Save to custom directory
        $ python3 sync_image_kinematics.py /data/episode_001 \
            --output-dir /results/sync

    Notes:
        - Always generates plots in CLI mode (plot=True)
        - Validates episode path exists before processing
        - Prints summary statistics on completion
        - Full error messages logged and displayed
    """
    parser = argparse.ArgumentParser(
        description="Synchronize images with kinematics data using timestamps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    %(prog)s /data/episode_001
    %(prog)s /data/episode_001 --camera right
    %(prog)s /data/episode_001 --max-time-diff 50.0 --output-dir /results

    Supported cameras: left, right, psm1, psm2
            """,
        )

    parser.add_argument(
        "episode_path",
        help="Path to episode directory containing images and CSV",
    )

    parser.add_argument(
        "--camera",
        default="left",
        choices=list(CAMERA_CONFIGS.keys()),
        help="Camera to analyze (default: left)",
    )

    parser.add_argument(
        "--output_dir",
        help="Directory to save results and plots (default: episode/sync_analysis)",
    )

    parser.add_argument(
        "--csv_file",
        default=DEFAULT_CSV_FILENAME,
        help=f"Name of kinematics CSV file (default: {DEFAULT_CSV_FILENAME})",
    )

    parser.add_argument(
        "--max_time_diff",
        type=float,
        default=DEFAULT_MAX_TIME_DIFF_MS,
        help=f"Maximum allowed time difference in ms (default: {DEFAULT_MAX_TIME_DIFF_MS})",
    )

    args = parser.parse_args()

    # Validate episode path exists
    episode_path: Path = Path(args.episode_path)
    if not episode_path.exists():
        logger.error(f"Episode path does not exist: {episode_path}")
        print(f"Error: Episode path not found: {episode_path}")
        return 1

    logger.info(f"Starting synchronization for: {episode_path}")

    # Delegate to processing function
    result: Dict[str, Any] = process_episode_sync(
        episode_path=episode_path,
        camera=args.camera,
        output_dir=args.output_dir,
        csv_filename=args.csv_file,
        max_time_diff_ms=args.max_time_diff,
        plot=True,  # Always plot in CLI mode
        save_results=True,
    )

    # Handle result
    if result["success"]:
        print("\n" + "=" * 60)
        print("SYNCHRONIZATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Valid images: {result['num_valid_images']}")
        print(f"Outliers removed: {result['outliers_removed']}")
        if result["sync_output_dir"]:
            print(f"Results saved to: {result['sync_output_dir']}")
        print("=" * 60)
        return 0
    else:
        print("\n" + "=" * 60)
        print("SYNCHRONIZATION FAILED")
        print("=" * 60)
        print(f"Error: {result['error']}")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main())