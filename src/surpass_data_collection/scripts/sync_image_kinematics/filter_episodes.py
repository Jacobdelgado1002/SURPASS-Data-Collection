#!/usr/bin/env python3
"""
filter_episodes.py

Filter and copy synchronized episodes from source to destination directory.

This script processes surgical robot datasets by:
    1. Discovering all episode directories in the source dataset
    2. Validating episode structure (presence of images and kinematics)
    3. Running synchronization analysis on each episode
    4. Enforcing strict multi-camera synchronization
    5. Copying only fully synchronized frames to filtered dataset
    6. Filtering kinematic data to match kept frames

The script ensures that every frame in the output has synchronized data
across ALL cameras and corresponding kinematic data within the time threshold.

Data Structure Expected:
    source_dir/
        cautery_tissue_001/
            run_timestamp_1/
                left_img_dir/
                    frame123_left.jpg
                    ...
                right_img_dir/
                    frame456_right.jpg  # Different timestamp OK
                    ...
                endo_psm1/
                endo_psm2/
                ee_csv.csv
            run_timestamp_2/
                ...

Output Structure:
    out_dir/
        cautery_tissue_001/
            run_timestamp_1/
                left_img_dir/
                    frame123_left.jpg  # Kept (all cameras matched)
                    ...
                right_img_dir/
                    frame123_right.jpg  # Renamed to match left timestamp
                    ...
                endo_psm1/
                    frame123_psm1.jpg   # Renamed to match left timestamp
                endo_psm2/
                    frame123_psm2.jpg   # Renamed to match left timestamp
                ee_csv.csv  # Filtered to match kept frames only

Synchronization Strategy:
    - Left camera is the synchronization source (reference)
    - For each valid left frame, find nearest frame in other cameras
    - Frame is kept only if ALL cameras have matches within threshold
    - Secondary camera frames are renamed to match left timestamp
    - Ensures 1:1 correspondence across all modalities

Usage:
    # Basic filtering with defaults
    python3 filter_episodes.py /source /destination

    # Custom synchronization threshold
    python3 filter_episodes.py /source /out --max-time-diff 50.0

    # Require minimum valid images per episode
    python3 filter_episodes.py /source /out --min-images 100

    # Dry run to preview processing
    python3 filter_episodes.py /source /out --dry-run

    # Parallel processing with 8 workers
    python3 filter_episodes.py /source /out --workers 8

    # Use hardlinks for faster processing (same filesystem required)
    python3 filter_episodes.py /source /out --hardlink

Notes:
    - Uses direct import of sync_image_kinematics (no subprocess overhead)
    - Parallel processing with ProcessPoolExecutor for efficiency
    - Progress bar with real-time statistics
    - Graceful degradation (skip problematic episodes, continue processing)
    - Hardlink support for faster copying (when source/out on same filesystem)
    - Maintains original directory structure in destination
"""

import argparse
import bisect
import os
import shutil
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from tqdm import tqdm

from surpass_data_collection.logger_config import get_logger

# Import synchronization module
# Handle multiple possible import locations
try:
    from sync_image_kinematics import (
        extract_timestamp_from_filename,
        process_episode_sync,
    )
except ImportError:
    try:
        current_dir = Path(__file__).parent
        if str(current_dir) not in sys.path:
            sys.path.append(str(current_dir))
        from sync_image_kinematics import (
            extract_timestamp_from_filename,
            process_episode_sync,
        )
    except ImportError:
        print(
            "Error: Could not import sync_image_kinematics. "
            "Ensure it is in the same directory or Python path."
        )
        sys.exit(1)

# ---------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------

# Default synchronization threshold in milliseconds
DEFAULT_MAX_TIME_DIFF_MS: float = 30.0

# Minimum valid images required to keep an episode
DEFAULT_MIN_IMAGES: int = 10

# Maximum number of parallel workers (limit to avoid overload)
MAX_WORKERS: int = 8

# Camera configurations (directory name, suffix, destination name)
CAMERA_CONFIGS: List[Tuple[str, str, str]] = [
    ("left_img_dir", "_left.jpg", "left_img_dir"),
    ("right_img_dir", "_right.jpg", "right_img_dir"),
    ("endo_psm1", "_psm1.jpg", "endo_psm1"),
    ("endo_psm2", "_psm2.jpg", "endo_psm2"),
]

# Kinematic CSV filename
KINEMATIC_CSV_NAME: str = "ee_csv.csv"

# Initialize module logger
logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Episode Discovery Functions
# ---------------------------------------------------------------------


def find_episodes(source_dir: Union[str, Path]) -> List[Path]:
    """
    Recursively find all directories containing valid episode data.

    A valid episode directory must contain both 'left_img_dir' subdirectory
    and 'ee_csv.csv' file. This ensures we only process directories with
    the minimum required data.

    Args:
        source_dir: Path to source directory to search. Must exist and
            be readable.

    Returns:
        Sorted list of Path objects pointing to valid episode directories.
        Empty list if no valid episodes found.

    Raises:
        FileNotFoundError: If source directory does not exist.

    Search Strategy:
        - Walks entire directory tree recursively
        - Checks each directory for required structure
        - Does not descend into identified episode directories
        - Sorts results for deterministic processing order
    """
    source_path: Path = Path(source_dir)

    logger.info(f"Searching for episodes in: {source_path}")

    if not source_path.exists():
        logger.error(f"Source directory not found: {source_dir}")
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    data_dirs: List[Path] = []

    # Walk through all directories
    for root, dirs, files in os.walk(source_path):
        current_path: Path = Path(root)

        # Check if this directory looks like an episode directory
        has_img_dir: bool = (current_path / "left_img_dir").is_dir()
        has_csv: bool = (current_path / KINEMATIC_CSV_NAME).exists()

        if has_img_dir and has_csv:
            data_dirs.append(current_path)
            logger.debug(f"Found valid episode: {current_path.name}")

    data_dirs.sort()
    logger.info(f"Found {len(data_dirs)} episode directories")

    return data_dirs


# ---------------------------------------------------------------------
# Episode Validation Functions
# ---------------------------------------------------------------------


def validate_episode_structure(episode_path: Path) -> Dict[str, Any]:
    """
    Validate that an episode has required structure.

    Checks for presence of left image directory and kinematic CSV file.
    These are the minimum requirements for processing an episode.

    Args:
        episode_path: Path to episode directory to validate.

    Returns:
        Dictionary with validation results:
            - has_left_images: bool - Whether left_img_dir exists with JPGs
            - has_kinematics: bool - Whether ee_csv.csv exists
            - left_img_dir: Optional[Path] - Path to left images if valid
            - kinematics_file: Optional[Path] - Path to CSV if exists
            
    Validation Checks:
        - left_img_dir must exist and be a directory
        - left_img_dir must contain at least one .jpg file
        - ee_csv.csv must exist and be a file
    """
    validation: Dict[str, Any] = {
        "has_left_images": False,
        "has_kinematics": False,
        "left_img_dir": None,
        "kinematics_file": None,
    }

    logger.debug(f"Validating episode structure: {episode_path.name}")

    # Check for left image directory
    left_img_dir: Path = episode_path / "left_img_dir"
    if left_img_dir.exists() and left_img_dir.is_dir():
        # Check if it contains any jpg files
        if any(left_img_dir.glob("*.jpg")):
            validation["has_left_images"] = True
            validation["left_img_dir"] = left_img_dir
            logger.debug(f"Found left images in: {left_img_dir}")
        else:
            logger.debug(f"Left image directory is empty: {left_img_dir}")
    else:
        logger.debug(f"Left image directory not found: {left_img_dir}")

    # Check for kinematic CSV file
    ee_csv: Path = episode_path / KINEMATIC_CSV_NAME
    if ee_csv.exists():
        validation["has_kinematics"] = True
        validation["kinematics_file"] = ee_csv
        logger.debug(f"Found kinematic CSV: {ee_csv}")
    else:
        logger.debug(f"Kinematic CSV not found: {ee_csv}")

    return validation


# ---------------------------------------------------------------------
# Synchronization Functions
# ---------------------------------------------------------------------


def run_sync_analysis_direct(
    episode_path: Path, max_time_diff: float = DEFAULT_MAX_TIME_DIFF_MS
) -> Dict[str, Any]:
    """
    Run synchronization analysis using direct module import.

    Calls process_episode_sync() directly without subprocess overhead.
    Optimized for batch processing by disabling file saves and plotting.

    Args:
        episode_path: Path to episode directory to analyze.
        max_time_diff: Maximum time difference threshold in milliseconds.

    Returns:
        Result dictionary from process_episode_sync() containing:
            - success: bool
            - valid_filenames: List[str]
            - sync_df: pd.DataFrame
            - num_valid_images: int
            - outliers_removed: int
            - error: str (if success=False)
    """
    logger.debug(f"Running sync analysis: {episode_path.name}")

    result: Dict[str, Any] = process_episode_sync(
        episode_path=episode_path,
        camera="left",
        output_dir=None,
        max_time_diff_ms=max_time_diff,
        plot=False,
        save_results=False,
    )

    if result["success"]:
        logger.debug(
            f"Sync successful: {result['num_valid_images']} valid images, "
            f"{result['outliers_removed']} outliers"
        )
    else:
        logger.debug(f"Sync failed: {result.get('error', 'Unknown error')}")

    return result


# ---------------------------------------------------------------------
# Kinematic Filtering Functions
# ---------------------------------------------------------------------


def write_filtered_kinematics(
    dest_episode_dir: Path,
    sync_df: pd.DataFrame,
    validation: Dict[str, Any],
    kept_filenames: List[str],
) -> None:
    """
    Write filtered kinematic CSV corresponding to kept image filenames.

    Filters the original kinematic data to include only rows corresponding
    to the frames that were kept after synchronization and multi-camera
    matching. Preserves 1:1 correspondence between images and kinematics.

    Args:
        dest_episode_dir: Destination episode directory where CSV will be saved.
        sync_df: Synchronization DataFrame from sync_image_kinematics.
            Must contain 'image_filename' and 'kinematics_idx' columns.
        validation: Validation dictionary containing 'kinematics_file' path.
        kept_filenames: List of left-camera filenames that were kept after
            multi-camera synchronization.

    Returns:
        None. Filtered CSV is written to dest_episode_dir/ee_csv.csv.

    Filtering Strategy:
        - Filter sync_df to only kept filenames
        - Extract corresponding kinematic indices
        - Select those rows from original kinematics
        - Reset index for clean output
        - Preserves duplicate indices (1:1 frame-to-kinematic mapping)

    Notes:
        - Handles empty kept_filenames gracefully (no CSV written)
        - Preserves original kinematic column names and types
        - Resets index to sequential integers
    """
    if sync_df.empty:
        logger.debug("Sync DataFrame is empty, skipping kinematic filtering")
        return

    if not kept_filenames:
        logger.debug("No filenames kept, skipping kinematic filtering")
        return

    logger.debug(
        f"Filtering kinematics for {len(kept_filenames)} kept filenames"
    )

    # Filter to only kept filenames
    kept_set: set = set(kept_filenames)
    mask: pd.Series = sync_df["image_filename"].isin(kept_set)
    final_sync_df: pd.DataFrame = sync_df[mask]

    if final_sync_df.empty:
        logger.warning("No matching sync entries for kept filenames")
        return

    # Get kinematic indices to keep
    kinematics_indices: np.ndarray = final_sync_df["kinematics_idx"].values

    # Load original kinematics
    try:
        original_kinematics: pd.DataFrame = pd.read_csv(
            validation["kinematics_file"]
        )
    except Exception as e:
        logger.error(
            f"Failed to load original kinematics: {e}",
            exc_info=True,
        )
        return

    # Filter and reindex
    filtered_kinematics: pd.DataFrame = original_kinematics.loc[
        kinematics_indices
    ].copy()
    filtered_kinematics.reset_index(drop=True, inplace=True)

    # Save filtered CSV
    dest_csv: Path = dest_episode_dir / KINEMATIC_CSV_NAME
    try:
        filtered_kinematics.to_csv(dest_csv, index=False)
        logger.info(
            f"Saved filtered kinematics: {len(filtered_kinematics)} rows to {dest_csv}"
        )
    except Exception as e:
        logger.error(f"Failed to save filtered kinematics: {e}", exc_info=True)
        raise


# ---------------------------------------------------------------------
# Multi-Camera Synchronization Functions
# ---------------------------------------------------------------------


def load_camera_timestamps(
    episode_path: Path, src_name: str, suffix: str
) -> List[Tuple[int, str]]:
    """
    Load timestamps for a secondary camera.

    Scans the camera directory and extracts timestamps from all valid
    image filenames. Returns sorted list for efficient binary search.

    Args:
        episode_path: Path to episode directory.
        src_name: Source directory name (e.g., "right_img_dir").
        suffix: Expected filename suffix (e.g., "_right.jpg").

    Returns:
        List of (timestamp, filename) tuples sorted by timestamp.
        Empty list if directory doesn't exist or has no valid images.

    """
    src_dir: Path = episode_path / src_name
    candidates: List[Tuple[int, str]] = []

    if not src_dir.exists():
        logger.debug(f"Camera directory not found: {src_dir}")
        return candidates

    try:
        # Use scandir for efficiency
        with os.scandir(src_dir) as entries:
            for entry in entries:
                if (
                    entry.is_file()
                    and entry.name.endswith(".jpg")
                    and suffix in entry.name
                ):
                    try:
                        ts: int = extract_timestamp_from_filename(entry.name)
                        candidates.append((ts, entry.name))
                    except ValueError:
                        logger.debug(
                            f"Skipping file with invalid timestamp: {entry.name}"
                        )
                        continue
    except Exception as e:
        logger.error(f"Error scanning camera directory {src_dir}: {e}")
        return candidates

    candidates.sort(key=lambda x: x[0])
    logger.debug(f"Loaded {len(candidates)} timestamps from {src_name}")

    return candidates


def find_best_camera_match(
    left_ts: int,
    candidate_timestamps: List[int],
    candidates: List[Tuple[int, str]],
    max_diff_ns: float,
) -> Optional[str]:
    """
    Find best matching frame from secondary camera for given left timestamp.

    Uses binary search to find candidates, then selects the one with
    minimum time difference within threshold.

    Args:
        left_ts: Left camera timestamp in nanoseconds.
        candidate_timestamps: Sorted list of candidate timestamps.
        candidates: List of (timestamp, filename) tuples.
        max_diff_ns: Maximum allowed difference in nanoseconds.

    Returns:
        Filename of best match if within threshold, None otherwise.

    Algorithm:
        - Binary search to find insertion point
        - Check both neighbors (idx and idx-1)
        - Select absolute minimum distance
        - Return None if minimum exceeds threshold

    Notes:
        - Handles edge cases (empty list, out of bounds)
        - Returns None rather than raising errors
    """
    if not candidate_timestamps:
        return None

    # Binary search for insertion point
    idx: int = bisect.bisect_left(candidate_timestamps, left_ts)

    best_match_file: Optional[str] = None
    best_diff: float = float("inf")

    # Check candidates at idx and idx-1
    check_indices: List[int] = []
    if idx < len(candidates):
        check_indices.append(idx)
    if idx > 0:
        check_indices.append(idx - 1)

    for i in check_indices:
        diff: float = abs(candidates[i][0] - left_ts)
        if diff < best_diff:
            best_diff = diff
            best_match_file = candidates[i][1]

    # Return match only if within threshold
    if best_match_file and best_diff <= max_diff_ns:
        return best_match_file
    else:
        return None


# ---------------------------------------------------------------------
# Episode Copying Functions
# ---------------------------------------------------------------------


def copy_filtered_episode(
    episode_path: Path,
    out_dir: Path,
    sync_result: Dict[str, Any],
    validation: Dict[str, Any],
    source_root: Path,
    use_hardlink: bool = False,
    max_time_diff: float = DEFAULT_MAX_TIME_DIFF_MS,
) -> bool:
    """
    Copy filtered episode data to destination with multi-camera synchronization.

    This function implements strict synchronization: a frame is kept only if
    ALL cameras have matching frames within the time threshold. Secondary
    camera frames are renamed to match the left camera timestamp to ensure
    1:1 correspondence.

    Args:
        episode_path: Source episode path.
        out_dir: Destination root directory.
        sync_result: Result dictionary from sync analysis containing
            'valid_filenames' and 'sync_df'.
        validation: Validation dictionary containing 'kinematics_file' path.
        source_root: Root of source directory for calculating relative paths.
        use_hardlink: If True, uses hardlinks instead of copying (faster,
            requires source and out on same filesystem).
        max_time_diff: Maximum time difference for sync in milliseconds.

    Returns:
        True if at least one fully synchronized frame was copied,
        False otherwise.

    Processing Steps:
        1. Calculate destination path preserving relative structure
        2. Load timestamps for all secondary cameras
        3. For each valid left frame, find matches in all cameras
        4. Keep frame only if ALL cameras have matches within threshold
        5. Copy left frame with original name
        6. Copy matched frames renamed to left timestamp
        7. Filter and save kinematic data

    Synchronization Rules:
        - Left camera is the reference
        - Frame dropped if any camera missing or out of threshold
        - Secondary frames renamed: frame{left_ts}_{camera}.jpg
        - Ensures exact 1:1 correspondence across modalities

    Notes:
        - Gracefully handles missing camera directories
        - Uses hardlinks when possible for speed
        - Creates destination directories as needed
        - Returns False on any error (logged)
    """
    logger.info(f"Copying filtered episode: {episode_path.name}")

    try:
        # Calculate destination path preserving structure
        try:
            rel: Path = episode_path.relative_to(source_root)
        except ValueError:
            # Fallback if source_root is not a parent
            rel = Path(episode_path.name)

        dest_episode_dir: Path = out_dir / rel
        dest_episode_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Destination: {dest_episode_dir}")

        # Valid filenames from left camera (synchronization source)
        left_filenames: List[str] = sync_result["valid_filenames"]

        if not left_filenames:
            logger.warning("No valid left filenames to copy")
            return False

        # Load timestamps for secondary cameras
        camera_indices: Dict[str, List[Tuple[int, str]]] = {}

        for src_name, suffix, dst_name in CAMERA_CONFIGS:
            if src_name == "left_img_dir":
                continue

            candidates: List[Tuple[int, str]] = load_camera_timestamps(
                episode_path, src_name, suffix
            )
            camera_indices[src_name] = candidates

        # Pre-calculate timestamp lists for search efficiency
        camera_timestamp_lists: Dict[str, List[int]] = {
            name: [x[0] for x in candidates]
            for name, candidates in camera_indices.items()
        }

        # Identify fully valid frames (all cameras match)
        fully_valid_frames: List[Tuple[str, Dict[str, str]]] = []
        max_camera_sync_diff_ns: float = max_time_diff * 1e6

        logger.debug(
            f"Matching {len(left_filenames)} left frames across "
            f"{len(camera_indices)} cameras"
        )

        for left_fname in left_filenames:
            try:
                left_ts: int = extract_timestamp_from_filename(left_fname)
            except ValueError:
                logger.debug(f"Skipping invalid left filename: {left_fname}")
                continue

            matches: Dict[str, str] = {}
            drop_frame: bool = False

            # Check each secondary camera for match
            for src_name, _, _ in CAMERA_CONFIGS:
                if src_name == "left_img_dir":
                    continue

                # Skip if camera directory didn't exist
                if src_name not in camera_timestamp_lists:
                    continue

                candidate_timestamps: List[int] = camera_timestamp_lists[src_name]

                if not candidate_timestamps:
                    # Camera exists but empty - drop frame
                    drop_frame = True
                    break

                # Find best match
                best_match: Optional[str] = find_best_camera_match(
                    left_ts,
                    candidate_timestamps,
                    camera_indices[src_name],
                    max_camera_sync_diff_ns,
                )

                if best_match:
                    matches[src_name] = best_match
                else:
                    # No match within threshold - drop frame
                    drop_frame = True
                    break

            if not drop_frame:
                fully_valid_frames.append((left_fname, matches))

        logger.info(
            f"Fully synchronized: {len(fully_valid_frames)}/{len(left_filenames)} frames"
        )

        if not fully_valid_frames:
            logger.warning("No frames passed multi-camera synchronization")
            return False

        # Create camera directories
        for _, _, dst_name in CAMERA_CONFIGS:
            (dest_episode_dir / dst_name).mkdir(exist_ok=True)

        # Copy files
        copied_count: int = 0

        for left_fname, matches in fully_valid_frames:
            # Copy left frame
            src_left: Path = episode_path / "left_img_dir" / left_fname
            dst_left: Path = dest_episode_dir / "left_img_dir" / left_fname

            if not src_left.exists():
                logger.warning(f"Source left frame missing: {src_left}")
                continue

            # Copy/link left frame
            try:
                if use_hardlink:
                    try:
                        if dst_left.exists():
                            dst_left.unlink()
                        os.link(src_left, dst_left)
                    except OSError:
                        # Fallback to copy if hardlink fails
                        shutil.copy2(src_left, dst_left)
                else:
                    shutil.copy2(src_left, dst_left)
            except Exception as e:
                logger.error(f"Failed to copy left frame {left_fname}: {e}")
                continue

            # Copy matched secondary frames (renamed to left timestamp)
            base_name: str = left_fname.replace("_left.jpg", "")

            for src_name, suffix, dst_name in CAMERA_CONFIGS:
                if src_name == "left_img_dir":
                    continue
                if src_name not in matches:
                    continue

                match_fname: str = matches[src_name]
                src_file: Path = episode_path / src_name / match_fname

                # New filename using left timestamp
                new_name: str = f"{base_name}{suffix}"
                dst_file: Path = dest_episode_dir / dst_name / new_name

                if not src_file.exists():
                    logger.warning(f"Source match missing: {src_file}")
                    continue

                try:
                    if use_hardlink:
                        try:
                            if dst_file.exists():
                                dst_file.unlink()
                            os.link(src_file, dst_file)
                        except OSError:
                            shutil.copy2(src_file, dst_file)
                    else:
                        shutil.copy2(src_file, dst_file)
                except Exception as e:
                    logger.error(f"Failed to copy {src_name} frame: {e}")
                    continue

            copied_count += 1

        logger.info(f"Copied {copied_count} fully synchronized frames")

        # Filter and save kinematics
        kept_left_filenames: List[str] = [x[0] for x in fully_valid_frames]

        write_filtered_kinematics(
            dest_episode_dir=dest_episode_dir,
            sync_df=sync_result["sync_df"],
            validation=validation,
            kept_filenames=kept_left_filenames,
        )

        return copied_count > 0

    except Exception as e:
        logger.error(
            f"Failed to copy episode {episode_path.name}: {e}",
            exc_info=True,
        )
        return False


# ---------------------------------------------------------------------
# Episode Processing Functions
# ---------------------------------------------------------------------


def process_single_episode(
    episode_path: Path,
    out_dir: Path,
    sync_script_path: str,  # Kept for API compatibility, unused
    max_time_diff: float,
    min_images: int,
    source_root: Path,
    dry_run: bool,
    use_hardlink: bool = False,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Process a single episode: validate, sync, and copy if successful.

    This is the main processing function called for each episode, typically
    in parallel. It orchestrates validation, synchronization, and copying.

    Args:
        episode_path: Path to episode directory to process.
        out_dir: Destination root directory.
        sync_script_path: Unused (kept for backward compatibility).
        max_time_diff: Maximum time difference for sync in milliseconds.
        min_images: Minimum valid images required to keep episode.
        source_root: Source root for calculating relative paths.
        dry_run: If True, simulates processing without copying files.
        use_hardlink: If True, uses hardlinks instead of copying.
        overwrite: If True, overwrites existing destination directories.

    Returns:
        Dictionary containing processing statistics:
            - processed: bool - Always True
            - valid_structure: bool - Has required files
            - sync_successful: bool - Sync completed without errors
            - copied: bool - Files were copied (or would be in dry run)
            - images_copied: int - Number of images copied
            - skipped_reason: Optional[str] - Reason for skipping
            - name: str - Episode name

    Processing Steps:
        1. Check if destination exists (skip if overwrite=False)
        2. Validate episode structure
        3. Run synchronization analysis
        4. Check if enough valid images
        5. Copy filtered data (if not dry run)

    """
    result_stats: Dict[str, Any] = {
        "processed": True,
        "valid_structure": False,
        "sync_successful": False,
        "copied": False,
        "images_copied": 0,
        "skipped_reason": None,
        "name": episode_path.name,
    }

    logger.debug(f"Processing episode: {episode_path.name}")

    # Calculate destination path
    try:
        rel_path: Path = episode_path.relative_to(source_root)
    except ValueError:
        rel_path = Path(episode_path.name)

    dest_episode_dir: Path = out_dir / rel_path

    # Check for existing destination
    if dest_episode_dir.exists() and not overwrite and not dry_run:
        result_stats["skipped_reason"] = "Destination exists (use --overwrite)"
        logger.debug(f"Skipping existing destination: {dest_episode_dir}")
        return result_stats

    # Validate episode structure
    validation: Dict[str, Any] = validate_episode_structure(episode_path)

    if not validation["has_left_images"] or not validation["has_kinematics"]:
        missing: List[str] = []
        if not validation["has_left_images"]:
            missing.append("left images")
        if not validation["has_kinematics"]:
            missing.append("kinematics CSV")
        result_stats["skipped_reason"] = f"Missing: {', '.join(missing)}"
        logger.debug(f"Invalid structure: {result_stats['skipped_reason']}")
        return result_stats

    result_stats["valid_structure"] = True

    # Run synchronization analysis
    sync_result: Dict[str, Any] = run_sync_analysis_direct(
        episode_path, max_time_diff
    )

    if not sync_result["success"]:
        result_stats["skipped_reason"] = (
            f"Sync failed: {sync_result.get('error', 'Unknown')}"
        )
        logger.debug(f"Sync failed: {result_stats['skipped_reason']}")
        return result_stats

    result_stats["sync_successful"] = True
    num_valid: int = sync_result["num_valid_images"]

    # Check minimum images threshold
    if num_valid < min_images:
        result_stats["skipped_reason"] = f"Only {num_valid} valid images"
        logger.debug(f"Insufficient images: {result_stats['skipped_reason']}")
        return result_stats

    # Copy episode (or simulate in dry run)
    if not dry_run:
        success: bool = copy_filtered_episode(
            episode_path,
            out_dir,
            sync_result,
            validation,
            source_root,
            use_hardlink,
            max_time_diff=max_time_diff,
        )
        if success:
            result_stats["copied"] = True
            result_stats["images_copied"] = num_valid
            logger.info(f"Successfully processed: {episode_path.name}")
        else:
            result_stats["skipped_reason"] = "Copy failed (multi-camera sync)"
            logger.warning(f"Copy failed: {episode_path.name}")
    else:
        # Dry run mode
        result_stats["copied"] = True
        result_stats["images_copied"] = num_valid
        logger.info(f"[DRY RUN] Would copy {num_valid} images from {episode_path.name}")

    return result_stats


# ---------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------


def main() -> int:
    """
    Main entry point for command-line usage.

    Parses arguments, discovers episodes, processes them in parallel,
    and displays comprehensive statistics.

    Command-line Arguments:
        source_dir: Source directory containing episodes (required)
        out_dir: Destination directory for filtered data (required)
        --sync-script: Legacy arg, unused (kept for compatibility)
        --max-time-diff: Sync threshold in ms (default: 30.0)
        --min-images: Minimum images per episode (default: 10)
        --dry-run: Preview without copying (default: False)
        --workers: Number of parallel workers (default: CPU count)
        --hardlink: Use hardlinks instead of copying (default: False)
        --overwrite: Overwrite existing destinations (default: False)

    Exit Codes:
        0: Success (all processing completed)
        1: Error (invalid arguments or source not found)

    Examples:
        # Basic usage
        $ python3 filter_episodes.py /source /out

        # Custom settings
        $ python3 filter_episodes.py /source /out \
            --max-time-diff 50.0 --min-images 100 --workers 8

        # Dry run to preview
        $ python3 filter_episodes.py /source /out --dry-run

    Output:
        - Progress bar with real-time statistics
        - Final summary with counts and skip reasons
        - List of skipped episodes with reasons

    Notes:
        - Uses ProcessPoolExecutor for parallel processing
        - Limits workers to MAX_WORKERS to avoid overload
        - Progress bar shows copied/skipped/images in real-time
        - Gracefully handles Ctrl+C interruption
        - Comprehensive error logging
    """
    parser = argparse.ArgumentParser(
        description="Filter and copy synchronized episodes with multi-camera support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    %(prog)s /source /out
    %(prog)s /source /out --max-time-diff 50.0 --min-images 100
    %(prog)s /source /out --dry-run --workers 8
    %(prog)s /source /out --hardlink --overwrite

    This script enforces strict multi-camera synchronization: frames are kept
    only if ALL cameras have matching frames within the time threshold.
            """,
        )

    parser.add_argument(
        "source_dir",
        help="Source directory containing episodes",
    )

    parser.add_argument(
        "out_dir",
        help="Destination directory for filtered episodes",
    )

    parser.add_argument(
        "--sync_script",
        default="",
        help="Legacy parameter (unused, kept for compatibility)",
    )

    parser.add_argument(
        "--max_time_diff",
        type=float,
        default=DEFAULT_MAX_TIME_DIFF_MS,
        help=f"Maximum time difference in ms (default: {DEFAULT_MAX_TIME_DIFF_MS})",
    )

    parser.add_argument(
        "--min_images",
        type=int,
        default=DEFAULT_MIN_IMAGES,
        help=f"Minimum valid images required (default: {DEFAULT_MIN_IMAGES})",
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Preview processing without copying files",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count, max: 8)",
    )

    parser.add_argument(
        "--hardlink",
        action="store_true",
        help="Use hardlinks instead of copying (faster, requires same filesystem)",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing episodes in destination",
    )

    args = parser.parse_args()

    # Setup paths
    source_dir: Path = Path(args.source_dir)
    out_dir: Path = Path(args.out_dir)

    # Validate source
    if not source_dir.exists():
        logger.error(f"Source directory not found: {source_dir}")
        print(f"Error: Source directory not found: {source_dir}")
        return 1

    # Create destination if not dry run
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Log configuration
    logger.info("=" * 70)
    logger.info("Starting episode filtering pipeline")
    logger.info(f"Source: {source_dir}")
    logger.info(f"Destination: {out_dir}")
    logger.info(f"Max time diff: {args.max_time_diff} ms")
    logger.info(f"Min images: {args.min_images}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info(f"Hardlink: {args.hardlink}")
    logger.info(f"Overwrite: {args.overwrite}")
    logger.info("=" * 70)

    # Print configuration to console
    print(f"Processing episodes from: {source_dir}")
    print(f"Destination directory: {out_dir}")
    print(f"Max time difference: {args.max_time_diff} ms")
    print(f"Min images required: {args.min_images}")
    if args.workers:
        print(f"Workers: {args.workers}")
    if args.dry_run:
        print("[DRY RUN MODE - No files will be copied]")
    print("-" * 60)

    # Find episodes
    try:
        episodes: List[Path] = find_episodes(str(source_dir))
    except FileNotFoundError as e:
        logger.error(str(e))
        print(f"Error: {e}")
        return 1

    if not episodes:
        logger.warning("No episodes found in source directory")
        print("No episodes found in source directory")
        return 0

    # Pre-filter episodes to skip existing destinations
    episodes_to_process: List[Path] = []
    skipped_pre: List[Tuple[str, str]] = []

    for ep in episodes:
        try:
            rel = ep.relative_to(source_dir)
        except ValueError:
            rel = Path(ep.name)

        dest_episode_dir = out_dir / rel

        if dest_episode_dir.exists() and not args.overwrite and not args.dry_run:
            skipped_pre.append((ep.name, "Destination already exists"))
            continue

        episodes_to_process.append(ep)

    logger.info(
        f"Processing {len(episodes_to_process)} episodes "
        f"({len(skipped_pre)} pre-filtered)"
    )

    # Initialize statistics
    stats: Dict[str, Any] = {
        "total_episodes": len(episodes),
        "valid_structure": 0,
        "sync_successful": 0,
        "copied_episodes": 0,
        "total_images_copied": 0,
        "skipped_episodes": skipped_pre.copy(),
    }

    # Determine worker count
    max_procs: int = min(args.workers or os.cpu_count() or 1, MAX_WORKERS)
    logger.info(f"Using {max_procs} parallel workers")

    # Process episodes in parallel
    try:
        with ProcessPoolExecutor(max_workers=max_procs) as executor:
            # Submit all tasks
            futures = {
                executor.submit(
                    process_single_episode,
                    episode,
                    out_dir,
                    "",  # sync_script_path unused
                    args.max_time_diff,
                    args.min_images,
                    source_dir,
                    args.dry_run,
                    args.hardlink,
                    args.overwrite,
                ): episode
                for episode in episodes_to_process
            }

            # Process results with progress bar
            with tqdm(
                total=len(futures),
                desc="Processing episodes",
                unit="episode",
                ncols=100,
            ) as pbar:
                for future in futures:
                    try:
                        result: Dict[str, Any] = future.result()

                        # Update statistics
                        if result["valid_structure"]:
                            stats["valid_structure"] += 1
                        if result["sync_successful"]:
                            stats["sync_successful"] += 1
                        if result["copied"]:
                            stats["copied_episodes"] += 1
                            stats["total_images_copied"] += result["images_copied"]

                        if result["skipped_reason"]:
                            stats["skipped_episodes"].append(
                                (result["name"], result["skipped_reason"])
                            )

                        # Update progress bar
                        pbar.set_postfix(
                            {
                                "copied": stats["copied_episodes"],
                                "skipped": len(stats["skipped_episodes"]),
                                "images": stats["total_images_copied"],
                            }
                        )

                    except Exception as e:
                        episode = futures[future]
                        logger.error(
                            f"Exception processing {episode.name}: {e}",
                            exc_info=True,
                        )
                        stats["skipped_episodes"].append(
                            (episode.name, f"Exception: {str(e)}")
                        )
                        pbar.set_postfix(
                            {
                                "copied": stats["copied_episodes"],
                                "skipped": len(stats["skipped_episodes"]),
                                "error": episode.name[:20],
                            }
                        )
                    finally:
                        pbar.update(1)

    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user (Ctrl+C)")
        print("\n\nProcessing interrupted by user")
        return 1

    # Print final summary
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total episodes found: {stats['total_episodes']}")
    print(f"Episodes with valid structure: {stats['valid_structure']}")
    print(f"Episodes with successful sync: {stats['sync_successful']}")
    print(f"Episodes copied: {stats['copied_episodes']}")
    print(f"Total images copied: {stats['total_images_copied']}")
    print(f"Episodes skipped: {len(stats['skipped_episodes'])}")

    if stats["skipped_episodes"]:
        print("\nSkipped episodes:")
        for episode_name, reason in stats["skipped_episodes"]:
            print(f"  - {episode_name}: {reason}")

    if args.dry_run:
        print("\n[DRY RUN] No files were actually copied.")

    print("=" * 60)

    logger.info("Processing complete")
    logger.info(f"Copied {stats['copied_episodes']} episodes")
    logger.info(f"Total images copied: {stats['total_images_copied']}")

    return 0

if __name__ == "__main__":
    sys.exit(main())