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
            sys.path.insert(0, str(current_dir))
        from sync_image_kinematics import (
            extract_timestamp_from_filename,
            process_episode_sync,
        )
    except ImportError as e:
        import traceback
        print(
            "Error: Could not import sync_image_kinematics. "
            "Ensure it is in the same directory or Python path."
        )
        print(f"Root cause: {e}")
        traceback.print_exc()
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


def find_all_camera_matches_vectorized(
    left_timestamps: np.ndarray,
    candidate_timestamps: np.ndarray,
    max_diff_ns: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized multi-camera matching for ALL left timestamps at once.

    Uses numpy searchsorted to find the closest candidate timestamp for
    every left timestamp in a single vectorized pass.

    Args:
        left_timestamps: Array of left-camera timestamps (int64, sorted).
        candidate_timestamps: Array of secondary-camera timestamps (int64, sorted).
        max_diff_ns: Maximum allowed time difference in nanoseconds.

    Returns:
        Tuple of:
            - best_indices: Array of indices into candidate_timestamps for
              each left timestamp (closest match). Invalid entries where
              the match exceeds the threshold are still present but masked.
            - valid_mask: Boolean array. True where the closest match is
              within max_diff_ns threshold.

    Algorithm:
        - np.searchsorted for bulk insertion-point lookup
        - Compare both neighbors (idx and idx-1) vectorized
        - Apply threshold mask

    Notes:
        - Returns empty arrays if candidate_timestamps is empty
    """
    if len(candidate_timestamps) == 0:
        return (
            np.zeros(len(left_timestamps), dtype=np.int64),
            np.zeros(len(left_timestamps), dtype=bool),
        )

    idx = np.searchsorted(candidate_timestamps, left_timestamps)
    idx = np.clip(idx, 0, len(candidate_timestamps) - 1)
    prev = np.clip(idx - 1, 0, len(candidate_timestamps) - 1)

    diff_curr = np.abs(candidate_timestamps[idx] - left_timestamps)
    diff_prev = np.abs(candidate_timestamps[prev] - left_timestamps)

    use_prev = diff_prev < diff_curr
    best_indices = np.where(use_prev, prev, idx)
    best_diffs = np.where(use_prev, diff_prev, diff_curr)

    valid_mask = best_diffs <= max_diff_ns

    return best_indices, valid_mask


# ---------------------------------------------------------------------
# Episode Copying Functions
# ---------------------------------------------------------------------


def copy_filtered_episode(
    episode_path: Path,
    out_dir: Path,
    filter_result: Dict[str, Any],
    validation: Dict[str, Any],
    source_root: Path,
    use_hardlink: bool = False,
) -> bool:
    """
    Copy pre-synchronised episode data to destination.

    This is a pure file-copy function.  All synchronisation (left↔kinematic
    and multi-camera) must already have been performed by
    ``run_filter_episode()`` whose output is passed in as *filter_result*.

    Args:
        episode_path: Source episode path.
        out_dir: Destination root directory.
        filter_result: Output dict from ``run_filter_episode()`` containing
            ``valid_left_filenames``, ``kinematics_indices``, and
            ``secondary_camera_filenames``.
        validation: Validation dict (must contain ``kinematics_file``).
        source_root: Root of source tree for relative-path calculation.
        use_hardlink: Use hardlinks instead of copying.

    Returns:
        True if at least one frame was copied, False otherwise.
    """
    logger.info(f"Copying filtered episode: {episode_path.name}")

    try:
        # Calculate destination path preserving structure
        try:
            rel: Path = episode_path.relative_to(source_root)
        except ValueError:
            rel = Path(episode_path.name)

        dest_episode_dir: Path = out_dir / rel
        dest_episode_dir.mkdir(parents=True, exist_ok=True)

        left_filenames: List[str] = filter_result["valid_left_filenames"]
        secondary_fnames: Dict[str, List[str]] = filter_result[
            "secondary_camera_filenames"
        ]

        if not left_filenames:
            logger.warning("No valid left filenames to copy")
            return False

        # Create camera directories
        for _, _, dst_name in CAMERA_CONFIGS:
            (dest_episode_dir / dst_name).mkdir(exist_ok=True)

        def _copy_or_link(src: Path, dst: Path) -> bool:
            """Copy or hardlink a single file.  Returns True on success."""
            try:
                if use_hardlink:
                    try:
                        if dst.exists():
                            dst.unlink()
                        os.link(src, dst)
                    except OSError:
                        shutil.copy2(src, dst)
                else:
                    shutil.copy2(src, dst)
                return True
            except Exception as e:
                logger.error(f"Failed to copy {src.name}: {e}")
                return False

        copied_count: int = 0

        for i, left_fname in enumerate(left_filenames):
            src_left = episode_path / "left_img_dir" / left_fname
            dst_left = dest_episode_dir / "left_img_dir" / left_fname

            if not src_left.exists():
                logger.warning(f"Source left frame missing: {src_left}")
                continue

            if not _copy_or_link(src_left, dst_left):
                continue

            # Copy matched secondary frames (renamed to left timestamp)
            base_name: str = left_fname.replace("_left.jpg", "")

            for src_name, suffix, dst_name in CAMERA_CONFIGS:
                if src_name == "left_img_dir":
                    continue
                if src_name not in secondary_fnames:
                    continue

                match_fname = secondary_fnames[src_name][i]
                src_file = episode_path / src_name / match_fname
                new_name = f"{base_name}{suffix}"
                dst_file = dest_episode_dir / dst_name / new_name

                if src_file.exists():
                    _copy_or_link(src_file, dst_file)
                else:
                    logger.warning(f"Source match missing: {src_file}")

            copied_count += 1

        logger.info(f"Copied {copied_count} fully synchronized frames")

        # Filter and save kinematics
        write_filtered_kinematics(
            dest_episode_dir=dest_episode_dir,
            sync_df=filter_result["sync_df"],
            validation=validation,
            kept_filenames=left_filenames,
        )

        return copied_count > 0

    except Exception as e:
        logger.error(
            f"Failed to copy episode {episode_path.name}: {e}",
            exc_info=True,
        )
        return False


# ---------------------------------------------------------------------
# Single-Episode Filtering API (for external callers)
# ---------------------------------------------------------------------


def run_filter_episode(
    episode_path: Union[str, Path],
    max_time_diff_ms: float = DEFAULT_MAX_TIME_DIFF_MS,
) -> Dict[str, Any]:
    """
    Run the full filtering pipeline on a single episode.

    This is the primary entry point for external callers (e.g. the
    accelerated LeRobot converter).  It encapsulates:
        1. Left-camera ↔ kinematic synchronization with outlier removal
           (via ``process_episode_sync``).
        2. Vectorized multi-camera synchronization — every secondary camera
           must also have a match within the same threshold, otherwise the
           frame is dropped.

    Args:
        episode_path: Path to the episode directory.  Must contain
            ``left_img_dir/``, ``right_img_dir/``, ``endo_psm1/``,
            ``endo_psm2/`` and ``ee_csv.csv``.
        max_time_diff_ms: Maximum allowed time difference in milliseconds.
            Frames where *any* modality exceeds this are discarded.

    Returns:
        Dictionary with the following keys:

        - **success** (*bool*) ``True`` if the pipeline produced at
          least one fully-synchronised frame.
        - **error** (*str*, optional) Human-readable error message when
          ``success`` is ``False``.
        - **valid_left_filenames** (*List[str]*) Left-camera filenames
          that passed both sync stages, in chronological order.
        - **kinematics_indices** (*np.ndarray[int]*) Row indices into the
          original ``ee_csv.csv`` for each surviving frame.
        - **secondary_camera_indices** (*Dict[str, np.ndarray[int]]*)
          Per-camera arrays of indices into that camera's sorted frame list.
          Keys are ``"right_img_dir"``, ``"endo_psm1"``, ``"endo_psm2"``.
        - **num_valid** (*int*) Total number of fully-synchronised frames.
        - **outliers_removed** (*int*) Frames removed during left↔kinematic
          sync.
        - **multicam_dropped** (*int*) Frames removed during multi-camera
          sync.
    """
    episode_path = Path(episode_path)
    max_time_diff_ns: float = max_time_diff_ms * 1e6

    logger.info(f"run_filter_episode: {episode_path.name}  "
                f"threshold={max_time_diff_ms:.1f}ms")

    # ------------------------------------------------------------------
    # Stage 1 – Left-camera ↔ kinematic synchronisation
    # ------------------------------------------------------------------
    sync_result = process_episode_sync(
        episode_path=episode_path,
        camera="left",
        output_dir=None,
        max_time_diff_ms=max_time_diff_ms,
        plot=False,
        save_results=False,
    )

    if not sync_result["success"]:
        return {"success": False,
                "error": sync_result.get("error", "Unknown sync error")}

    valid_left_filenames: List[str] = sync_result["valid_filenames"]
    sync_df: pd.DataFrame = sync_result["sync_df"]

    if not valid_left_filenames:
        return {"success": False,
                "error": "No frames passed left↔kinematic sync threshold"}

    logger.info(
        f"  Left↔kinematic sync: {len(valid_left_filenames)} valid, "
        f"{sync_result['outliers_removed']} outliers removed"
    )

    # ------------------------------------------------------------------
    # Stage 2 – Vectorized multi-camera synchronisation
    # ------------------------------------------------------------------
    # Get left-camera timestamps for the surviving frames
    valid_left_ts = np.array(
        [extract_timestamp_from_filename(f) for f in valid_left_filenames],
        dtype=np.int64,
    )

    secondary_configs = [
        ("right_img_dir", "_right.jpg"),
        ("endo_psm1",     "_psm1.jpg"),
        ("endo_psm2",     "_psm2.jpg"),
    ]

    combined_valid_mask = np.ones(len(valid_left_ts), dtype=bool)
    secondary_match_indices: Dict[str, np.ndarray] = {}
    cached_candidates: Dict[str, List[Tuple[int, str]]] = {}

    for cam_dir, cam_suffix in secondary_configs:
        cam_candidates = load_camera_timestamps(episode_path, cam_dir, cam_suffix)
        cached_candidates[cam_dir] = cam_candidates
        if not cam_candidates:
            combined_valid_mask[:] = False
            logger.warning(f"  {cam_dir} has no frames — all frames dropped")
            break

        cam_ts = np.array([t for t, _ in cam_candidates], dtype=np.int64)
        best_idx, valid_mask = find_all_camera_matches_vectorized(
            valid_left_ts, cam_ts, max_time_diff_ns,
        )
        combined_valid_mask &= valid_mask
        secondary_match_indices[cam_dir] = best_idx

    fully_valid_positions = np.where(combined_valid_mask)[0]
    n_dropped_multicam = len(valid_left_ts) - len(fully_valid_positions)

    logger.info(
        f"  Multi-camera sync: {len(fully_valid_positions)} fully synced, "
        f"{n_dropped_multicam} dropped (camera mismatch)"
    )

    if len(fully_valid_positions) == 0:
        return {"success": False,
                "error": "No frames passed multi-camera sync threshold"}

    # ------------------------------------------------------------------
    # Build result
    # ------------------------------------------------------------------
    final_filenames = [valid_left_filenames[i] for i in fully_valid_positions]
    final_kinematics_indices = sync_df["kinematics_idx"].values[fully_valid_positions]
    final_secondary_indices = {
        cam: idx_arr[fully_valid_positions]
        for cam, idx_arr in secondary_match_indices.items()
    }

    # Materialise matched filenames using the cached candidates.
    final_secondary_filenames: Dict[str, List[str]] = {}
    for cam_dir, cam_suffix in secondary_configs:
        if cam_dir in secondary_match_indices:
            indices = final_secondary_indices[cam_dir]
            final_secondary_filenames[cam_dir] = [
                cached_candidates[cam_dir][idx][1] for idx in indices
            ]

    return {
        "success": True,
        "valid_left_filenames": final_filenames,
        "kinematics_indices": final_kinematics_indices,
        "secondary_camera_indices": final_secondary_indices,
        "secondary_camera_filenames": final_secondary_filenames,
        "sync_df": sync_df,
        "num_valid": len(final_filenames),
        "outliers_removed": sync_result["outliers_removed"],
        "multicam_dropped": n_dropped_multicam,
    }


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

    # Run full sync + multi-camera filtering via the single entry point
    filter_result: Dict[str, Any] = run_filter_episode(
        episode_path, max_time_diff
    )

    if not filter_result["success"]:
        result_stats["skipped_reason"] = (
            f"Sync failed: {filter_result.get('error', 'Unknown')}"
        )
        logger.debug(f"Sync failed: {result_stats['skipped_reason']}")
        return result_stats

    result_stats["sync_successful"] = True
    num_valid: int = filter_result["num_valid"]

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
            filter_result,
            validation,
            source_root,
            use_hardlink,
        )
        if success:
            result_stats["copied"] = True
            result_stats["images_copied"] = num_valid
            logger.info(f"Successfully processed: {episode_path.name}")
        else:
            result_stats["skipped_reason"] = "Copy failed"
            logger.warning(f"Copy failed: {episode_path.name}")
    else:
        # Dry run mode
        result_stats["copied"] = True
        result_stats["images_copied"] = num_valid
        logger.info(f"[DRY RUN] Would copy {num_valid} images from {episode_path.name}")

    return result_stats

def run_filter_episodes(
    source_dir: Union[str, Path],
    out_dir: Union[str, Path],
    max_time_diff: float = DEFAULT_MAX_TIME_DIFF_MS,
    min_images: int = DEFAULT_MIN_IMAGES,
    dry_run: bool = False,
    workers: Optional[int] = None,
    use_hardlink: bool = False,
    overwrite: bool = False,
) -> int:
    """
    Run the full batch filtering pipeline on a directory of episodes.

    Discovers all episodes in *source_dir*, validates them, runs strict
    synchronisation (left-camera <-> kinematics + multi-camera), and copies
    the surviving frames to *out_dir*.

    This function encapsulates the logic that was previously inlined in
    ``main()``, making it callable from other scripts and GUIs.

    Args:
        source_dir: Root directory containing episode subdirectories.
        out_dir: Destination root for filtered data.
        max_time_diff: Maximum allowed time difference in milliseconds.
        min_images: Minimum valid images per episode to keep it.
        dry_run: If True, preview without copying files.
        workers: Number of parallel workers (default: CPU count, max 8).
        use_hardlink: Use hardlinks instead of file copies.
        overwrite: Overwrite existing destination episodes.

    Returns:
        0 on success, 1 on error.
    """
    source_dir = Path(source_dir)
    out_dir = Path(out_dir)

    # Validate source
    if not source_dir.exists():
        logger.error(f"Source directory not found: {source_dir}")
        print(f"Error: Source directory not found: {source_dir}")
        return 1

    # Create destination if not dry run
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Log configuration
    logger.info("=" * 70)
    logger.info("Starting episode filtering pipeline")
    logger.info(f"Source: {source_dir}")
    logger.info(f"Destination: {out_dir}")
    logger.info(f"Max time diff: {max_time_diff} ms")
    logger.info(f"Min images: {min_images}")
    logger.info(f"Dry run: {dry_run}")
    logger.info(f"Hardlink: {use_hardlink}")
    logger.info(f"Overwrite: {overwrite}")
    logger.info("=" * 70)

    # Print configuration to console
    print(f"Processing episodes from: {source_dir}")
    print(f"Destination directory: {out_dir}")
    print(f"Max time difference: {max_time_diff} ms")
    print(f"Min images required: {min_images}")
    if workers:
        print(f"Workers: {workers}")
    if dry_run:
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

        if dest_episode_dir.exists() and not overwrite and not dry_run:
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
    max_procs: int = min(workers or os.cpu_count() or 1, MAX_WORKERS)
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
                    max_time_diff,
                    min_images,
                    source_dir,
                    dry_run,
                    use_hardlink,
                    overwrite,
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

    if dry_run:
        print("\n[DRY RUN] No files were actually copied.")

    print("=" * 60)

    logger.info("Processing complete")
    logger.info(f"Copied {stats['copied_episodes']} episodes")
    logger.info(f"Total images copied: {stats['total_images_copied']}")

    return 0

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

    return run_filter_episodes(
        source_dir=Path(args.source_dir),
        out_dir=Path(args.out_dir),
        max_time_diff=args.max_time_diff,
        min_images=args.min_images,
        dry_run=args.dry_run,
        workers=args.workers,
        use_hardlink=args.hardlink,
        overwrite=args.overwrite,
    )

if __name__ == "__main__":
    sys.exit(main())