#!/usr/bin/env python3
"""
merge_actions_to_videos.py

This script processes dVRK (da Vinci Research Kit) surgical robot data collected
during cautery tasks. It consolidates multiple action runs within each tissue
folder into unified video streams and merged kinematic data.

Data Structure:
    root_dir/
        cautery_tissue_001/
            2024_01_15_10_30_45/  # Run 1 (timestamped)
                endo_psm1/         # Endoscope PSM1 camera frames
                    frame_0001.png
                    frame_0002.png
                    ...
                endo_psm2/         # Endoscope PSM2 camera frames
                left_img_dir/      # Left stereo camera frames
                right_img_dir/     # Right stereo camera frames
                ee_csv.csv         # End-effector kinematic data
            2024_01_15_11_45_30/  # Run 2
                ...
        cautery_tissue_002/
            ...

Processing Pipeline:
    For each cautery_tissue folder, the script:
    1. Identifies all timestamped run directories
    2. Concatenates frames from each run into continuous videos (one per modality)
    3. Merges all ee_csv.csv files vertically (temporal concatenation)
    4. Saves outputs to a "videos" subfolder within the tissue directory
    5. Resizes frames within each modality to match the first frame's dimensions
    6. Maintains temporal ordering based on directory sorting

Output Structure:
    cautery_tissue_001/
        videos/
            endo_psm1.mp4      # Concatenated video from all runs
            endo_psm2.mp4      # Concatenated video from all runs
            left_img_dir.mp4   # Concatenated video from all runs
            right_img_dir.mp4  # Concatenated video from all runs
            ee_csv.csv         # Merged kinematic data from all runs

Usage:
    # Process all tissue folders in the default cautery directory
    python3 merge_actions_to_videos.py /path/to/cautery

    # Specify custom FPS
    python3 merge_actions_to_videos.py /path/to/cautery --fps 60

    # Overwrite existing outputs
    python3 merge_actions_to_videos.py /path/to/cautery --overwrite

    # Combine options
    python3 merge_actions_to_videos.py /path/to/cautery --fps 30 --overwrite

Notes:
    - Frames are assumed to be naturally ordered by filename
    - All frames within a modality are resized to the first frame's dimensions
    - CSV merging uses raw append (headers from first run only)
    - Runs are processed in sorted order (typically chronological)
    - Missing modalities or CSVs in runs are handled gracefully
    - Video codec: MP4V (widely compatible)
"""

import argparse
import csv
import os
from pathlib import Path
from typing import List, Optional

import cv2

from logger_config import get_logger

# ---------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------

# Supported image modalities in the dVRK data structure
MODALITIES: List[str] = ["endo_psm1", "endo_psm2", "left_img_dir", "right_img_dir"]

# Supported image file extensions
IMAGE_EXTENSIONS: tuple = (".jpg", ".jpeg", ".png")

# CSV filename containing end-effector kinematic data
KINEMATIC_CSV_NAME: str = "ee_csv.csv"

# Output directory name for processed videos and merged data
OUTPUT_DIR_NAME: str = "videos"

# Video encoding settings
VIDEO_CODEC: str = "mp4v"
DEFAULT_FPS: int = 30

# Tissue directory prefix for identification
TISSUE_DIR_PREFIX: str = "cautery_tissue"

# Initialize module logger
logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Directory Discovery Functions
# ---------------------------------------------------------------------


def list_tissue_dirs(root_dir: str) -> List[str]:
    """
    List all cautery tissue directories within the root directory.

    Scans the root directory for subdirectories that match the tissue
    naming convention (starting with "cautery_tissue"). These directories
    represent different tissue samples used in the experiments.

    Args:
        root_dir: Path to the root directory containing tissue folders.
            Must be a valid, readable directory.

    Returns:
        Sorted list of full paths to tissue directories. Returns empty
        list if no matching directories are found.

    Notes:
        - Only directories are included (files are ignored)
        - Sorting ensures consistent processing order
        - Does not validate internal structure (done later)
        - Does not recurse into subdirectories
    """
    dirs: List[str] = []
    
    logger.debug(f"Scanning for tissue directories in: {root_dir}")
    
    for entry in os.listdir(root_dir):
        path: str = os.path.join(root_dir, entry)
        
        # Only include directories that match the naming pattern
        if os.path.isdir(path) and entry.startswith(TISSUE_DIR_PREFIX):
            dirs.append(path)
            logger.debug(f"Found tissue directory: {entry}")
    
    dirs_sorted: List[str] = sorted(dirs)
    logger.info(f"Discovered {len(dirs_sorted)} tissue directories")
    
    return dirs_sorted


def list_run_dirs(tissue_dir: str) -> List[str]:
    """
    List all valid run directories within a tissue folder.

    A valid run directory must contain all four required modality
    subdirectories (endo_psm1, endo_psm2, left_img_dir, right_img_dir).
    Run directories are typically timestamped but this function validates
    by structure rather than naming pattern for flexibility.

    Args:
        tissue_dir: Path to a tissue folder containing run directories.
            Should be a valid directory path.

    Returns:
        Sorted list of full paths to valid run directories. Returns empty
        list if no valid runs are found.

    Notes:
        - Validation requires ALL four modalities to be present
        - Timestamp format is not enforced (flexible for different datasets)
        - Sorting preserves chronological order for timestamped directories
        - Runs missing any modality are excluded entirely
        - Does not check for ee_csv.csv (handled separately during processing)
    """
    runs: List[str] = []
    
    logger.debug(f"Scanning for run directories in: {tissue_dir}")
    
    for entry in os.listdir(tissue_dir):
        path: str = os.path.join(tissue_dir, entry)
        
        # Skip non-directories
        if not os.path.isdir(path):
            continue
        
        # Check if all required modalities are present
        has_all_modalities: bool = all(
            os.path.isdir(os.path.join(path, modality)) for modality in MODALITIES
        )
        
        if has_all_modalities:
            runs.append(path)
            logger.debug(f"Found valid run directory: {entry}")
        else:
            logger.warning(
                f"Skipping incomplete run directory (missing modalities): {entry}"
            )
    
    runs_sorted: List[str] = sorted(runs)
    logger.info(f"Discovered {len(runs_sorted)} valid run directories")
    
    return runs_sorted


# ---------------------------------------------------------------------
# CSV Processing Functions
# ---------------------------------------------------------------------


def read_csv(filepath: str) -> List[List[str]]:
    """
    Read a CSV file into a list of rows.

    Reads the entire CSV file into memory as a list of lists, where each
    inner list represents one row of string fields. This simple approach
    works well for the typically small kinematic CSV files in dVRK data.

    Args:
        filepath: Full path to the CSV file to read. File must exist
            and be readable.

    Returns:
        List of rows, where each row is a list of string fields.
        Returns empty list if file is empty.

    Raises:
        FileNotFoundError: If filepath does not exist.
        PermissionError: If file cannot be read due to permissions.
        csv.Error: If file is malformed and cannot be parsed.

    Notes:
        - Does not validate header consistency across files
        - All fields are read as strings (no type conversion)
        - Empty lines are preserved as empty lists
        - Uses Python's csv.reader with default settings
    """
    rows: List[List[str]] = []
    
    logger.debug(f"Reading CSV file: {filepath}")
    
    try:
        with open(filepath, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append(row)
        
        logger.debug(f"Successfully read {len(rows)} rows from CSV")
        
    except Exception as e:
        logger.error(f"Failed to read CSV file {filepath}: {e}", exc_info=True)
        raise
    
    return rows


def write_csv(filepath: str, rows: List[List[str]]) -> None:
    """
    Write rows to a CSV file.

    Writes a list of rows to a CSV file, creating or overwriting the file.
    Parent directories are created automatically if they don't exist.

    Args:
        filepath: Output file path where CSV should be written.
        rows: List of rows to write, where each row is a list of fields.

    Returns:
        None. Results are written to disk.

    Raises:
        OSError: If file cannot be written (permissions, disk full, etc.).
        csv.Error: If rows contain data that cannot be CSV-encoded.

    Side Effects:
        - Creates parent directories if they don't exist
        - Overwrites existing file without warning
        - Writes file with UTF-8 encoding

    Notes:
        - Uses Python's csv.writer with default settings
        - Automatically handles field quoting when necessary
        - Newline handling is platform-independent (newline="" parameter)
    """
    logger.debug(f"Writing {len(rows)} rows to CSV: {filepath}")
    
    # Ensure parent directory exists
    output_path: Path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        
        logger.info(f"Successfully wrote CSV file: {filepath}")
        
    except Exception as e:
        logger.error(f"Failed to write CSV file {filepath}: {e}", exc_info=True)
        raise


def merge_csvs(run_dirs: List[str], output_csv_path: str) -> None:
    """
    Merge kinematic CSV files from multiple runs via vertical concatenation.

    Combines ee_csv.csv files from all runs into a single merged CSV file.
    The first run's header is preserved, and subsequent headers are stripped
    to avoid duplication. Data rows are appended in the order of run_dirs.

    Args:
        run_dirs: List of run directory paths, in desired merge order
            (typically chronological). Each run should contain ee_csv.csv.
        output_csv_path: Full path where the merged CSV should be saved.

    Returns:
        None. Merged CSV is written to disk.

    Raises:
        OSError: If output file cannot be written.

    Side Effects:
        - Creates output file (or overwrites if exists)
        - Logs warnings for missing or empty CSVs
        - Continues processing even if some runs have missing CSVs

    Merge Strategy:
        - First CSV: Header + all data rows preserved
        - Subsequent CSVs: Only data rows appended (header stripped)
        - Header mismatch: Warning logged but data still appended
        - Missing CSVs: Skipped with warning

    Notes:
        - No new columns are introduced (raw vertical append)
        - Assumes headers are consistent across runs (not enforced)
        - Run order determines temporal sequence in merged output
        - Empty CSV files are skipped
        - If ALL CSVs are missing/empty, output file is not created
    """
    logger.info(f"Merging {len(run_dirs)} CSV files")
    
    merged_rows: List[List[str]] = []
    header: Optional[List[str]] = None
    files_processed: int = 0

    for idx, run_dir in enumerate(run_dirs):
        csv_path: str = os.path.join(run_dir, KINEMATIC_CSV_NAME)
        
        # Check if CSV exists
        if not os.path.exists(csv_path):
            logger.warning(f"Missing {KINEMATIC_CSV_NAME} in {run_dir}. Skipping.")
            continue

        try:
            rows: List[List[str]] = read_csv(csv_path)
        except Exception as e:
            logger.error(f"Failed to read {csv_path}: {e}. Skipping this run.")
            continue
        
        if not rows:
            logger.warning(f"Empty CSV in {run_dir}. Skipping.")
            continue

        # Handle header logic
        if idx == 0 or header is None:
            # First valid CSV: preserve header and all data
            header = rows[0]
            merged_rows.append(header)
            merged_rows.extend(rows[1:])
            files_processed += 1
            logger.debug(f"Added header and {len(rows)-1} data rows from first CSV")
        else:
            # Subsequent CSVs: check header consistency, then append data
            if rows[0] == header:
                # Header matches - strip it and append data
                merged_rows.extend(rows[1:])
                files_processed += 1
                logger.debug(f"Added {len(rows)-1} data rows from {csv_path}")
            else:
                # Header mismatch - warn but still append all rows
                logger.warning(
                    f"Header mismatch in {csv_path}. "
                    f"Expected {header}, got {rows[0]}. Appending raw data."
                )
                merged_rows.extend(rows)
                files_processed += 1

    # Write merged output if we have data
    if merged_rows:
        write_csv(output_csv_path, merged_rows)
        logger.info(
            f"Created merged CSV with {len(merged_rows)} total rows "
            f"from {files_processed} files: {output_csv_path}"
        )
    else:
        logger.warning(
            f"No CSV data to merge. Output not created: {output_csv_path}"
        )


# ---------------------------------------------------------------------
# Frame Processing Functions
# ---------------------------------------------------------------------


def collect_frames(run_dir: str, modality: str) -> List[str]:
    """
    Collect all image frame paths for a modality within a run.

    Scans the modality subdirectory for image files and returns them
    in sorted order. Only files with recognized image extensions are
    included.

    Args:
        run_dir: Path to the run directory containing modality subdirectories.
        modality: Name of the modality subdirectory (e.g., "endo_psm1").

    Returns:
        Sorted list of full paths to image files. Returns empty list if
        modality directory doesn't exist or contains no images.

    Notes:
        - Only files with extensions in IMAGE_EXTENSIONS are included
        - Case-insensitive extension matching (.JPG and .jpg both work)
        - Sorted alphabetically (assumes natural ordering in filenames)
        - Subdirectories within modality folder are ignored
        - Returns empty list (not error) if directory doesn't exist
    """
    modality_dir: str = os.path.join(run_dir, modality)
    
    if not os.path.isdir(modality_dir):
        logger.debug(f"Modality directory does not exist: {modality_dir}")
        return []
    
    # Collect all image files
    frames: List[str] = sorted(
        [
            os.path.join(modality_dir, f)
            for f in os.listdir(modality_dir)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        ]
    )
    
    logger.debug(f"Collected {len(frames)} frames from {modality_dir}")
    return frames


def get_reference_frame_size(frame_path: str) -> Optional[tuple]:
    """
    Get the dimensions of a reference frame for video sizing.

    Reads an image file and extracts its width and height. This is used
    to establish consistent dimensions for all frames in a video.

    Args:
        frame_path: Full path to an image file.

    Returns:
        Tuple of (width, height) in pixels if successful, None if the
        frame cannot be read.

    Notes:
        - Reads image in BGR color format (OpenCV default)
        - Returns None for corrupted or unreadable images
        - Does not validate that the image is actually an image file
    """
    frame = cv2.imread(frame_path)
    
    if frame is None:
        logger.warning(f"Failed to read reference frame: {frame_path}")
        return None
    
    height, width, _ = frame.shape
    logger.debug(f"Reference frame size: {width}x{height}")
    
    return (width, height)


# ---------------------------------------------------------------------
# Video Generation Functions
# ---------------------------------------------------------------------


def build_video_for_modality(
    run_dirs: List[str],
    modality: str,
    output_path: str,
    overwrite: bool = False,
    fps: int = DEFAULT_FPS,
) -> None:
    """
    Concatenate frames from multiple runs into a single video for one modality.

    This function implements the core video generation pipeline:
        1. Collect frames from all runs for the specified modality
        2. Determine reference dimensions from the first valid frame
        3. Initialize video writer with appropriate codec and settings
        4. Process each frame with automatic resizing if needed
        5. Handle corrupted frames gracefully

    Args:
        run_dirs: List of run directory paths to process, in desired order.
            Frames from these runs will be concatenated chronologically.
        modality: The modality to process (e.g., "endo_psm1", "left_img_dir").
            Must match a subdirectory name within each run.
        output_path: Full path where the output video should be saved.
            Parent directories will be created if needed.
        overwrite: If False, skips processing if output file already exists.
            If True, overwrites existing output. Default is False.
        fps: Target frames per second for the output video. Must be positive.
            Default is 30 FPS.

    Returns:
        None. Video is written to disk and progress is logged.

    Raises:
        OSError: If output directory cannot be created.

    Side Effects:
        - Creates output directory if it doesn't exist
        - Writes video file to disk (or skips if exists and overwrite=False)
        - Logs progress, warnings for corrupted frames, and final summary

    Processing Details:
        - All frames are resized to match first frame's dimensions
        - Corrupted frames are skipped (logged but non-fatal)
        - Runs without frames for this modality are silently skipped
        - Video codec: MP4V (broad compatibility)
        - Pixel format: BGR (OpenCV standard)

    Notes:
        - If no frames found across all runs, no video is created
        - If first frame cannot be read, entire processing is aborted
        - Video writer is always properly released
        - Progress logged every 100 frames
    """
    logger.info(f"Building video for modality: {modality}")
    
    # Check if output already exists and overwrite is disabled
    if os.path.exists(output_path) and not overwrite:
        logger.info(
            f"Skipping '{output_path}' (already exists, overwrite=False)"
        )
        return

    # Collect all frames across all runs
    all_frame_paths: List[str] = []
    for run_dir in run_dirs:
        frames: List[str] = collect_frames(run_dir, modality)
        all_frame_paths.extend(frames)

    if not all_frame_paths:
        logger.warning(
            f"No frames found for modality '{modality}' across all runs. "
            f"Video not created."
        )
        return

    logger.info(f"Collected {len(all_frame_paths)} total frames for {modality}")

    # Get reference frame dimensions
    first_frame_path: str = all_frame_paths[0]
    frame_size: Optional[tuple] = get_reference_frame_size(first_frame_path)
    
    if frame_size is None:
        logger.error(
            f"Failed to read first frame for modality '{modality}': "
            f"{first_frame_path}. Cannot create video."
        )
        return

    width, height = frame_size
    logger.info(f"Reference frame size for {modality}: {width}x{height}")

    # Ensure output directory exists
    output_parent: Path = Path(output_path).parent
    output_parent.mkdir(parents=True, exist_ok=True)

    # Initialize video writer
    fourcc: int = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
    writer: cv2.VideoWriter = cv2.VideoWriter(
        output_path, fourcc, fps, (width, height)
    )

    if not writer.isOpened():
        logger.error(
            f"Failed to initialize VideoWriter for {output_path}. "
            f"Check codec availability."
        )
        return

    # Process frames
    frames_written: int = 0
    frames_skipped: int = 0

    try:
        for idx, frame_path in enumerate(all_frame_paths):
            frame = cv2.imread(frame_path)
            
            if frame is None:
                # Non-fatal: skip corrupted frames
                frames_skipped += 1
                logger.debug(
                    f"Skipped unreadable frame ({frames_skipped} total): "
                    f"{frame_path}"
                )
                continue

            # Resize if dimensions don't match reference
            current_size: tuple = (frame.shape[1], frame.shape[0])
            if current_size != (width, height):
                logger.debug(
                    f"Resizing frame from {current_size} to {width}x{height}"
                )
                frame = cv2.resize(frame, (width, height))

            # Write frame to video
            writer.write(frame)
            frames_written += 1

            # Log progress periodically
            if frames_written % 100 == 0:
                logger.debug(
                    f"Progress: {frames_written}/{len(all_frame_paths)} frames written"
                )

    finally:
        # Always release writer, even if error occurs
        writer.release()
        logger.debug("Released VideoWriter")

    # Final summary
    logger.info(
        f"Completed video for {modality}: {frames_written} frames written, "
        f"{frames_skipped} frames skipped. Output: {output_path}"
    )


# ---------------------------------------------------------------------
# High-Level Processing Functions
# ---------------------------------------------------------------------


def process_tissue_directory(
    tissue_dir: str, overwrite: bool = False, fps: int = DEFAULT_FPS
) -> None:
    """
    Process a single tissue directory: create videos and merge CSVs.

    This function orchestrates the complete processing pipeline for one
    tissue sample:
        1. Find all valid run directories
        2. Create output directory
        3. Generate concatenated videos for each modality
        4. Merge kinematic CSV files

    Args:
        tissue_dir: Full path to the tissue directory to process.
            Should contain run subdirectories.
        overwrite: Whether to overwrite existing output files.
            If False, existing videos and CSVs are skipped.
        fps: Frames per second for output videos. Must be positive.

    Returns:
        None. All outputs are written to tissue_dir/videos/

    Side Effects:
        - Creates tissue_dir/videos/ directory
        - Generates up to 4 video files (one per modality)
        - Creates merged ee_csv.csv file
        - Logs detailed progress for each step

    Output Files:
        - videos/endo_psm1.mp4
        - videos/endo_psm2.mp4
        - videos/left_img_dir.mp4
        - videos/right_img_dir.mp4
        - videos/ee_csv.csv

    Notes:
        - Skips tissue directory if no valid runs are found
        - Each modality processed independently (one failure doesn't affect others)
        - CSV merge is independent of video generation
        - All file I/O errors are logged but don't stop other processing
    """
    logger.info("=" * 70)
    logger.info(f"Processing tissue directory: {os.path.basename(tissue_dir)}")
    logger.info("=" * 70)

    # Find all valid run directories
    run_dirs: List[str] = list_run_dirs(tissue_dir)
    
    if not run_dirs:
        logger.warning(f"No valid runs found in {tissue_dir}. Skipping.")
        return

    logger.info(f"Found {len(run_dirs)} runs to process")

    # Create output directory
    videos_dir: str = os.path.join(tissue_dir, OUTPUT_DIR_NAME)
    os.makedirs(videos_dir, exist_ok=True)
    logger.info(f"Output directory: {videos_dir}")

    # Build videos for each modality
    logger.info(f"Generating videos for {len(MODALITIES)} modalities")
    for modality in MODALITIES:
        output_path: str = os.path.join(videos_dir, f"{modality}.mp4")
        
        try:
            build_video_for_modality(
                run_dirs=run_dirs,
                modality=modality,
                output_path=output_path,
                overwrite=overwrite,
                fps=fps,
            )
        except Exception as e:
            logger.error(
                f"Failed to build video for modality '{modality}': {e}",
                exc_info=True,
            )
            # Continue with other modalities even if one fails

    # Merge CSV files
    logger.info("Merging kinematic CSV files")
    output_csv_path: str = os.path.join(videos_dir, KINEMATIC_CSV_NAME)
    
    if not os.path.exists(output_csv_path) or overwrite:
        try:
            merge_csvs(run_dirs, output_csv_path)
        except Exception as e:
            logger.error(f"Failed to merge CSV files: {e}", exc_info=True)
    else:
        logger.info(
            f"Skipping CSV merge (exists, overwrite=False): {output_csv_path}"
        )

    logger.info(f"Completed processing for {os.path.basename(tissue_dir)}")


def process_root(
    root_dir: str, overwrite: bool = False, fps: int = DEFAULT_FPS
) -> None:
    """
    Process all tissue folders within the root directory.

    This is the top-level processing function that orchestrates the
    entire pipeline across all tissue samples in the dataset.

    Args:
        root_dir: Root directory containing cautery_tissue_* folders.
            Must exist and be readable.
        overwrite: Whether to overwrite existing output files.
            Applies to all tissue directories.
        fps: Frames per second for all output videos.

    Returns:
        None. All outputs written to respective tissue_dir/videos/

    Raises:
        FileNotFoundError: If root_dir does not exist.

    Side Effects:
        - Processes all discovered tissue directories
        - Creates multiple output files across tissue folders
        - Logs comprehensive processing summary

    Processing Order:
        - Tissue directories processed in sorted (alphabetical) order
        - Within each tissue, runs processed in sorted order
        - Deterministic and reproducible

    Notes:
        - One tissue failure doesn't stop processing of others
        - Comprehensive logging for debugging and monitoring
        - Warns if no tissue directories found
    """
    logger.info("=" * 70)
    logger.info("Starting dVRK data processing pipeline")
    logger.info(f"Root directory: {root_dir}")
    logger.info(f"Overwrite mode: {overwrite}")
    logger.info(f"Video FPS: {fps}")
    logger.info("=" * 70)

    # Validate root directory exists
    if not os.path.isdir(root_dir):
        logger.error(f"Root directory does not exist: {root_dir}")
        raise FileNotFoundError(f"Root directory does not exist: {root_dir}")

    # Find all tissue directories
    tissues: List[str] = list_tissue_dirs(root_dir)
    
    if not tissues:
        logger.warning("No tissue directories found in root directory")
        return

    logger.info(f"Found {len(tissues)} tissue directories to process")

    # Process each tissue directory
    tissues_processed: int = 0
    tissues_failed: int = 0

    for tissue_dir in tissues:
        try:
            process_tissue_directory(tissue_dir, overwrite=overwrite, fps=fps)
            tissues_processed += 1
        except Exception as e:
            tissues_failed += 1
            logger.error(
                f"Failed to process tissue directory {tissue_dir}: {e}",
                exc_info=True,
            )
            # Continue with remaining tissues

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("Processing complete!")
    logger.info(f"Total tissue directories: {len(tissues)}")
    logger.info(f"Successfully processed: {tissues_processed}")
    logger.info(f"Failed: {tissues_failed}")
    logger.info("=" * 70)


# ---------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------


def main() -> None:
    """
    Entry point with command-line argument parsing.

    Parses command-line arguments and initiates the processing pipeline
    for dVRK surgical robot data. Provides a user-friendly CLI interface
    with validation and helpful error messages.

    Command-line Arguments:
        root_dir: Positional argument specifying the root directory
                 containing cautery_tissue_* folders. Required.
        --fps: Optional integer specifying frames per second for videos.
              Default: 30. Must be positive.
        --overwrite: Optional flag to overwrite existing outputs.
                    Default: False (skip existing files).

    Exit Codes:
        0: Success (all processing completed)
        1: Error (exception during processing)
        2: Invalid arguments

    Examples:
        # Basic usage with defaults
        $ python3 merge_actions_to_videos.py /data/cautery

        # Custom FPS
        $ python3 merge_actions_to_videos.py /data/cautery --fps 60

        # Overwrite existing outputs
        $ python3 merge_actions_to_videos.py /data/cautery --overwrite

        # Combine options
        $ python3 merge_actions_to_videos.py /data/cautery --fps 30 --overwrite

    Notes:
        - Validates that root_dir is provided (required argument)
        - FPS must be positive integer
        - All errors are logged with full stack traces
        - Graceful handling of keyboard interrupts (Ctrl+C)
    """
    parser = argparse.ArgumentParser(
        description="Merge dVRK action runs into consolidated videos and CSV data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    %(prog)s /data/cautery
    %(prog)s /data/cautery --fps 60
    %(prog)s /data/cautery --overwrite
    %(prog)s /data/cautery --fps 30 --overwrite

    This script processes surgical robot data by concatenating video frames
    and merging kinematic CSV files from multiple runs within each tissue folder.
            """,
        )

    parser.add_argument(
        "root_dir",
        type=str,
        help="Path to root directory containing cautery_tissue_* folders",
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help=f"Frames per second for output videos (default: {DEFAULT_FPS})",
    )
    
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files instead of skipping them",
    )

    args = parser.parse_args()

    # Validate FPS is positive
    if args.fps <= 0:
        logger.error(f"FPS must be positive, got: {args.fps}")
        parser.error("FPS must be a positive integer")

    # Execute processing pipeline
    try:
        process_root(root_dir=args.root_dir, overwrite=args.overwrite, fps=args.fps)
    except KeyboardInterrupt:
        logger.warning("\nProcessing interrupted by user (Ctrl+C)")
        return
    except Exception as e:
        logger.error(f"Processing failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()