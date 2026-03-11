#!/usr/bin/env python3
"""
frames_to_vids.py

Traverse a 'cautery' folder structure and convert images in camera folders
(endo_psm1, endo_psm2, left_img_dir, right_img_dir) into videos (one per camera/run).

This script processes medical imaging data organized in a hierarchical folder structure:
    Cholecystectomy/
        SubjectName/
            Tissue#1/
                20260304-175158-967720/
                    left_img_dir/
                        frame_0001.png
                        frame_0002.png
                        ...

The script will:
    1. Traverse all subject directories
    2. Traverse all tissue directories within each subject
    3. Process each run within tissue directories
    4. Convert image sequences from left_img_dir into sequentially named MP4 videos (Cholecystectomy_tissue######.mp4)
    5. Handle missing frames and resizing automatically

Usage:
    # Basic usage with default settings
    python3 frames_to_vids.py

    # Custom root directory and frame rate
    python3 frames_to_vids.py --root_dir mydata --fps 30

    # Dry run to preview what will be processed
    python3 frames_to_vids.py --dry_run

    # Overwrite existing videos
    python3 frames_to_vids.py --overwrite

    # Custom output directory
    python3 frames_to_vids.py --out_dir my_videos

Notes:
    - Images are sorted naturally (frame1, frame2, ..., frame10, frame11)
    - All frames are resized to match the first readable frame
    - Unreadable frames are skipped and counted
    - Progress is logged every 500 frames
    - Videos use MP4V codec by default
"""

import argparse
import concurrent.futures
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

from surpass_data_collection.logger_config import get_logger

# ---------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------

# Supported camera folder names in the expected directory structure (only left_img_dir required now)
CAM_LIST: Tuple[str, ...] = ("left_img_dir",)

# File extensions to skip when collecting images
SKIP_EXTENSIONS: Tuple[str, ...] = (".csv", ".txt", ".json", ".xml")

# Video codec fourcc code (Motion JPEG as fallback if MP4V fails)
PRIMARY_CODEC: str = "mp4v"
FALLBACK_CODEC: str = "MJPG"

# Progress reporting interval (number of frames)
PROGRESS_INTERVAL: int = 500

# Skipped frame warning interval
SKIP_WARNING_INTERVAL: int = 50

# Default video parameters
DEFAULT_FPS: int = 30
DEFAULT_ROOT_DIR: str = r"D:\Data\Cholecystectomy"
DEFAULT_OUT_DIR: str = "Cholecystectomy_videos"

# Initialize module logger
logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------


_DIGIT_SPLIT_RE = re.compile(r"(\d+)")

def natural_key(s: str) -> List:
    """
    Generate a natural sort key for a string containing numbers.

    Splits the string into digit and non-digit segments, converting
    numeric parts to integers for proper numerical ordering. This ensures
    that "frame2" sorts before "frame10" (as opposed to lexicographic
    sorting where "frame10" would come before "frame2").

    Args:
        s: Input string to generate sort key for, typically a filename.

    Returns:
        List of alternating integers and strings suitable for sorting.
        Example: "frame10" -> ["frame", 10, ""]
    """
    parts: List[str] = _DIGIT_SPLIT_RE.split(s)
    key: List = []
    for p in parts:
        key.append(int(p) if p.isdigit() else p.lower())
    return key


def collect_image_files(img_dir: str) -> List[str]:
    """
    Collect and sort image files from a directory, excluding non-image files.

    Scans the specified directory for image files, filtering out common
    non-image file types (CSV, TXT, JSON, XML). Returns a naturally sorted
    list of full paths to ensure frames are processed in the correct order.

    Args:
        img_dir: Path to directory containing image files. Must be a valid
            directory path (not validated here - caller's responsibility).

    Returns:
        List of full file paths to image files, sorted in natural order.
        Returns empty list if directory is empty or contains no valid images.

    Raises:
        OSError: If the directory cannot be read (permission denied, etc.)
            This exception is not caught here - caller should handle it.

    Notes:
        - Only regular files are included (directories are skipped)
        - Subdirectories are not traversed (non-recursive)
        - Files with extensions in SKIP_EXTENSIONS are excluded
        - Case-insensitive extension matching (both .CSV and .csv are skipped)
    """
    entries: List[str] = []
    
    try:
        with os.scandir(img_dir) as it:
            for entry in it:
                # Skip directories - we only want files
                if not entry.is_file():
                    continue
                
                # Skip non-image files by extension
                name = entry.name
                if any(name.lower().endswith(ext) for ext in SKIP_EXTENSIONS):
                    continue
                
                entries.append(entry.path)
    except OSError as e:
        logger.error(f"Failed to scan {img_dir}: {e}")
        return []

    # Sort based on natural ordering of the basename
    # This ensures frame_1, frame_2, ..., frame_10, frame_11 order
    entries.sort(key=lambda p: natural_key(os.path.basename(p)))
    
    logger.debug(f"Collected {len(entries)} image files from {img_dir}")
    return entries


def choose_writer(
    path_out: str, fps: int, frame_size: Tuple[int, int]
) -> Tuple[Optional[cv2.VideoWriter], Optional[str]]:
    """
    Create a cv2.VideoWriter with automatic codec fallback.

    Attempts to create a video writer using the MP4V codec. If that fails,
    falls back to MJPG codec with an .avi extension. This provides robustness
    across different OpenCV builds and operating systems.

    Args:
        path_out: Desired output video file path. Should end in .mp4 for
            primary codec attempt.
        fps: Target frames per second for the output video. Must be positive.
        frame_size: (width, height) tuple specifying frame dimensions in pixels.
            Both values must be positive integers.

    Returns:
        Tuple of (VideoWriter, actual_output_path):
            - VideoWriter: Initialized cv2.VideoWriter object if successful
            - actual_output_path: The path that will be used (may differ from
              path_out if fallback codec is used)
            - (None, None) if no writer could be created

    Notes:
        - Primary attempt uses MP4V codec (widely compatible)
        - Fallback uses MJPG codec with .avi extension (more compatible but larger files)
        - The isOpened() method is used to verify writer initialization
        - Caller is responsible for releasing the returned writer
        - Some OpenCV builds may not support certain codecs
    """
    # Primary attempt: MP4 using mp4v codec
    # This is the preferred format for compatibility and file size
    writer: cv2.VideoWriter = cv2.VideoWriter(
        path_out, cv2.VideoWriter_fourcc(*PRIMARY_CODEC), fps, frame_size
    )
    
    if writer.isOpened():
        logger.debug(f"Successfully opened VideoWriter with {PRIMARY_CODEC} codec")
        return writer, path_out

    # If primary codec fails, try fallback
    logger.warning(f"{PRIMARY_CODEC} codec failed, attempting fallback to {FALLBACK_CODEC}")
    
    # Change extension to .avi for MJPG codec
    fallback_path: str = str(Path(path_out).with_suffix(".avi"))
    writer = cv2.VideoWriter(
        fallback_path, cv2.VideoWriter_fourcc(*FALLBACK_CODEC), fps, frame_size
    )
    
    if writer.isOpened():
        print(f"Using fallback {FALLBACK_CODEC} codec, output: {fallback_path}")
        return writer, fallback_path

    # Both attempts failed
    logger.error(f"Failed to open VideoWriter with both {PRIMARY_CODEC} and {FALLBACK_CODEC}")
    return None, None


def find_first_readable_image(
    files: List[str],
) -> Tuple[Optional[cv2.Mat], Optional[int], Optional[Tuple[int, int]]]:
    """
    Find the first readable image in a list of file paths.

    Iterates through file paths attempting to read each as an image until
    one succeeds. This is used to determine the reference frame size for
    the video and to validate that the directory contains processable images.

    Args:
        files: List of full file paths to potential image files. Should be
            in the desired processing order (typically naturally sorted).

    Returns:
        Tuple of (image, index, frame_size):
            - image: The first successfully loaded cv2.Mat image, or None
            - index: The index in the files list where the image was found, or None
            - frame_size: (width, height) tuple of the image dimensions, or None

    Examples:
        >>> files = ["/path/frame1.png", "/path/frame2.png"]
        >>> img, idx, size = find_first_readable_image(files)
        >>> if img is not None:
        ...     print(f"Found at index {idx}, size {size}")
        Found at index 0, size (1920, 1080)

    Notes:
        - Returns (None, None, None) if no readable images found
        - Uses cv2.IMREAD_COLOR which loads images in BGR format
        - Corrupted or invalid image files are skipped silently
        - The returned index can be used to report which file was used
    """
    for i, file_path in enumerate(files):
        img: Optional[cv2.Mat] = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if img is not None:
            frame_size: Tuple[int, int] = (img.shape[1], img.shape[0])
            logger.debug(
                f"First readable image: {os.path.basename(file_path)} "
                f"at index {i}, size {frame_size}"
            )
            return img, i, frame_size
    
    logger.error("No readable images found in file list")
    return None, None, None


# ---------------------------------------------------------------------
# Core Processing Functions
# ---------------------------------------------------------------------


def process_camera_run(img_dir: str, out_video_path: str, fps: int) -> None:
    """
    Process a single camera folder and create a video from all images.

    This function handles the complete pipeline for converting a directory
    of image frames into a single video file:
        1. Collect and sort all image files
        2. Determine reference frame size from first readable image
        3. Initialize video writer
        4. Process each frame with automatic resizing
        5. Track and report skipped frames

    Args:
        img_dir: Directory path containing the image sequence. Should exist
            and be readable (validated by caller).
        out_video_path: Full path where the output video should be saved.
            Parent directories will be created if they don't exist.
        fps: Frames per second for the output video. Must be a positive integer.

    Returns:
        None. Results are logged and video is written to disk.

    Raises:
        OSError: If output directory cannot be created or written to.
        cv2.error: If video writer operations fail (logged, not raised).

    Side Effects:
        - Creates output directory if it doesn't exist
        - Writes video file to disk
        - Logs progress, warnings, and errors

    Examples:
        >>> process_camera_run(
        ...     "/data/cautery_tissue_001/run_001/endo_psm1",
        ...     "/videos/cautery_tissue_001/run_001/endo_psm1.mp4",
        ...     30
        ... )
        # Creates video with all frames from the directory

    Notes:
        - All frames are resized to match the first readable frame's dimensions
        - Unreadable frames are skipped and counted
        - Progress is logged every PROGRESS_INTERVAL frames (default: 500)
        - Warnings for skipped frames every SKIP_WARNING_INTERVAL (default: 50)
        - Video writer is always released, even if errors occur
        - Empty directories or directories with no readable images are skipped
    """
    print(f"Processing camera run: {img_dir}")
    
    # Collect all image files in natural order
    files: List[str] = collect_image_files(img_dir)
    
    if not files:
        logger.warning(f"No image files found in {img_dir} — skipping")
        return

    print(f"Found {len(files)} potential image files")

    # Find the first readable image to determine frame size
    first_img, first_idx, frame_size = find_first_readable_image(files)
    
    if first_img is None or first_idx is None or frame_size is None:
        logger.error(f"No readable images found in {img_dir}. Skipping camera run.")
        return

    # print(
    #     f"Reference frame: {os.path.basename(files[first_idx])}, "
    #     f"size: {frame_size[0]}x{frame_size[1]}"
    # )

    # Ensure parent directory exists for video output
    output_parent: Path = Path(out_video_path).parent
    output_parent.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured output directory exists: {output_parent}")

    # Initialize VideoWriter with automatic codec fallback
    writer, actual_out = choose_writer(out_video_path, fps, frame_size)
    
    if writer is None or actual_out is None:
        logger.error(f"Could not open VideoWriter for {out_video_path}. Skipping.")
        return

    print(f"Writing video to: {actual_out}")
    
    written: int = 0
    skipped: int = 0

    try:
        # Process each frame
        for idx, file_path in enumerate(files):
            img: Optional[cv2.Mat] = cv2.imread(file_path, cv2.IMREAD_COLOR)
            
            if img is None:
                skipped += 1
                logger.debug(f"Skipped unreadable frame: {os.path.basename(file_path)}")
                
                # Periodic warning for skipped frames
                if skipped % SKIP_WARNING_INTERVAL == 0:
                    logger.warning(f"Skipped {skipped} unreadable frames so far")
                continue

            # Ensure all frames match the reference frame size
            current_size: Tuple[int, int] = (img.shape[1], img.shape[0])
            if current_size != frame_size:
                logger.debug(
                    f"Resizing frame from {current_size} to {frame_size}: "
                    f"{os.path.basename(file_path)}"
                )
                img = cv2.resize(img, frame_size, interpolation=cv2.INTER_LINEAR)

            # Write frame to video
            writer.write(img)
            written += 1

            # Periodic progress update (downgraded to debug for parallel mode)
            if written % PROGRESS_INTERVAL == 0:
                logger.debug(f"Progress ({os.path.basename(out_video_path)}): {written}/{len(files)} frames written")

    finally:
        # Always release the writer, even if an error occurred
        writer.release()
        logger.debug("Released VideoWriter")

    # Final summary
    print(
        f"Completed: {written} frames written, {skipped} frames skipped. "
        f"Output: {actual_out}"
    )


def validate_root_directory(root_dir: str) -> bool:
    """
    Validate that the root directory exists and is accessible.

    Args:
        root_dir: Path to the root directory to validate.

    Returns:
        True if directory exists and is accessible, False otherwise.

    """
    if not os.path.isdir(root_dir):
        logger.error(f"Root directory '{root_dir}' not found or is not a directory")
        return False
    
    print(f"Validated root directory: {root_dir}")
    return True


def find_subject_directories(root_dir: str) -> List[str]:
    """
    Find all subject directories in the root directory.

    Args:
        root_dir: Path to the root directory to search.

    Returns:
        Sorted list of full paths to subject directories.
        Empty list if no matching directories found.

    Notes:
        - Results are sorted alphabetically
        - Subdirectories are not traversed
    """
    subject_dirs: List[str] = []
    try:
        with os.scandir(root_dir) as it:
            for entry in it:
                if entry.is_dir():
                    subject_dirs.append(entry.path)
    except OSError as e:
        logger.error(f"Failed to scan {root_dir}: {e}")
        
    subject_dirs.sort()
    print(f"Found {len(subject_dirs)} subject directories")
    return subject_dirs




# ---------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------


def main() -> None:
    """
    Main entry point for the frames-to-video conversion script.

    Parses command-line arguments, validates inputs, and orchestrates
    the processing pipeline:
        1. Validate root directory
        2. Find all subject directories
        3. For each subject directory, find all tissue directories
        4. For each tissue directory, find all run directories
        5. For each run, process all camera folders
        6. Convert image sequences to videos

    Command-line Arguments:
        --root_dir: Root directory containing subject folders
                   (default: "cautery")
        --fps: Frames per second for output videos (default: 30)
        --out_dir: Directory to save output videos (default: "videos")
        --dry_run: List directories without processing (default: False)
        --overwrite: Overwrite existing videos (default: False)

    Exit Codes:
        0: Success (all processing completed)
        1: No subject directories found
        2: Root directory not found or invalid

    Examples:
        # Process with defaults
        $ python3 frames_to_vids.py

        # Custom settings
        $ python3 frames_to_vids.py --root_dir data --fps 60 --overwrite

        # Preview what would be processed
        $ python3 frames_to_vids.py --dry_run

    Notes:
        - Processes directories in sorted order for reproducibility
        - Skips missing camera folders automatically
        - Continues processing even if individual runs fail
        - All errors are logged with full stack traces
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Convert image frame sequences to videos using OpenCV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    %(prog)s
    %(prog)s --root_dir mydata --fps 30
    %(prog)s --dry_run
    %(prog)s --overwrite --out_dir my_videos
            """,
        )
    
    parser.add_argument(
        "--root_dir",
        type=str,
        default=DEFAULT_ROOT_DIR,
        help=f"Root directory containing cautery_tissue* folders (default: {DEFAULT_ROOT_DIR})",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help=f"Frames per second for output videos (default: {DEFAULT_FPS})",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=DEFAULT_OUT_DIR,
        help=f"Directory to save output videos (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="List directories that would be processed without writing videos",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing video files instead of skipping them",
    )
    
    args = parser.parse_args()

    # Extract arguments
    root_dir: str = args.root_dir
    fps: int = args.fps
    out_dir: str = args.out_dir
    dry_run: bool = args.dry_run
    overwrite: bool = args.overwrite

    # Log configuration
    print("=" * 70)
    print("Starting frames-to-video conversion")
    print(f"Root directory: {root_dir}")
    print(f"Output directory: {out_dir}")
    print(f"FPS: {fps}")
    print(f"Dry run: {dry_run}")
    print(f"Overwrite: {overwrite}")
    print("=" * 70)

    # Validate root directory exists
    if not validate_root_directory(root_dir):
        sys.exit(2)

    # Find all subject directories
    subject_dirs: List[str] = find_subject_directories(root_dir)

    if not subject_dirs:
        logger.error(f"No subject directories found in {root_dir}")
        sys.exit(1)

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Collect all tasks (runs) to process
    # Build provenance-based video names: Cholec_{subject}_{tissue}_{session}.mp4
    tasks: List[Tuple[str, str, int]] = []
    
    for subject in subject_dirs:
        subject_name: str = os.path.basename(subject)

        try:
            with os.scandir(subject) as it:
                tissue_dirs = sorted([e.path for e in it if e.is_dir()])
        except OSError:
            continue
            
        for tissue in tissue_dirs:
            tissue_name: str = os.path.basename(tissue).replace("#", "")

            try:
                with os.scandir(tissue) as it:
                    run_dirs = sorted([e.path for e in it if e.is_dir()])
            except OSError:
                continue
                
            for run in run_dirs:
                run_name: str = os.path.basename(run)
                video_name: str = f"Cholec_{subject_name}_{tissue_name}_{run_name}.mp4"

                for cam in CAM_LIST:
                    img_dir = os.path.join(run, cam)
                    if os.path.isdir(img_dir):
                        out_video = os.path.join(out_dir, video_name)
                        if not os.path.exists(out_video) or overwrite:
                            tasks.append((img_dir, out_video, fps))
                        else:
                            print(f"Skipping existing video (use --overwrite to replace): {out_video}")

    # Dry run mode - just list what would be processed
    if dry_run:
        print("DRY RUN MODE - No videos will be created")
        for img_dir, out_video, _ in tasks:
            print(f"Would process camera run: {os.path.basename(os.path.dirname(img_dir))} -> {os.path.basename(out_video)}")
        return

    print(f"Found {len(tasks)} videos to process")
    
    max_workers = os.cpu_count() or 4
    print(f"Processing with up to {max_workers} threads in parallel")
    
    # Process tasks concurrently using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_camera_run, img_dir, out_video, v_fps): out_video for img_dir, out_video, v_fps in tasks}
        
        completed = 0
        total = len(tasks)
        
        # Wait for all to complete
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            out_video = futures[future]
            try:
                future.result()
                print(f"[{completed}/{total}] Created video: {os.path.basename(out_video)}")
            except Exception as e:
                logger.error(f"[{completed}/{total}] Error creating {os.path.basename(out_video)}: {e}", exc_info=True)

    # Final summary
    print("\n" + "=" * 70)
    print("Processing complete!")
    print(f"Total videos created: {completed}/{total}")
    print(f"Output location: {out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()