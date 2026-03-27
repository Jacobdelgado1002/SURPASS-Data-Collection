#!/usr/bin/env python3
"""
frames_to_vids.pOp
This script traverses a hierarchical directory structure of medical imaging data
and converts image sequences into MP4 videos. It is specifically designed for 
high-performance computing by optimizing algorithmic complexity and utilizing
multi-processing.

Expected Directory Structure:
    Root/
        CollectorName/
            Tissue#N/
                RunID (Timestamp)/
                    left_img_dir/
                        frame_0001.png
                        ...

Performance Optimizations:
    1. Multi-processing (ProcessPoolExecutor): Bypasses Python's Global Interpreter Lock (GIL)
       to achieve true parallel execution across multiple CPU cores.
    2. O(1) Lookups (frozenset): Uses hash-based lookups for file extension filtering,
       reducing complexity from O(K) to O(1) per file.
    3. Locality Optimization: Caches global function references (e.g., cv2.imread) into 
       local variables within tight loops to reduce look-up overhead during byte-code execution.
    4. Minimal Object Allocation: Compares image dimensions directly via array slices
       to avoid redundant tuple creation in high-frequency loops.
"""

import argparse
import concurrent.futures
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Set

import cv2
from tqdm import tqdm

from surpass_data_collection.logger_config import get_logger

# ---------------------------------------------------------------------
# Module Constants & Configuration
# ---------------------------------------------------------------------

# Only process the left camera directory as per current requirement
CAM_LIST: Tuple[str, ...] = ("left_img_dir",)

# Use frozenset for O(1) membership lookups to filter out non-image files efficiently
SKIP_EXTENSIONS_SET: Set[str] = frozenset(ext.lower() for ext in (".csv", ".txt", ".json", ".xml"))

# Video encoding settings
PRIMARY_CODEC, FALLBACK_CODEC = "mp4v", "MJPG"

# Logging and reporting intervals
PROGRESS_INTERVAL, SKIP_WARNING_INTERVAL = 500, 50

# Default execution parameters
DEFAULT_FPS = 30
DEFAULT_ROOT_DIR = r"D:\Data\Cholecystectomy"
DEFAULT_OUT_DIR = r"D:\Data\Cholecystectomy_videos"

# Initialize structured logger
logger = get_logger(__name__)

# Regex for splitting strings into numeric and non-numeric parts for natural sorting
_DIGIT_SPLIT_RE = re.compile(r"(\d+)")


# ---------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------

def natural_key(s: str) -> List:
    """
    Generate a sort key that orders strings naturally (e.g., 'frame2' before 'frame10').
    
    Args:
        s: The string to generate a key for.
        
    Returns:
        A list of alternating strings and integers.
    """
    return [int(p) if p.isdigit() else p.lower() for p in _DIGIT_SPLIT_RE.split(s)]


def collect_image_files(img_dir: str) -> List[str]:
    """
    Collects image files from a directory, filtering by extension and sorting naturally.
    
    Optimized with O(1) membership check for skipped extensions.
    
    Args:
        img_dir: Path to the directory containing image frames.
        
    Returns:
        A list of full paths to image files, sorted in natural order.
    """
    entries: List[str] = []
    try:
        # os.scandir is faster than os.listdir as it avoids unnecessary stat calls
        with os.scandir(img_dir) as it:
            for entry in it:
                if entry.is_file():
                    _, ext = os.path.splitext(entry.name)
                    # O(1) hash lookup is used here to scale linearly with file count
                    if ext.lower() not in SKIP_EXTENSIONS_SET:
                        entries.append(entry.name)
    except OSError as e:
        logger.error(f"Failed to scan {img_dir}: {e}")
        return []

    # Sort based on natural order of file names
    entries.sort(key=natural_key)
    return [os.path.join(img_dir, name) for name in entries]


def choose_writer(path_out: str, fps: int, frame_size: Tuple[int, int]) -> Tuple[Optional[cv2.VideoWriter], Optional[str]]:
    """
    Attempts to initialize a cv2.VideoWriter with a primary codec, falling back if needed.
    
    Args:
        path_out: Desired output path.
        fps: Frames per second.
        frame_size: (width, height) of the video.
        
    Returns:
        A tuple of (VideoWriter instance, actual output path) or (None, None) if failed.
    """
    # Try primary MP4V codec first
    writer = cv2.VideoWriter(path_out, cv2.VideoWriter_fourcc(*PRIMARY_CODEC), fps, frame_size)
    if writer.isOpened(): 
        return writer, path_out
    
    # Fallback to MJPG in .avi format if MP4V is unsupported by the environment
    fallback = str(Path(path_out).with_suffix(".avi"))
    writer = cv2.VideoWriter(fallback, cv2.VideoWriter_fourcc(*FALLBACK_CODEC), fps, frame_size)
    if writer.isOpened(): 
        return writer, fallback
        
    return None, None


def find_first_readable_image(files: List[str]) -> Tuple[Optional[cv2.Mat], Optional[int], Optional[Tuple[int, int]]]:
    """
    Iterates through file paths to find the first valid image and determine its dimensions.
    
    Args:
        files: List of file paths to check.
        
    Returns:
        A tuple of (image matrix, index, (width, height)) or (None, None, None).
    """
    for i, file_path in enumerate(files):
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if img is not None: 
            # Slice shape[1::-1] to get (width, height) efficiently
            return img, i, img.shape[1::-1]
    return None, None, None


# ---------------------------------------------------------------------
# Core Processing Functions
# ---------------------------------------------------------------------

def process_camera_run(img_dir: str, out_video_path: str, fps: int) -> None:
    """
    Converts a single directory of images into a video file.
    
    This function handles the frame-by-frame processing loop and is designed
    to run as a standalone process in the ProcessPoolExecutor.
    """
    files = collect_image_files(img_dir)
    if not files:
        return

    # Find reference frame size from the first valid image
    first_img, first_idx, frame_size = find_first_readable_image(files)
    if frame_size is None:
        logger.error(f"No readable images in {img_dir}")
        return

    # Create output directory hierarchy if it doesn't exist
    Path(out_video_path).parent.mkdir(parents=True, exist_ok=True)
    
    writer, actual_out = choose_writer(out_video_path, fps, frame_size)
    if writer is None:
        logger.error(f"Failed to open VideoWriter for {out_video_path}")
        return

    target_w, target_h = frame_size
    written, skipped = 0, 0

    # Optimization: Cache global functions to local variables.
    # This reduces dictionary lookups in the built-ins/globals namespace during the loop,
    # significantly speeding up the processing of millions of frames.
    imread_local = cv2.imread
    resize_local = cv2.resize
    IMREAD_COLOR_VAL = cv2.IMREAD_COLOR
    INTER_LINEAR_VAL = cv2.INTER_LINEAR

    try:
        for file_path in files:
            img = imread_local(file_path, IMREAD_COLOR_VAL)
            if img is None:
                skipped += 1
                continue
            
            # Efficiently check dimensions without creating a new tuple
            h, w = img.shape[:2]
            if w != target_w or h != target_h:
                img = resize_local(img, frame_size, interpolation=INTER_LINEAR_VAL)
                
            writer.write(img)
            written += 1
            
            # Periodic logging for visibility into long-running tasks
            if written % PROGRESS_INTERVAL == 0:
                logger.debug(f"[{os.path.basename(actual_out)}] Processed {written}/{len(files)} frames")
                
    finally:
        # Ensure the writer is always released to avoid file corruption
        writer.release()
    
    # Final print is moved out or silenced to avoid interfering with tqdm
    # logger.info(f"Completed: {written} written, {skipped} skipped -> {actual_out}")


# ---------------------------------------------------------------------
# Main Orchestration
# ---------------------------------------------------------------------

def main() -> None:
    """
    Main entry point: parses arguments, builds task list, and executes sub-processes.
    """
    parser = argparse.ArgumentParser(description="Batch convert image sequences to videos.")
    parser.add_argument("--root_dir", type=str, default=DEFAULT_ROOT_DIR, help="Source data root")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Frames per second")
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR, help="Video output directory")
    parser.add_argument("--dry_run", action="store_true", help="Print tasks without processing")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing videos")
    args = parser.parse_args()

    # Basic input validation
    if not os.path.isdir(args.root_dir):
        print(f"Error: Root directory not found: {args.root_dir}")
        sys.exit(2)
        
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("=" * 70)
    print(f"Starting Video Conversion")
    print(f"Source: {args.root_dir}")
    print(f"Destination: {args.out_dir}")
    print("=" * 70)

    # Gather tasks by traversing the specific data hierarchy
    tasks: List[Tuple[str, str, int]] = []
    
    # Nested loops use os.scandir internally via list comprehensions for speed
    # We sort each level to ensure deterministic processing order
    try:
        # Level 1: Collector Directories
        collectors = sorted([e for e in os.scandir(args.root_dir) if e.is_dir()], key=lambda e: e.name)
        for collector in collectors:
            collector_name = collector.name
            
            # Level 2: Tissue Directories
            tissues = sorted([e for e in os.scandir(collector.path) if e.is_dir()], key=lambda e: e.name)
            for tissue in tissues:
                tissue_name = tissue.name.replace("#", "")
                
                # Level 3: Run (Session) Directories
                runs = sorted([e for e in os.scandir(tissue.path) if e.is_dir()], key=lambda e: e.name)
                for run in runs:
                    # Check each camera directory in the run
                    for cam in CAM_LIST:
                        img_dir = os.path.join(run.path, cam)
                        if os.path.isdir(img_dir):
                            # Construct unified video filename
                            out_video = os.path.join(args.out_dir, f"{os.path.basename(args.root_dir)}_{collector_name}_{tissue_name}_{run.name}.mp4")
                            
                            # Skip if video exists and overwrite is not requested
                            if not os.path.exists(out_video) or args.overwrite:
                                tasks.append((img_dir, out_video, args.fps))
                            else:
                                print(f"Skipping (exists): {os.path.basename(out_video)}")
    except OSError as e:
        logger.error(f"Error during directory traversal: {e}")

    if not tasks:
        print("No tasks found to process.")
        return

    if args.dry_run:
        print(f"Dry run: Found {len(tasks)} tasks.")
        for img_dir, out_path, _ in tasks:
            print(f"Would process camera run: {os.path.basename(os.path.dirname(img_dir))} -> {os.path.basename(out_path)}")
        return

    print(f"Found {len(tasks)} videos to process.")
    
    # Process tasks concurrently using ProcessPoolExecutor
    max_cpus = os.cpu_count() or 4
    print(f"Allocating {max_cpus} parallel processes...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_cpus) as executor:
        # Wrap future submissions in as_completed within a tqdm progress bar
        futures = [executor.submit(process_camera_run, *t) for t in tasks]
        
        # tqdm progress bar tracks the completion of each 'future'
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc="Converting Videos"):
            pass

    print("\n" + "=" * 70)
    print("Processing complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()