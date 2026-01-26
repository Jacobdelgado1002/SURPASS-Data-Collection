#!/usr/bin/env python3
"""
frames_to_vids.py

Traverse a 'cautery' folder structure and convert images in camera folders
(endo_psm1, endo_psm2, left_img_dir, right_img_dir) into videos (one per camera/run).

Usage:
    python3 frames_to_vids.py
    python3 frames_to_vids.py --root-dir mydata --fps 30

Dependencies:
    pip install opencv-python
"""

import os
import sys
import cv2
import argparse
import re
from typing import List, Tuple, Optional

# Supported camera folders
CAM_LIST: Tuple[str, ...] = ("endo_psm1", "endo_psm2", "left_img_dir", "right_img_dir")


def natural_key(s: str) -> List:
    """
    Generate a natural sort key for a string containing numbers.

    Splits digits and non-digits, converting digits to integers
    so that "frame2" < "frame10".

    Args:
        s (str): Input string to generate sort key.

    Returns:
        List: A list of integers and strings for natural sorting.
    """
    parts = re.split(r"(\d+)", s)
    key: List = []
    for p in parts:
        key.append(int(p) if p.isdigit() else p.lower())
    return key


def collect_image_files(img_dir: str) -> List[str]:
    """
    Collect image files from a directory, skipping non-image files like CSV.

    Args:
        img_dir (str): Directory containing image files.

    Returns:
        List[str]: Sorted list of image file paths in natural order.
    """
    entries: List[str] = []
    # Iterate over directory entries
    for name in os.listdir(img_dir):
        full_path = os.path.join(img_dir, name)
        if not os.path.isfile(full_path):
            continue
        # Skip CSV or other non-image files
        if name.lower().endswith(".csv"):
            continue
        entries.append(full_path)

    # Sort based on the basename
    entries.sort(key=lambda p: natural_key(os.path.basename(p)))
    return entries


def choose_writer(path_out: str, fps: int, frame_size: Tuple[int, int]) -> Tuple[Optional[cv2.VideoWriter], Optional[str]]:
    """
    Attempt to create a cv2.VideoWriter for the given path, FPS, and frame size.

    Args:
        path_out (str): Desired output video file path.
        fps (int): Frames per second for the video.
        frame_size (Tuple[int, int]): Width and height of frames (w, h).

    Returns:
        Tuple[Optional[cv2.VideoWriter], Optional[str]]:
            VideoWriter object if successful, and the actual output path used.
            Returns (None, None) if no suitable writer could be opened.
    """
    # Primary attempt: MP4 using mp4v
    writer = cv2.VideoWriter(path_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
    if writer.isOpened():
        return writer, path_out

    return None, None


def process_camera_run(img_dir: str, out_video_path: str, fps: int) -> None:
    """
    Process a single camera folder and create a video from all images.

    Args:
        img_dir (str): Directory containing images for this camera run.
        out_video_path (str): Output video file path.
        fps (int): Frames per second for the output video.

    Notes:
        - Skips unreadable images but reports the count.
        - Resizes all images to match the first readable frame.
        - Prints progress for every 500 frames written.
    """
    files = collect_image_files(img_dir)
    if not files:
        print(f"No files in {img_dir} — skipping.")
        return

    # Determine frame size from first readable image
    first_img: Optional[cv2.Mat] = None
    first_idx: Optional[int] = None
    # Find the first readable image
    for i, f in enumerate(files):
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        if img is not None:
            first_img = img
            first_idx = i
            break

    if first_img is None:
        print(f"[ERROR] No readable images found in {img_dir}. Skipping.")
        return
    
    frame_size: Tuple[int, int] = (first_img.shape[1], first_img.shape[0])
    print(f"Found {len(files)} files. First readable: {os.path.basename(files[first_idx])}, frame size: {frame_size}")

    # Ensure parent directory exists for video output
    os.makedirs(os.path.dirname(out_video_path), exist_ok=True)
      
    # Initialize VideoWriter
    writer, actual_out = choose_writer(out_video_path, fps, frame_size)
    if writer is None:
        print(f"[ERROR] Could not open VideoWriter for {out_video_path}. Skipping.")
        return

    print(f"Writing video to: {actual_out}")
    written, skipped = 0, 0

    # Write frames to video
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        if img is None:
            skipped += 1
            if skipped % 50 == 0:
                print(f"  Skipped {skipped} unreadable frames so far...")
            continue

        # Ensure all frames match first frame size
        if (img.shape[1], img.shape[0]) != frame_size:
            img = cv2.resize(img, frame_size, interpolation=cv2.INTER_LINEAR)
        
        writer.write(img)
        written += 1

        if written % 500 == 0:
            print(f"  Written {written}/{len(files)} frames...")

    # Release the writer
    writer.release()
    print(f"Finished: {written} frames written, {skipped} skipped. Output: {actual_out}")


def main() -> None:
    """
    Parses command-line arguments, traverses the root directory,
    and processes each camera folder to generate videos.
    """
    parser = argparse.ArgumentParser(description="Convert image frames to videos (OpenCV).")
    parser.add_argument("--root-dir", type=str, default="cautery",
                        help="Root directory containing cautery_tissue* folders")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frames per second for output videos")
    parser.add_argument("--out-dir", type=str, default="videos",
                        help="Directory to save output videos (preserves folder structure)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List directories that would be processed without writing videos")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing video files instead of skipping them")
    args = parser.parse_args()

    root_dir = args.root_dir    # Root directory with cautery_tissue* folders
    fps = args.fps              # Frames per second for output videos
    out_dir = args.out_dir     # Output root directory
    overwrite = args.overwrite  # Whether to overwrite existing videos

    if not os.path.isdir(root_dir):
        print(f"[ERROR] Root directory '{root_dir}' not found.", file=sys.stderr)
        sys.exit(2)

    # List all tissue directories sorted naturally
    tissue_dirs = sorted(
        [os.path.join(root_dir, d) for d in os.listdir(root_dir)
         if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("cautery_tissue")]
    )

    # Exit if no tissue directories found
    if not tissue_dirs:
        print(f"No 'cautery_tissue*' directories found in {root_dir}. Exiting.")
        return

    # Iterate over tissue directories, runs, and camera folders
    for tissue in tissue_dirs:
        run_dirs = sorted([os.path.join(tissue, r) for r in os.listdir(tissue)
                           if os.path.isdir(os.path.join(tissue, r))])
        # Process each run directory
        for run in run_dirs:
            for cam in CAM_LIST:
                img_dir = os.path.join(run, cam)
                if not os.path.isdir(img_dir):
                    continue

                # Preserve relative structure under out_dir
                rel_path = os.path.relpath(run, root_dir)
                out_video = os.path.join(out_dir, rel_path, f"{cam}.mp4")

                # Skip if output exists and overwrite not enabled
                if os.path.exists(out_video) and not overwrite:
                    print(f"[SKIP] Video already exists and overwrite is False: {out_video}")
                    continue

                try:
                    process_camera_run(img_dir, out_video, fps)
                except Exception as e:
                    print(f"[EXCEPTION] Error processing {img_dir}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
