#!/usr/bin/env python3
"""
Normalize image frame names in-place to sequential 0-indexed patterns.

This script processes dataset episodes and renames all image frames to a standardized
format: frame000000_left.jpg, frame000001_right.jpg, etc. It uses a two-pass rename
strategy to avoid filename collisions and supports parallel processing for efficiency.

Typical usage:
    python normalize_frame_names.py --data-path dataset_sliced --workers 12

The script is compatible with the action-based slicing structure:
    dataset_sliced/tissue_N/action_subdir/episode_XXX/
"""

import argparse
import logging
import os
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("normalize_dataset")

# Expected subdirectories and their suffixes
IMAGE_DIRS = {
    "left_img_dir": "_left",
    "right_img_dir": "_right",
    "endo_psm1": "_psm1",
    "endo_psm2": "_psm2",
}


def normalize_episode_inplace(
    episode_path: Path,
    sort_by: str = "name"
) -> Tuple[int, int, str]:
    """Normalizes image filenames within a single episode directory.

    This function processes all image subdirectories (left_img_dir, right_img_dir,
    endo_psm1, endo_psm2) within an episode and renames files to a sequential
    0-indexed format. Uses a two-pass rename strategy to prevent collisions.

    Args:
        episode_path: Path to the episode directory containing image subdirectories.
        sort_by: Sorting criterion for determining frame order. Either 'name' for
                 lexicographic sorting or 'mtime' for modification time sorting.

    Returns:
        A tuple containing:
            - images_renamed: Number of images successfully renamed.
            - errors: Number of errors encountered during renaming.
            - episode_name: Name of the episode directory processed.

    Implementation Details:
        Pass 1: Renames files to temporary UUID-based names to avoid collisions.
        Pass 2: Renames temporary files to final normalized format (frameNNNNNN_suffix.ext).
        This ensures correctness even when re-running on partially normalized data.
    """
    images_renamed = 0
    errors = 0

    # Process each camera view
    for img_subdir, suffix in IMAGE_DIRS.items():
        src_dir = episode_path / img_subdir
        if not src_dir.is_dir():
            continue

        # Collect and sort files
        files = [p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in ('.jpg', '.png')]
        
        if sort_by == "mtime":
            files.sort(key=lambda p: p.stat().st_mtime)
        else:
            files.sort(key=lambda p: p.name)

        if not files:
            continue

        # Pass 1: Rename to temporary names to avoid collision (e.g. if frame000000 already existed but was at wrong offset)
        temp_renames = []
        unique_run_id = uuid.uuid4().hex[:8]
        
        for idx, src_file in enumerate(files):
            ext = src_file.suffix.lower()
            temp_name = f"tmp_{unique_run_id}_{idx:06d}{suffix}{ext}"
            temp_file = src_dir / temp_name
            try:
                os.rename(src_file, temp_file)
                temp_renames.append((temp_file, idx, ext))
            except Exception as e:
                logger.error(f"Failed to rename {src_file.name} to temporary name: {e}")
                errors += 1

        # Pass 2: Rename from temporary to final normalized names
        for temp_file, idx, ext in temp_renames:
            final_name = f"frame{idx:06d}{suffix}{ext}"
            final_file = src_dir / final_name
            try:
                # Use os.replace to allow overwriting if something went wrong in Pass 1 or if we are re-running
                os.replace(temp_file, final_file)
                images_renamed += 1
            except Exception as e:
                logger.error(f"Failed to rename {temp_file.name} to final normalized name: {e}")
                errors += 1
    
    return images_renamed, errors, episode_path.name


def find_episodes(root: Path) -> List[Path]:
    """Recursively discovers all episode directories in the dataset.

    An episode directory is identified by the presence of an 'ee_csv.csv' file,
    which contains the kinematics data for that episode.

    Args:
        root: Root directory to search for episodes.

    Returns:
        Sorted list of Path objects pointing to episode directories.

    Performance:
        Uses os.walk for efficient traversal of large directory trees (O(n) where
        n is the total number of directories).
    """
    episodes = []
    # Using os.walk is faster than recursive glob for large trees
    for dirpath, dirs, files in os.walk(root):
        if "ee_csv.csv" in files:
            episodes.append(Path(dirpath))
    return sorted(episodes)


def main() -> None:
    """Main entry point for the frame normalization script.

    Parses command-line arguments, discovers episodes, and orchestrates parallel
    normalization of frame names across all episodes in the dataset.
    """
    parser = argparse.ArgumentParser(
        description="Normalize filtered dataset frame names in-place for training."
    )
    parser.add_argument(
        "--data-path", 
        required=True, 
        type=Path, 
        help="Source dataset folder (e.g. dataset_sliced)"
    )
    parser.add_argument(
        "--sort-by", 
        choices=("name", "mtime"), 
        default="name", 
        help="Sort criteria for original frames (default: name)"
    )
    parser.add_argument(
        "--workers", 
        type=int, 
        default=None, 
        help="Number of parallel processes (default: cpu count)"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Print what would happen without renaming"
    )

    args = parser.parse_args()

    if not args.data_path.exists():
        logger.error(f"Source path not found: {args.data_path}")
        sys.exit(1)

    if args.dry_run:
        logger.info(f"DRY RUN: Would normalize frame names in {args.data_path}")
        episodes = find_episodes(args.data_path)
        print(f"Found {len(episodes)} episodes.")
        sys.exit(0)

    # 1. Find Episodes
    logger.info("Scanning for episodes...")
    episodes = find_episodes(args.data_path)
    logger.info(f"Found {len(episodes)} episodes to process.")

    if not episodes:
        logger.warning("No episodes found. Exiting.")
        sys.exit(0)

    # 2. Process in Parallel
    total_images = 0
    total_episodes_done = 0
    
    workers = args.workers if args.workers else (os.cpu_count() or 4)
    logger.info(f"Starting in-place normalization with {workers} workers...")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(normalize_episode_inplace, ep_path, args.sort_by): ep_path for ep_path in episodes}

        # Monitor progress
        for future in as_completed(futures):
            src_ep = futures[future]
            try:
                c, e, name = future.result()
                total_images += c
                total_episodes_done += 1
                
                if e > 0:
                    logger.warning(f"[{total_episodes_done}/{len(episodes)}] Finished {src_ep.name} with {e} errors ({c} images ok).")
                else:
                    if total_episodes_done % 10 == 0 or total_episodes_done == len(episodes):
                        print(f"Progress: [{total_episodes_done}/{len(episodes)}] episodes...", end='\r', flush=True)
            except Exception as ex:
                logger.error(f"Failed to process episode {src_ep}: {ex}")

    print("") # Newline after progress bar
    logger.info("Normalization Complete.")
    logger.info(f"Total Episodes: {total_episodes_done}")
    logger.info(f"Total Images Renamed: {total_images}")

if __name__ == "__main__":
    main()
