#!/usr/bin/env python3
"""
Normalize timestamps in ee_csv.csv files from nanoseconds to relative seconds.

This script processes all episodes in a sliced dataset and converts the first column
(timestamp) from absolute nanoseconds to relative seconds starting from 0. The conversion
formula for each row is: (current_timestamp - first_timestamp) / 1e9.

Typical usage:
    python normalize_timestamps.py --data-path dataset_sliced --workers 8

The script is compatible with the action-based slicing structure:
    dataset_sliced/tissue_N/action_subdir/episode_XXX/ee_csv.csv
"""

import argparse
import csv
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("normalize_timestamps")


def normalize_episode_timestamps(episode_path: Path) -> Tuple[int, str, str]:
    """Normalizes timestamps in a single episode's ee_csv.csv file.

    Converts the first column (timestamp) from absolute nanoseconds to relative
    seconds. The first timestamp becomes 0, and subsequent timestamps are calculated
    as (current_ns - first_ns) / 1e9.

    Args:
        episode_path: Path to the episode directory containing ee_csv.csv.

    Returns:
        A tuple containing:
            - rows_processed: Number of data rows processed (excluding header).
            - status: Status string ('ok', 'missing', 'empty', or 'error').
            - episode_name: Name of the episode directory.

    Implementation Details:
        - Preserves the header row if present
        - Detects header by attempting to convert first cell to int
        - Writes to a temporary file first, then replaces original for safety
        - Time complexity: O(n) where n is the number of rows
    """
    csv_path = episode_path / "ee_csv.csv"
    
    if not csv_path.exists():
        return 0, "missing", episode_path.name
    
    rows_processed = 0
    temp_path = csv_path.with_suffix(".csv.tmp")
    
    try:
        with csv_path.open("r", newline="") as fin:
            reader = csv.reader(fin)
            
            try:
                first_row = next(reader)
            except StopIteration:
                return 0, "empty", episode_path.name
            
            # Detect header: if first cell converts to int, it's data
            header = None
            first_data_row = None
            first_timestamp_ns = None
            
            try:
                first_timestamp_ns = int(first_row[0])
                # First row is data
                first_data_row = first_row
            except (ValueError, IndexError):
                # First row is header
                header = first_row
                # Read first data row to get initial timestamp
                try:
                    first_data_row = next(reader)
                    first_timestamp_ns = int(first_data_row[0])
                except StopIteration:
                    return 0, "empty", episode_path.name
                except (ValueError, IndexError) as e:
                    logger.error(f"Invalid timestamp in {csv_path}: {e}")
                    return 0, "error", episode_path.name
            
            # Write normalized data to temporary file
            with temp_path.open("w", newline="") as fout:
                writer = csv.writer(fout)
                
                # Write header if present
                if header is not None:
                    writer.writerow(header)
                
                # Process first data row (timestamp becomes 0.0)
                normalized_row = first_data_row.copy()
                normalized_row[0] = "0.0"
                writer.writerow(normalized_row)
                rows_processed += 1
                
                # Process remaining rows
                for row in reader:
                    try:
                        current_timestamp_ns = int(row[0])
                        # Convert to relative seconds
                        relative_seconds = (current_timestamp_ns - first_timestamp_ns) / 1e9
                        
                        normalized_row = row.copy()
                        normalized_row[0] = f"{relative_seconds:.9f}"
                        writer.writerow(normalized_row)
                        rows_processed += 1
                        
                    except (ValueError, IndexError) as e:
                        logger.warning(
                            f"Skipping invalid row in {csv_path}: {e}"
                        )
                        continue
        
        # Replace original file with normalized version
        temp_path.replace(csv_path)
        return rows_processed, "ok", episode_path.name
        
    except Exception as e:
        logger.error(f"Failed to normalize {csv_path}: {e}")
        # Clean up temp file if it exists
        if temp_path.exists():
            temp_path.unlink()
        return rows_processed, "error", episode_path.name


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
    for dirpath, dirs, files in os.walk(root):
        if "ee_csv.csv" in files:
            episodes.append(Path(dirpath))
    return sorted(episodes)


def main() -> None:
    """Main entry point for the timestamp normalization script.

    Parses command-line arguments, discovers episodes, and orchestrates parallel
    normalization of timestamps across all episodes in the dataset.
    """
    parser = argparse.ArgumentParser(
        description="Normalize timestamps in ee_csv.csv files from nanoseconds to relative seconds."
    )
    parser.add_argument(
        "--data-path",
        required=True,
        type=Path,
        help="Root dataset folder (e.g., dataset_sliced)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel processes (default: cpu count)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without modifying files",
    )
    
    args = parser.parse_args()
    
    if not args.data_path.exists():
        logger.error(f"Dataset path not found: {args.data_path}")
        sys.exit(1)
    
    # Find all episodes
    logger.info("Scanning for episodes...")
    episodes = find_episodes(args.data_path)
    logger.info(f"Found {len(episodes)} episodes to process.")
    
    if not episodes:
        logger.warning("No episodes found. Exiting.")
        sys.exit(0)
    
    if args.dry_run:
        logger.info(f"DRY RUN: Would normalize timestamps in {len(episodes)} episodes.")
        for ep in episodes[:5]:  # Show first 5 as examples
            logger.info(f"  - {ep}")
        if len(episodes) > 5:
            logger.info(f"  ... and {len(episodes) - 5} more")
        sys.exit(0)
    
    # Process episodes in parallel
    workers = args.workers if args.workers else (os.cpu_count() or 4)
    logger.info(f"Starting timestamp normalization with {workers} workers...")
    
    total_rows = 0
    total_episodes_done = 0
    errors = 0
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(normalize_episode_timestamps, ep_path): ep_path
            for ep_path in episodes
        }
        
        for future in as_completed(futures):
            ep_path = futures[future]
            try:
                rows, status, name = future.result()
                total_rows += rows
                total_episodes_done += 1
                
                if status == "error":
                    errors += 1
                    logger.warning(
                        f"[{total_episodes_done}/{len(episodes)}] Error in {name}"
                    )
                elif status == "missing":
                    logger.debug(f"[{total_episodes_done}/{len(episodes)}] No CSV in {name}")
                else:
                    if total_episodes_done % 10 == 0 or total_episodes_done == len(episodes):
                        print(
                            f"Progress: [{total_episodes_done}/{len(episodes)}] episodes...",
                            end="\r",
                            flush=True,
                        )
                        
            except Exception as ex:
                errors += 1
                logger.error(f"Failed to process episode {ep_path}: {ex}")
    
    print("")  # Newline after progress bar
    logger.info("Timestamp normalization complete.")
    logger.info(f"Total Episodes Processed: {total_episodes_done}")
    logger.info(f"Total Rows Normalized: {total_rows}")
    if errors > 0:
        logger.warning(f"Errors Encountered: {errors}")


if __name__ == "__main__":
    main()
