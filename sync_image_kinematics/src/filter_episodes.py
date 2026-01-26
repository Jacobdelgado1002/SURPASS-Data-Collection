#!/usr/bin/env python3
"""
Script to filter and copy synchronized episodes from source to destination directory.

This script:
1. Processes all episodes in the source directory
2. Runs synchronization and outlier removal on each episode (using sync_image_kinematics module)
3. Copies only valid (synchronized) images and kinematics to filtered dataset
4. Maintains the original naming convention (left_img_dir, ee_csv.csv)
"""

import os
import shutil
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Any, Union, Tuple
import argparse
import sys
import bisect
import re
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Add the current directory to sys.path to allow importing sync_image_kinematics
# This assumes the script is run from the directory containing both files or src/
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

try:
    from sync_image_kinematics import process_episode_sync, extract_timestamp_from_filename
except ImportError:
    # Fallback if running from root relative to src
    try:
        from src.sync_image_kinematics import process_episode_sync, extract_timestamp_from_filename
    except ImportError:
         print("Error: Could not import sync_image_kinematics. Make sure it is in the same directory or python path.")
         sys.exit(1)

def write_filtered_kinematics(
    dest_episode_dir: Path,
    sync_df: pd.DataFrame,
    validation: Dict[str, Any],
    kept_filenames: List[str],
) -> None:
    """Write filtered kinematics CSV corresponding to kept image filenames.

    Args:
        dest_episode_dir: Destination episode directory.
        sync_df: Synchronization DataFrame returned by sync_image_kinematics.
        validation: Validation dictionary containing original kinematics file path.
        kept_filenames: List of left-camera filenames that were kept.
    """
    if sync_df.empty:
        return

    kept_set = set(kept_filenames)
    mask = sync_df['image_filename'].isin(kept_set)
    final_sync_df = sync_df[mask]

    if final_sync_df.empty:
        return

    # Preserve 1-to-1 mapping (allow duplicates)
    kinematics_indices = final_sync_df['kinematics_idx'].values

    original_kinematics = pd.read_csv(validation['kinematics_file'])

    filtered_kinematics = original_kinematics.loc[kinematics_indices].copy()
    filtered_kinematics.reset_index(drop=True, inplace=True)

    dest_csv = dest_episode_dir / "ee_csv.csv"
    filtered_kinematics.to_csv(dest_csv, index=False)

def find_episodes(source_dir: Union[str, Path]) -> List[Path]:
    """Recursively finds all directories containing valid data.

    Valid data directories must contain 'left_img_dir' and 'ee_csv.csv'.

    Args:
        source_dir: Path to the source directory to search as a string or Path object.

    Returns:
        A list of Path objects pointing to valid episode directories.

    Raises:
        FileNotFoundError: If the source directory does not exist.
    """
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    data_dirs: List[Path] = []
    
    # Walk through all directories
    for root, dirs, files in os.walk(source_path):
        current_path = Path(root)
        
        # Check if this directory looks like a data directory
        # It must have 'left_img_dir' subdirectory and 'ee_csv.csv' file
        has_img_dir = (current_path / "left_img_dir").is_dir()
        has_csv = (current_path / "ee_csv.csv").exists()
        
        if has_img_dir and has_csv:
            data_dirs.append(current_path)

    data_dirs.sort()
    print(f"Found {len(data_dirs)} data directories in {source_dir}")
    return data_dirs


def validate_episode_structure(episode_path: Path) -> Dict[str, Any]:
    """Validates that an episode has the required structure (left images and kinematics).

    Args:
        episode_path: Path to the episode directory.

    Returns:
        A dictionary with validation results:
        - has_left_images: bool
        - has_kinematics: bool
        - left_img_dir: Path or None
        - kinematics_file: Path or None
    """
    validation = {
        'has_left_images': False,
        'has_kinematics': False,
        'left_img_dir': None,
        'kinematics_file': None
    }
    
    # Check for left image directory
    left_img_dir = episode_path / "left_img_dir"
    if left_img_dir.exists() and left_img_dir.is_dir():
        # Check if it has any jpg files
        # Inline comment: Using globo enables us to quickly verify content without loading all files
        if any(left_img_dir.glob("*.jpg")):
            validation['has_left_images'] = True
            validation['left_img_dir'] = left_img_dir
    
    # Check for kinematics CSV file
    ee_csv = episode_path / "ee_csv.csv"
    if ee_csv.exists():
        validation['has_kinematics'] = True
        validation['kinematics_file'] = ee_csv
    
    return validation


def run_sync_analysis_direct(
    episode_path: Path, max_time_diff: float = 30.0
) -> Dict[str, Any]:
    """Runs the synchronization analysis directly using the imported module.
    """
    # optimization: don't write intermediate CSVs to disk during batch filtering
    result = process_episode_sync(
        episode_path=episode_path,
        camera="left",
        output_dir=None,
        max_time_diff_ms=max_time_diff,
        plot=False,
        save_results=False
    )
    
    return result


def copy_filtered_episode(
    episode_path: Path, 
    dest_dir: Path, 
    sync_result: Dict[str, Any], 
    validation: Dict[str, Any], 
    source_root: Path, 
    use_hardlink: bool = False,
    max_time_diff: float = 30.0
) -> bool:
    """Copies filtered episode data to the destination directory.

    Uses approximate timestamp matching for multi-camera synchronization.
    Enforces strict synchronization: A frame is kept only if ALL cameras have a match.
    Renames secondary camera frames to match the Left timestamp to ensure 1:1 correspondence.

    Args:
        episode_path: Source episode path.
        dest_dir: Destination root directory.
        sync_result: Result dictionary from sync analysis.
        validation: Validation dictionary.
        source_root: Root of the source directory (for relative path structure).
        use_hardlink: If True, uses hardlinks instead of copying (faster).
        max_time_diff: Maximum time difference for sync in milliseconds.

    Returns:
        True if at least one fully synchronized frame was copied, False otherwise.
    """
    try:
        # Preserve relative path under destination
        try:
            rel = episode_path.relative_to(source_root)
        except ValueError:
            # Fallback if source_root is not a parent
            rel = Path(episode_path.name)
            
        dest_episode_dir = dest_dir / rel
        dest_episode_dir.mkdir(parents=True, exist_ok=True)
        
        # Valid filenames from left camera (the synchronization source)
        left_filenames = sync_result['valid_filenames']
        
        # 0. Preparation: available cameras and their files
        cameras = [
            ("left_img_dir", "_left.jpg", "left_img_dir"),
            ("right_img_dir", "_right.jpg", "right_img_dir"),
            ("endo_psm1", "_psm1.jpg", "endo_psm1"),
            ("endo_psm2", "_psm2.jpg", "endo_psm2")
        ]
        
        # Load timestamps for secondary cameras
        camera_indices = {} # name -> [(ts, filename), ...]
        
        for src_name, suffix, dst_name in cameras:
            if src_name == "left_img_dir": continue
            
            src_dir = episode_path / src_name
            candidates = []
            if src_dir.exists():
                # Faster directory scanning with os.scandir
                with os.scandir(src_dir) as entries:
                    for entry in entries:
                        if entry.is_file() and entry.name.endswith('.jpg') and suffix in entry.name:
                            try:
                                ts = extract_timestamp_from_filename(entry.name)
                                candidates.append((ts, entry.name))
                            except ValueError:
                                continue
            candidates.sort(key=lambda x: x[0])
            camera_indices[src_name] = candidates

        # Pre-calculate timestamp lists for search speed
        camera_timestamp_lists = {
            name: [x[0] for x in candidates] 
            for name, candidates in camera_indices.items()
        }

        # 1. Identify Fully Valid Frames (All cameras match)
        fully_valid_frames = [] # List of tuples: (left_filename, {cam_name: match_filename})
        
        MAX_CAMERA_SYNC_DIFF_NS = max_time_diff * 1e6 # Use provided tolerance

        for left_fname in left_filenames:
            try:
                left_ts = extract_timestamp_from_filename(left_fname)
            except ValueError:
                continue
                
            matches = {}
            drop_frame = False
            
            for src_name, _, _ in cameras:
                if src_name == "left_img_dir": continue
                
                # If camera didn't exist in source, we skip checking it (permissive for data missing whole views)
                if src_name not in camera_timestamp_lists:
                    continue
                    
                candidate_timestamps = camera_timestamp_lists[src_name]
                if not candidate_timestamps:
                    drop_frame = True # Camera exists but empty?
                    break
                    
                idx = bisect.bisect_left(candidate_timestamps, left_ts)
                
                # Check idx and idx-1
                candidates = camera_indices[src_name]
                best_match_file = None
                best_diff = float('inf')
                
                check_indices = []
                if idx < len(candidates): check_indices.append(idx)
                if idx > 0: check_indices.append(idx-1)
                
                for i in check_indices:
                    diff = abs(candidates[i][0] - left_ts)
                    if diff < best_diff:
                        best_diff = diff
                        best_match_file = candidates[i][1]
                        
                if best_match_file and best_diff <= MAX_CAMERA_SYNC_DIFF_NS:
                    matches[src_name] = best_match_file
                else:
                    drop_frame = True
                    break
            
            if not drop_frame:
                fully_valid_frames.append((left_fname, matches))

        # Inline comment: Log sync success rate for this episode
        # print(f"  {len(fully_valid_frames)}/{len(left_filenames)} frames fully synchronized across cameras.")

        # 2. Copy Files
        if not fully_valid_frames:
             return False

        # Create dirs
        for _, _, dst_name in cameras:
            (dest_episode_dir / dst_name).mkdir(exist_ok=True)

        copied_count = 0
        
        for left_fname, matches in fully_valid_frames:
            # Copy Left
            src_left = episode_path / "left_img_dir" / left_fname
            dst_left = dest_episode_dir / "left_img_dir" / left_fname
            
            if src_left.exists():
                # Copy/Link Left
                if use_hardlink:
                    try:
                        if dst_left.exists(): dst_left.unlink()
                        os.link(src_left, dst_left)
                    except OSError:
                        shutil.copy2(src_left, dst_left)
                else:
                    shutil.copy2(src_left, dst_left)
                
                # Copy Matches (Renamed)
                # left_fname is like frame123_left.jpg
                # We want frame123_right.jpg (using content of match file)
                base_name = left_fname.replace("_left.jpg", "")
                
                for src_name, suffix, dst_name in cameras:
                    if src_name == "left_img_dir": continue
                    if src_name not in matches: continue 
                    
                    match_fname = matches[src_name]
                    src_file = episode_path / src_name / match_fname
                    
                    # New name: use LEFT timestamp
                    new_name = f"{base_name}{suffix}"
                    dst_file = dest_episode_dir / dst_name / new_name
                    
                    if src_file.exists():
                         if use_hardlink:
                            try:
                                if dst_file.exists(): dst_file.unlink()
                                os.link(src_file, dst_file)
                            except OSError:
                                shutil.copy2(src_file, dst_file)
                         else:
                            shutil.copy2(src_file, dst_file)
                
                copied_count += 1

        # 3. Filter Kinematics
        # sync_result['sync_df'] contains the filtered results for the left camera.
        # We need to filter it further to only include 'fully_valid_frames'
        
        kept_left_filenames = [x[0] for x in fully_valid_frames]

        write_filtered_kinematics(
            dest_episode_dir=dest_episode_dir,
            sync_df=sync_result['sync_df'],
            validation=validation,
            kept_filenames=kept_left_filenames,
        )   
        
        return True
        
    except Exception as e:
        return False


# Cleanup of temp_sync_analysis removed because we no longer save results to disk


def process_single_episode(
    episode_path: Path, 
    dest_dir: Path, 
    sync_script_path: str, # Kept for API compatibility, but unused
    max_time_diff: float, 
    min_images: int, 
    source_root: Path, 
    dry_run: bool,
    use_hardlink: bool = False,
    overwrite: bool = False
) -> Dict[str, Any]:
    """Processes a single episode: validate, sync, and copy if successful.

    Args:
        episode_path: Path to the episode directory.
        dest_dir: Destination directory.
        sync_script_path: Unused path to sync script (kept for signature).
        max_time_diff: Maximum time difference for sync.
        min_images: Minimum valid images to keep the episode.
        source_root: Source root path for relative path calculation.
        dry_run: If True, simulates the process.
        use_hardlink: If True, uses hardlinks.
        overwrite: If False, skip if destination exists.

    Returns:
        A dictionary containing processing statistics.
    """
    result_stats = {
        'processed': True,
        'valid_structure': False,
        'sync_successful': False,
        'copied': False,
        'images_copied': 0,
        'skipped_reason': None,
        'name': episode_path.name
    }

    # Identify destination directory
    rel_path = episode_path.relative_to(source_root)
    dest_episode_dir = dest_dir / rel_path

    # Check for existing destination
    if dest_episode_dir.exists() and not overwrite and not dry_run:
        result_stats['skipped_reason'] = "Destination already exists (skipping)"
        return result_stats

    # Validate episode structure
    validation = validate_episode_structure(episode_path)
    
    if not validation['has_left_images'] or not validation['has_kinematics']:
        missing = []
        if not validation['has_left_images']:
            missing.append("left images")
        if not validation['has_kinematics']:
            missing.append("kinematics CSV")
        result_stats['skipped_reason'] = f"Missing: {', '.join(missing)}"
        return result_stats
    
    result_stats['valid_structure'] = True
    
    # Run sync analysis directly
    sync_result = run_sync_analysis_direct(episode_path, max_time_diff)
    
    if not sync_result['success']:
        result_stats['skipped_reason'] = f"Sync failed: {sync_result.get('error', 'Unknown error')}"
        cleanup_temp_files(episode_path)
        return result_stats
    
    result_stats['sync_successful'] = True
    num_valid = sync_result['num_valid_images']
    
    if num_valid < min_images:
        result_stats['skipped_reason'] = f"Only {num_valid} valid images"
        cleanup_temp_files(episode_path)
        return result_stats
    
    # Copy episode if not dry run
    if not dry_run:
        success = copy_filtered_episode(
            episode_path, 
            dest_dir, 
            sync_result, 
            validation, 
            source_root, 
            use_hardlink,
            max_time_diff=max_time_diff
        )
        if success:
            result_stats['copied'] = True
            result_stats['images_copied'] = num_valid
        else:
             result_stats['skipped_reason'] = "Copy failed (likely sync mismatch across cameras)"
    else:
        result_stats['copied'] = True
        result_stats['images_copied'] = num_valid
    
    return result_stats


def main():
    parser = argparse.ArgumentParser(description="Filter and copy synchronized episodes")
    parser.add_argument("source_dir", help="Source directory containing episodes")
    parser.add_argument("dest_dir", help="Destination directory for filtered episodes")
    # sync_script is no longer needed but kept optional for CLI compatibility if users have old scripts
    parser.add_argument("--sync-script", default="src/sync_image_kinematics.py", 
                       help="Path to sync_image_kinematics.py script (unused in new direct mode)")
    parser.add_argument("--max-time-diff", type=float, default=30.0,
                       help="Maximum time difference threshold in ms (default: 30.0)")
    parser.add_argument("--min-images", type=int, default=10,
                       help="Minimum number of valid images required (default: 10)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be processed without actually copying")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of parallel workers (default: CPU count)")
    parser.add_argument("--hardlink", action="store_true",
                       help="Use hardlinks instead of copying (faster)")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing episodes in destination")
    
    args = parser.parse_args()
    
    source_dir = Path(args.source_dir)
    dest_dir = Path(args.dest_dir)
    
    if not source_dir.exists():
        print(f"Error: Source directory not found: {source_dir}")
        return 1
    
    if not args.dry_run:
        dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Find episodes
    print(f"Processing episodes from: {source_dir}")
    print(f"Destination directory: {dest_dir}")
    print(f"Max time difference: {args.max_time_diff} ms")
    print(f"Min images required: {args.min_images}")
    if args.workers:
        print(f"Workers: {args.workers}")
    print("-" * 60)
    
    episodes = find_episodes(str(source_dir))
    # --- NEW: early filtering to avoid spawning useless processes ---
    episodes_to_process: List[Path] = []
    skipped_pre = []


    for ep in episodes:
        try:
            rel = ep.relative_to(source_dir)
        except ValueError:
            rel = Path(ep.name)

        dest_episode_dir = dest_dir / rel

        if dest_episode_dir.exists() and not args.overwrite and not args.dry_run:
            skipped_pre.append((ep.name, "Destination already exists"))
            continue

        episodes_to_process.append(ep)


    # Update stats early
    stats = {
        'total_episodes': len(episodes),
        'valid_structure': 0,
        'sync_successful': 0,
        'copied_episodes': 0,
        'total_images_copied': 0,
        'skipped_episodes': skipped_pre.copy()
    }
    
    # Process episodes in parallel
    max_procs = min(args.workers or os.cpu_count(), 8)
    with ProcessPoolExecutor(max_workers=max_procs) as executor:
        futures = {
            executor.submit(
                process_single_episode, 
                episode, 
                dest_dir, 
                "",  # sync_script_path unused
                args.max_time_diff, 
                args.min_images, 
                source_dir, 
                args.dry_run,
                args.hardlink,
                args.overwrite
            ): episode for episode in episodes_to_process
        }
        
        # Create progress bar
        with tqdm(
            total=len(futures), 
            desc="Processing episodes", 
            unit="episode",
            ncols=100
        ) as pbar:
            for future in concurrent.futures.as_completed(futures):
                episode = futures[future]
                try:
                    result = future.result()
                    
                    if result['valid_structure']:
                        stats['valid_structure'] += 1
                    if result['sync_successful']:
                        stats['sync_successful'] += 1
                    if result['copied']:
                        stats['copied_episodes'] += 1
                        stats['total_images_copied'] += result['images_copied']
                    
                    if result['skipped_reason']:
                        stats['skipped_episodes'].append((result['name'], result['skipped_reason']))
                    
                    # Update progress bar with current stats
                    pbar.set_postfix({
                        'copied': stats['copied_episodes'],
                        'skipped': len(stats['skipped_episodes']),
                        'images': stats['total_images_copied']
                    })
                        
                except Exception as e:
                    stats['skipped_episodes'].append((episode.name, f"Exception: {str(e)}"))
                    pbar.set_postfix({
                        'copied': stats['copied_episodes'],
                        'skipped': len(stats['skipped_episodes']),
                        'error': episode.name[:20]
                    })
                finally:
                    pbar.update(1)

    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total episodes found: {stats['total_episodes']}")
    print(f"Episodes with valid structure: {stats['valid_structure']}")
    print(f"Episodes with successful sync: {stats['sync_successful']}")
    print(f"Episodes copied: {stats['copied_episodes']}")
    print(f"Total images copied: {stats['total_images_copied']}")
    print(f"Episodes skipped: {len(stats['skipped_episodes'])}")
    
    if stats['skipped_episodes']:
        print("\nSkipped episodes:")
        for episode_name, reason in stats['skipped_episodes']:
            print(f"  - {episode_name}: {reason}")
    
    if args.dry_run:
        print("\n[DRY RUN] No files were actually copied.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())