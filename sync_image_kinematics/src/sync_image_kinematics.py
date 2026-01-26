#!/usr/bin/env python3
"""
Script to synchronize images with kinematics data using timestamps and plot time differences.

This script:
1. Extracts timestamps from image filenames (nanosecond precision)
2. Finds the nearest kinematics data point for each image
3. Plots the time differences between images and their matched kinematics data

This module can be imported to use `process_episode_sync` for programmatic synchronization,
or run as a script for CLI usage.
"""

import os
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any, Union
import argparse


def extract_timestamp_from_filename(filename: str) -> int:
    """Extracts nanosecond timestamp from image filename.

    Args:
        filename: Image filename like 'frame1756826516968031906_left.jpg'.

    Returns:
        The extracted timestamp in nanoseconds.

    Raises:
        ValueError: If the filename does not match the expected pattern.
    """
    # Pattern to match frame{timestamp}_{camera}.jpg
    pattern = r"frame(\d+)_\w+\.jpg"
    match = re.search(pattern, filename)

    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Could not extract timestamp from filename: {filename}")


def load_image_timestamps(
    image_dir: Union[str, Path], camera_suffix: str = "_left"
) -> List[Tuple[str, int]]:
    """Loads all image files from a directory and extracts their timestamps.

    Args:
        image_dir: Directory containing image files.
        camera_suffix: Camera suffix to filter images (e.g., "_left", "_right").

    Returns:
        A list of tuples, where each tuple contains (filename, timestamp).
        The list is sorted by timestamp.
    """
    image_dir_path = Path(image_dir)
    pattern = image_dir_path / f"frame*{camera_suffix}.jpg"
    # glob.glob returns strings, so we wrap in str() for safety if Path is passed
    image_files = glob.glob(str(pattern))

    image_timestamps = []
    for img_file in image_files:
        filename = os.path.basename(img_file)
        try:
            timestamp = extract_timestamp_from_filename(filename)
            image_timestamps.append((filename, timestamp))
        except ValueError as e:
            # Inline comment: Skip files that don't match the expected format
            print(f"Warning: {e}")
            continue

    # Sort by timestamp to ensure chronological order
    image_timestamps.sort(key=lambda x: x[1])
    return image_timestamps


def load_kinematics_data(csv_file: Union[str, Path]) -> pd.DataFrame:
    """Loads kinematics data from a CSV file.

    Args:
        csv_file: Path to the CSV file containing kinematics data.

    Returns:
        A pandas DataFrame containing the kinematics data.
        The DataFrame will have a 'timestamp_ns' column added if not present.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    csv_path = Path(csv_file)
    if not csv_path.exists():
        raise FileNotFoundError(f"Kinematics CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Check if there's a timestamp column
    timestamp_cols = [
        col for col in df.columns if "time" in col.lower() or "stamp" in col.lower()
    ]

    if timestamp_cols:
        # Use the first timestamp column found
        timestamp_col = timestamp_cols[0]
        df["timestamp_ns"] = df[timestamp_col]
    else:
        # Inline comment: Create synthetic timestamps if missing, assuming 30Hz
        start_time = 0
        freq_ns = int(1e9 / 30)  # 30 Hz in nanoseconds
        df["timestamp_ns"] = start_time + np.arange(len(df)) * freq_ns

    return df


def find_nearest_kinematics(
    image_timestamps: List[Tuple[str, int]], kinematics_df: pd.DataFrame
) -> List[Dict[str, Any]]:
    """Finds the nearest kinematics data point for each image timestamp.

    Uses O(N log M) search with np.searchsorted for performance.
    """
    sync_results = []
    kinematics_timestamps = kinematics_df["timestamp_ns"].values
    
    # Sort check (though kinematics should always be sorted)
    # We assume kinematics_timestamps is sorted.
    
    for filename, img_timestamp in image_timestamps:
        # Find index such that kinematics_timestamps[idx-1] <= img_timestamp < kinematics_timestamps[idx]
        idx = np.searchsorted(kinematics_timestamps, img_timestamp)
        
        # Check idx and idx-1 to find the absolute nearest
        best_idx = 0
        if idx == 0:
            best_idx = 0
        elif idx == len(kinematics_timestamps):
            best_idx = len(kinematics_timestamps) - 1
        else:
            diff_curr = abs(kinematics_timestamps[idx] - img_timestamp)
            diff_prev = abs(kinematics_timestamps[idx-1] - img_timestamp)
            if diff_curr < diff_prev:
                best_idx = idx
            else:
                best_idx = idx - 1
                
        nearest_timestamp = kinematics_timestamps[best_idx]
        time_diff_ns = img_timestamp - nearest_timestamp
        time_diff_ms = time_diff_ns / 1e6

        sync_info = {
            "image_filename": filename,
            "image_timestamp_ns": img_timestamp,
            "kinematics_idx": int(best_idx),
            "kinematics_timestamp_ns": int(nearest_timestamp),
            "time_diff_ns": int(time_diff_ns),
            "time_diff_ms": float(time_diff_ms),
        }
        sync_results.append(sync_info)

    return sync_results


def remove_outliers(
    sync_results: List[Dict[str, Any]], max_time_diff_ms: float = 30.0
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Removes outliers where the time difference exceeds a threshold.

    Args:
        sync_results: List of sync information dictionaries.
        max_time_diff_ms: Maximum allowed time difference in milliseconds.

    Returns:
        A tuple containing:
        - filtered_results: List of sync dicts within the threshold.
        - outliers_removed: List of sync dicts exceeding the threshold.
    """
    filtered_results = []
    outliers_removed = []

    for result in sync_results:
        abs_time_diff = abs(result["time_diff_ms"])
        if abs_time_diff <= max_time_diff_ms:
            filtered_results.append(result)
        else:
            outliers_removed.append(result)

    # if outliers_removed:
    #     outlier_diffs = [abs(r["time_diff_ms"]) for r in outliers_removed]
    #     if outlier_diffs:
    #          print(
    #             f"  Outlier time differences: {min(outlier_diffs):.2f} - {max(outlier_diffs):.2f} ms"
    #         )

    return filtered_results, outliers_removed


def get_valid_image_filenames(sync_results: List[Dict[str, Any]]) -> List[str]:
    """Extracts a list of valid image filenames from sync results.

    Args:
        sync_results: List of sync information dictionaries.

    Returns:
        List of image filenames that passed the sync check.
    """
    return [result["image_filename"] for result in sync_results]


def plot_time_differences(
    sync_results: List[Dict[str, Any]], output_dir: Optional[str] = None
) -> None:
    """Plots the time differences between images and kinematics data.

    Args:
        sync_results: List of sync information dictionaries.
        output_dir: Directory to save plots. If None, plot is shown but not saved.
    """
    if not sync_results:
        print("No sync results to plot")
        return

    time_diffs_ms = [result["time_diff_ms"] for result in sync_results]
    image_indices = list(range(len(sync_results)))

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Time differences over image sequence
    ax1.plot(image_indices, time_diffs_ms, "b-", alpha=0.7, linewidth=1)
    ax1.scatter(image_indices, time_diffs_ms, c="red", s=20, alpha=0.6)
    ax1.set_xlabel("Image Index")
    ax1.set_ylabel("Time Difference (ms)")
    ax1.set_title("Time Difference Between Images and Nearest Kinematics Data")
    ax1.grid(True, alpha=0.3)

    # Add statistics text
    mean_diff = np.mean(time_diffs_ms)
    std_diff = np.std(time_diffs_ms)
    max_diff = np.max(np.abs(time_diffs_ms))
    ax1.text(
        0.02,
        0.98,
        f"Mean: {mean_diff:.2f} ms\nStd: {std_diff:.2f} ms\nMax |diff|: {max_diff:.2f} ms",
        transform=ax1.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
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
        plot_path = os.path.join(output_dir, "sync_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {plot_path}")

    # Note: when running in batch mode without a display, plt.show() might block or fail if not handled.
    # For now, we leave it as is for CLI usage, but caution is advised in headless environments.
    # plt.show() 


def save_sync_results(sync_results: List[Dict[str, Any]], output_file: str) -> None:
    """Saves synchronization results to a CSV file.

    Args:
        sync_results: List of sync information dictionaries.
        output_file: Path to the output CSV file.
    """
    df = pd.DataFrame(sync_results)
    df.to_csv(output_file, index=False)


def process_episode_sync(
    episode_path: Union[str, Path],
    camera: str = "left",
    output_dir: Optional[str] = None,
    csv_filename: str = "ee_csv.csv",
    max_time_diff_ms: float = 30.0,
    plot: bool = False,
    save_results: bool = True
) -> Dict[str, Any]:
    """Processes a single episode to synchronize images and kinematics.
    
    This function encapsulates the core logic of the script, allowing it to be
    imported and used by other modules (e.g., filter_episodes.py) without
    incurring subprocess overhead.

    Args:
        episode_path: Path to the episode directory.
        camera: Camera view to analyze (default: "left").
        output_dir: Directory to save results. If None, uses a 'sync_analysis' subdir.
        csv_filename: Name of the kinematics CSV file.
        max_time_diff_ms: Maximum time difference threshold for outlier removal.
        plot: Whether to generate plots.

    Returns:
        A dictionary containing synchronization results:
        - success: Boolean indicating if the process completed successfully.
        - valid_filenames: List of valid image filenames.
        - sync_df: DataFrame of filtered sync results.
        - error: Error message if success is False.
    """
    episode_path = Path(episode_path)
    
    # Determine image directory based on camera
    if camera == "left":
        image_dir = episode_path / "left_img_dir"
        camera_suffix = "_left"
    elif camera == "right":
        image_dir = episode_path / "right_img_dir"
        camera_suffix = "_right"
    elif camera == "psm1":
        image_dir = episode_path / "endo_psm1"
        camera_suffix = "_psm1"
    elif camera == "psm2":
        image_dir = episode_path / "endo_psm2"
        camera_suffix = "_psm2"
    else:
        return {'success': False, 'error': f"Unknown camera: {camera}"}

    csv_file = episode_path / csv_filename

    try:
        # Load data
        image_timestamps = load_image_timestamps(image_dir, camera_suffix)
        if not image_timestamps:
            return {'success': False, 'error': "No images found"}
            
        kinematics_df = load_kinematics_data(csv_file)
        
        # Synchronize
        sync_results = find_nearest_kinematics(image_timestamps, kinematics_df)

        # Remove outliers
        filtered_sync_results, outliers = remove_outliers(sync_results, max_time_diff_ms)

        if save_results:
            # Determine output locations
            if output_dir:
                out_dir = Path(output_dir)
            else:
                out_dir = episode_path / "sync_analysis"
                
            out_dir.mkdir(parents=True, exist_ok=True)

            # Save artifacts
            save_sync_results(sync_results, str(out_dir / "sync_results_original.csv"))
            save_sync_results(filtered_sync_results, str(out_dir / "sync_results_filtered.csv"))
            
            if outliers:
                save_sync_results(outliers, str(out_dir / "sync_results_outliers.csv"))

            valid_filenames = get_valid_image_filenames(filtered_sync_results)
            filenames_path = out_dir / "valid_image_filenames.txt"
            with open(filenames_path, "w") as f:
                for filename in valid_filenames:
                    f.write(f"{filename}\n")

            if plot:
                # Only plot if requested (saves time in batch mode)
                plot_time_differences(filtered_sync_results, str(out_dir))
        else:
            out_dir = None
            valid_filenames = get_valid_image_filenames(filtered_sync_results)

        return {
            'success': True,
            'valid_filenames': valid_filenames,
            'sync_df': pd.DataFrame(filtered_sync_results),
            'sync_output_dir': out_dir,
            'num_valid_images': len(valid_filenames),
            'outliers_removed': len(outliers)
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Synchronize images with kinematics data using timestamps"
    )
    parser.add_argument(
        "episode_path", help="Path to episode directory containing images and CSV"
    )
    parser.add_argument(
        "--camera", default="left", help="Camera to analyze (default: left)"
    )
    parser.add_argument("--output-dir", help="Directory to save results and plots")
    parser.add_argument(
        "--csv-file",
        default="ee_csv.csv",
        help="Name of kinematics CSV file (default: ee_csv.csv)",
    )
    parser.add_argument(
        "--max-time-diff",
        type=float,
        default=30.0,
        help="Maximum allowed time difference in ms for outlier removal (default: 30.0)",
    )

    args = parser.parse_args()
    
    print(f"Processing episode: {args.episode_path}")
    
    # Delegate to the function
    result = process_episode_sync(
        episode_path=args.episode_path,
        camera=args.camera,
        output_dir=args.output_dir,
        csv_filename=args.csv_file,
        max_time_diff_ms=args.max_time_diff,
        plot=True # Always plot in CLI mode
    )
    
    if result['success']:
        print("Sync completed successfully.")
        print(f"Valid images: {result['num_valid_images']}")
        print(f"Outliers removed: {result['outliers_removed']}")
    else:
        print(f"Sync failed: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    import sys
    main()
