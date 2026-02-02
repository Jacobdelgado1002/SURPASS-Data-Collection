#!/usr/bin/env python3
"""
Example script demonstrating how to use the sync_image_kinematics module.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from sync_image_kinematics import (
    extract_timestamp_from_filename,
    load_image_timestamps,
    load_kinematics_data,
    find_nearest_kinematics,
    plot_time_differences,
    save_sync_results,
)


def example_usage():
    """
    Example of how to use the synchronization functions.
    """

    # Example 1: Extract timestamp from filename
    print("Example 1: Extract timestamp from filename")
    filename = "frame1756826516968031906_left.jpg"
    timestamp = extract_timestamp_from_filename(filename)
    print(f"Filename: {filename}")
    print(f"Extracted timestamp: {timestamp} ns")
    print(f"Timestamp in seconds: {timestamp / 1e9:.6f} s")
    print()

    # Example 2: Demonstrate with sample episode path
    print("Example 2: Process sample episode")

    # You can replace this with an actual episode path
    sample_episode_path = "/path/to/your/episode"

    if os.path.exists(sample_episode_path):
        try:
            # Load image timestamps
            image_dir = os.path.join(sample_episode_path, "left_img_dir")
            image_timestamps = load_image_timestamps(image_dir, "_left")
            print(f"Found {len(image_timestamps)} images")

            # Load kinematics data
            csv_file = os.path.join(sample_episode_path, "ee_csv.csv")
            kinematics_df = load_kinematics_data(csv_file)
            print(f"Loaded {len(kinematics_df)} kinematics data points")

            # Find nearest matches
            sync_results = find_nearest_kinematics(image_timestamps, kinematics_df)

            # Show first few results
            print("\nFirst 5 sync results:")
            for i, result in enumerate(sync_results[:5]):
                print(
                    f"  {i+1}. {result['image_filename']} -> "
                    f"kinematics[{result['kinematics_idx']}], "
                    f"diff: {result['time_diff_ms']:.2f} ms"
                )

            # Plot results
            plot_time_differences(sync_results)

        except Exception as e:
            print(f"Error processing episode: {e}")
    else:
        print(f"Sample episode path does not exist: {sample_episode_path}")
        print("Please update the path to point to a real episode directory.")


def create_sample_data():
    """
    Create sample data for testing purposes.
    """
    import pandas as pd
    import numpy as np

    print("Creating sample data for testing...")

    # Create sample directory structure
    sample_dir = "sample_episode"
    os.makedirs(os.path.join(sample_dir, "left_img_dir"), exist_ok=True)

    # Create sample image filenames (simulate timestamps)
    base_timestamp = 1756826516968031906
    num_images = 100
    image_freq_hz = 30  # 30 FPS
    image_interval_ns = int(1e9 / image_freq_hz)

    image_files = []
    for i in range(num_images):
        timestamp = base_timestamp + i * image_interval_ns
        filename = f"frame{timestamp}_left.jpg"
        image_files.append(filename)
        # Create empty files for demonstration
        filepath = os.path.join(sample_dir, "left_img_dir", filename)
        with open(filepath, "w") as f:
            f.write("")  # Empty file

    # Create sample kinematics CSV
    kinematics_freq_hz = 100  # 100 Hz kinematics
    kinematics_interval_ns = int(1e9 / kinematics_freq_hz)
    num_kinematics = int(num_images * kinematics_freq_hz / image_freq_hz)

    kinematics_data = []
    for i in range(num_kinematics):
        timestamp = base_timestamp + i * kinematics_interval_ns
        # Add some random noise to simulate real timing variations
        timestamp += np.random.randint(-1000000, 1000000)  # ±1ms noise

        kinematics_data.append(
            {
                "timestamp_ns": timestamp,
                "psm1_pose.position.x": np.random.random(),
                "psm1_pose.position.y": np.random.random(),
                "psm1_pose.position.z": np.random.random(),
                "psm1_pose.orientation.x": np.random.random(),
                "psm1_pose.orientation.y": np.random.random(),
                "psm1_pose.orientation.z": np.random.random(),
                "psm1_pose.orientation.w": np.random.random(),
            }
        )

    df = pd.DataFrame(kinematics_data)
    csv_path = os.path.join(sample_dir, "ee_csv.csv")
    df.to_csv(csv_path, index=False)

    print(f"Created sample data in: {sample_dir}")
    print(f"  - {num_images} sample image files")
    print(f"  - {num_kinematics} kinematics data points")

    return sample_dir


if __name__ == "__main__":
    print("Sync Image-Kinematics Example")
    print("=" * 40)

    # Run basic example
    example_usage()

    # Optionally create and test with sample data
    print("\n" + "=" * 40)
    print("Creating sample data for testing...")

    try:
        sample_dir = create_sample_data()

        # Test with sample data
        print(f"\nTesting with sample data in {sample_dir}...")

        # Load and process sample data
        image_dir = os.path.join(sample_dir, "left_img_dir")
        image_timestamps = load_image_timestamps(image_dir, "_left")

        csv_file = os.path.join(sample_dir, "ee_csv.csv")
        kinematics_df = load_kinematics_data(csv_file)

        sync_results = find_nearest_kinematics(image_timestamps, kinematics_df)

        print(f"Successfully synchronized {len(sync_results)} images with kinematics data")

        # Show statistics
        time_diffs_ms = [result["time_diff_ms"] for result in sync_results]
        print(f"Mean time difference: {np.mean(time_diffs_ms):.2f} ms")
        print(f"Std time difference: {np.std(time_diffs_ms):.2f} ms")

        # Plot results
        plot_time_differences(sync_results, sample_dir)

    except Exception as e:
        print(f"Error creating/testing sample data: {e}")
