#!/usr/bin/env python
"""
A script to convert DVRK (da Vinci Research Kit) robotics data into the LeRobot format (v2.1).

This script processes DVRK surgical robot datasets organized in directory structures
with CSV kinematics data and multiple camera views. It handles both perfect and
recovery demonstrations, extracting dual-arm PSM states, actions, and multi-camera
observations into a LeRobotDataset for the Hugging Face Hub.

Expected DVRK Dataset Structure:
--------------------------------
The script expects a directory structure organized by tissue and subtasks:

/path/to/dataset/
├── tissue_10/                          # Tissue phantom number
│   ├── 1_suture_throw/                 # Subtask directory
│   │   ├── episode_001/                # Individual episode
│   │   │   ├── left_img_dir/           # Left endoscope images
│   │   │   │   └── frame000000_left.jpg
│   │   │   ├── right_img_dir/          # Right endoscope images  
│   │   │   │   └── frame000000_right.jpg
│   │   │   ├── endo_psm1/              # PSM1 wrist camera
│   │   │   │   └── frame000000_psm1.jpg
│   │   │   ├── endo_psm2/              # PSM2 wrist camera
│   │   │   │   └── frame000000_psm2.jpg
│   │   │   └── ee_csv.csv              # Kinematics data (16D state + actions)
│   │   └── episode_002/
│   └── 2_needle_pass_recovery/         # Recovery demonstrations
└── tissue_11/

Data Format:
------------
- **Actions**: 16D dual-PSM Cartesian poses + jaw positions (absolute coordinates + quaternions)
- **States**: 16D dual-PSM current poses + jaw positions
- **Images**: 4 camera views (endoscope left/right, PSM1/2 wrist cameras)
- **Metadata**: Tool types, instruction text, recovery/perfect labels

Usage:
------
    python dvrk_zarr_to_lerobot.py --data-path /path/to/dataset --repo-id username/dataset-name

To also push to the Hugging Face Hub:
    python dvrk_zarr_to_lerobot.py --data-path /path/to/dataset --repo-id username/dataset-name --push-to-hub

Dependencies:
-------------
- lerobot v0.3.3
- tyro
- pandas
- PIL
- numpy
"""

import shutil
from pathlib import Path

import tyro
import numpy as np
import os
import pandas as pd
from PIL import Image
import time
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.constants import HF_LEROBOT_HOME

from lerobot.datasets.utils import write_info

states_name = [
    "psm1_pose.position.x",
    "psm1_pose.position.y",
    "psm1_pose.position.z",
    "psm1_pose.orientation.x",
    "psm1_pose.orientation.y",
    "psm1_pose.orientation.z",
    "psm1_pose.orientation.w",
    "psm1_jaw",
    "psm2_pose.position.x",
    "psm2_pose.position.y",
    "psm2_pose.position.z",
    "psm2_pose.orientation.x",
    "psm2_pose.orientation.y",
    "psm2_pose.orientation.z",
    "psm2_pose.orientation.w",
    "psm2_jaw",
]
actions_name = [
    "psm1_sp.position.x",
    "psm1_sp.position.y",
    "psm1_sp.position.z",
    "psm1_sp.orientation.x",
    "psm1_sp.orientation.y",
    "psm1_sp.orientation.z",
    "psm1_sp.orientation.w",
    "psm1_jaw_sp",
    "psm2_sp.position.x",
    "psm2_sp.position.y",
    "psm2_sp.position.z",
    "psm2_sp.orientation.x",
    "psm2_sp.orientation.y",
    "psm2_sp.orientation.z",
    "psm2_sp.orientation.w",
    "psm2_jaw_sp",
]


def read_images(image_dir: str, file_pattern: str, target_shape: tuple | None = None) -> np.ndarray:
    """Reads images from a directory into a NumPy array.

    Loads all images matching the pattern, optionally resizes them, and stacks
    them into a single 4D NumPy array (N, H, W, C).

    Args:
        image_dir (str): Directory containing the image files.
        file_pattern (str): F-string pattern for filenames (e.g., "frame{:06d}.jpg").
        target_shape (tuple, optional): Target (height, width) for resizing. 
            Defaults to None (no resizing).

    Returns:
        np.ndarray: Array of shape (N, H, W, C) containing the images.
            Returns an empty (0,0,0,3) array if directory doesn't exist or is empty.
    """
    images = []
    # If directory doesn't exist, return empty
    if not os.path.exists(image_dir) or not os.path.isdir(image_dir):
        return np.empty((0, 0, 0, 3), dtype=np.uint8)

    ## count images in the dir
    num_images = len(
        [
            name
            for name in os.listdir(image_dir)
            if os.path.isfile(os.path.join(image_dir, name))
        ]
    )
    for idx in range(num_images):
        filename = os.path.join(image_dir, file_pattern.format(int(idx)))
        if not os.path.exists(filename):
            print(f"Warning: {filename} does not exist.")
            continue
        img = Image.open(filename).convert("RGB")
        if target_shape is not None:
            height, width = target_shape
            img = img.resize((width, height), resample=Image.BILINEAR)
        img_array = np.array(img)[..., :3]  # Ensure 3 channels
        images.append(img_array)
    if images:
        return np.stack(images)
    else:
        return np.empty((0, 0, 0, 3), dtype=np.uint8)


def process_episode(
    dataset: LeRobotDataset,
    episode_path: str,
    states_name: list[str],
    actions_name: list[str],
    subtask_prompt: str,
) -> LeRobotDataset:
    """Processes a single episode and adds it to the LeRobot dataset.

    Reads images, kinematics CSV, and organizes them into a frame-by-frame
    dictionary to be added to the dataset.

    Args:
        dataset (LeRobotDataset): The target dataset object to add frames to.
        episode_path (str): Path to the directory containing episode data.
        states_name (list[str]): List of column names for state data in the CSV.
        actions_name (list[str]): List of column names for action data in the CSV.
        subtask_prompt (str): Natural language description of the task.

    Returns:
        LeRobotDataset: The updated dataset object.

    Implementation Details:
        - Reads images from 4 camera views: left/right endoscope, PSM1/PSM2 wrist.
        - Resizes images to standard resolutions (540x960 for endoscope, 480x640 for wrist).
        - Aligns CSV timestamps and data with image frames.
        - Handles potential mismatch in frame counts by using the minimum length.
    """

    # Paths to image directories
    left_dir = os.path.join(episode_path, "left_img_dir")
    right_dir = os.path.join(episode_path, "right_img_dir")
    psm1_dir = os.path.join(episode_path, "endo_psm1")
    psm2_dir = os.path.join(episode_path, "endo_psm2")
    csv_file = os.path.join(episode_path, "ee_csv.csv")

    # Read CSV to determine the number of frames (excluding header)
    df = pd.read_csv(csv_file)

    # Read images from each camera and resize to expected shapes if present
    left_images = read_images(left_dir, "frame{:06d}_left.jpg", target_shape=(540, 960))
    right_images = read_images(right_dir, "frame{:06d}_right.jpg", target_shape=(540, 960))
    # wrist cameras expected shape is (480, 640, 3)
    psm1_images = read_images(psm1_dir, "frame{:06d}_psm1.jpg", target_shape=(480, 640))
    psm2_images = read_images(psm2_dir, "frame{:06d}_psm2.jpg", target_shape=(480, 640))
    # print(left_images.shape, right_images.shape, psm1_images.shape, psm2_images.shape)
    num_frames = min(len(df), left_images.shape[0])

    # Read kinematics data and convert to structured array with headers
    kinematics_data = np.array(
        [tuple(row) for row in df.to_numpy()],
        dtype=[(col, df[col].dtype.str) for col in df.columns],
    )
    # print(kinematics_data[0])

    # Create frame dictionary for each timestamp
    for i in range(num_frames):
        frame = {
            "observation.state": np.hstack(
                [kinematics_data[n][i] for n in states_name]
            ).astype(np.float32),
            "action": np.hstack([kinematics_data[n][i] for n in actions_name]).astype(
                np.float32
            ),
            "instruction.text": subtask_prompt,
            "observation.meta.tool.psm1": "Large Needle Driver",
            "observation.meta.tool.psm2": "Debakey Forceps",
        }

        for cam_name, images in [
            ("endoscope.left", left_images),
            ("endoscope.right", right_images),
            ("wrist.left", psm2_images),
            ("wrist.right", psm1_images),
        ]:
            if images.size > 0:
                frame[f"observation.images.{cam_name}"] = images[i]
        timestamp_sec = kinematics_data["timestamp"][i] * 1e-9  ## turn nano sec to sec
        dataset.add_frame(frame, task=subtask_prompt, timestamp=timestamp_sec)

    return dataset

def convert_data_to_lerobot(
    data_path: Path, repo_id: str, *, push_to_hub: bool = False
):

    """Converts a Zarr-based DVRK dataset to the LeRobotDataset v2.1 format.

    Scans the directory structure for tissues and subtasks, processes each episode
    to extract synchronized video and kinematics, and saves the resulting dataset
    in the standard LeRobot format.

    Args:
        data_path (Path): Root directory of the DVRK dataset (containing 'tissue_X' folders).
        repo_id (str): Hugging Face repository ID for the output dataset.
        push_to_hub (bool, optional): If True, uploads the dataset to Hugging Face Hub 
            after conversion. Defaults to False.

    Implementation Details:
         - Uses `LeRobotDataset.create` to initialize the dataset schema.
         - Iterates through `tissue_X/subtask_Y/episode_Z` structure.
         - Handles 'recovery' vs 'perfect' demonstration labeling.
         - Saves train/val/test splits in metadata.
         - CRITICAL: Ensures `dataset.stop_image_writer()` is called to flush data.
    """
    final_output_path = os.path.join(HF_LEROBOT_HOME, repo_id)
    print(final_output_path)
    if os.path.exists(final_output_path):
        print(f"Removing existing dataset at {final_output_path}")
        shutil.rmtree(final_output_path)

    # Initialize a LeRobotDataset with the desired features.
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        use_videos=True,
        robot_type="dvrk",
        fps=30,
        features={
            "observation.images.endoscope.left": {
                "dtype": "video",
                "shape": (540, 960, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.endoscope.right": {
                "dtype": "video",
                "shape": (540, 960, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.wrist.left": {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.wrist.right": {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(states_name),),
                "names": [states_name],
            },
            "action": {
                "dtype": "float32",
                "shape": (len(actions_name),),
                "names": [actions_name],
            },
            "observation.meta.tool.psm1": {
                "dtype": "string",
                "shape": (1,),
                "names": ["value"],
            },
            "observation.meta.tool.psm2": {
                "dtype": "string",
                "shape": (1,),
                "names": ["value"],
            },
            "instruction.text": {
                "dtype": "string",
                "shape": (1,),
                "description": "Natural language command for the robot",
            },
        },
        image_writer_processes=16,
        image_writer_threads=20,
        tolerance_s=0.3,
        batch_encoding_size=150,
    )
    
    # Wrap processing in try/finally to ensure the image writer is stopped properly.
    # This prevents missing videos/images if the script exits early or finishes
    # while background threads are still flushing data.
    try:
        start_time = time.time()
        perfect_demo_count = 0
        recovery_demo_count = 0

        # Scan for all tissue directories
        tissue_dirs = sorted(
            [d for d in data_path.iterdir() if d.is_dir() and d.name.startswith("tissue_")],
            key=lambda x: int(x.name.split("_")[-1]) if x.name.split("_")[-1].isdigit() else float('inf')
        )
        
        if not tissue_dirs:
            print(f"Warning: No 'tissue_X' directories found in {data_path}.")
            return

        print(f"Found {len(tissue_dirs)} tissue directories: {[d.name for d in tissue_dirs]}")

        # Pre-scan to count total episodes for progress bar
        total_episodes_to_process = 0
        episode_paths = []
        
        print("Scanning episodes...")
        for tissue_dir in tissue_dirs:
            for subtask_name in os.listdir(tissue_dir):
                subtask_dir = os.path.join(tissue_dir, subtask_name)
                if not os.path.isdir(subtask_dir):
                    continue
                for episode_name in os.listdir(subtask_dir):
                    episode_dir = os.path.join(subtask_dir, episode_name)
                    if not os.path.isdir(episode_dir):
                        continue
                    episode_paths.append((tissue_dir, subtask_name, episode_dir))
        
        total_episodes_to_process = len(episode_paths)
        print(f"Total episodes found: {total_episodes_to_process}")

        # specific tqdm formatting
        pbar = tqdm(total=total_episodes_to_process, desc="Processing Episodes", unit="ep")

        for tissue_dir in tissue_dirs:
            # print(f"Processing {tissue_dir.name}...") 
            for subtask_name in os.listdir(tissue_dir):
                try:
                    subtask_dir = os.path.join(tissue_dir, subtask_name)
                    if not os.path.isdir(subtask_dir):
                        continue

                    subtask_prompt = " ".join(subtask_name.split("_")[1:])
                    is_recovery = subtask_prompt.endswith("recovery")
                    
                    if is_recovery:
                        subtask_prompt = subtask_prompt[:-9]  # Remove " recovery" suffix
                        
                    for episode_name in os.listdir(subtask_dir):
                        episode_dir = os.path.join(subtask_dir, episode_name)
                        if not os.path.isdir(episode_dir):
                            continue
                        
                        dataset = process_episode(
                            dataset, episode_dir, states_name, actions_name, subtask_prompt
                        )

                        dataset.save_episode()

                        if is_recovery:
                            recovery_demo_count += 1
                        else:
                            perfect_demo_count += 1
                        
                        pbar.update(1)

                except Exception as e:
                    print(f"Error processing subtask {subtask_dir}: {e}")
                    dataset.clear_episode_buffer() # Clear potentially partial episode
                
                # print(
                #     f"subtask {subtask_name} processed successful, time taken: {time.time() - start_time}"
                # )
        
        pbar.close()

        print(f"perfect_demo_count: {perfect_demo_count}")

        print(f"recovery_demo_count: {recovery_demo_count}")
        total_episode_count = perfect_demo_count + recovery_demo_count
        print(f"Total episodes processed: {total_episode_count}")
        train_count = int(0.8 * total_episode_count)
        val_count = int(0.1 * total_episode_count)
        # test_count = total_episode_count - train_count - val_count
        ## write split in meta
        dataset.meta.info["splits"] = {
            "train": "0:{}".format(train_count),
            "val": "{}:{}".format(train_count, train_count + val_count),
            "test": "{}:{}".format(train_count + val_count, total_episode_count),
            "perfect": f"0:{perfect_demo_count}",  # perfect episodes
            "recovery": f"{perfect_demo_count}:{perfect_demo_count + recovery_demo_count}",  # recovery episodes
            # "failure": "140:150",   # failure episodes
        }
        write_info(dataset.meta.info, dataset.root)

        print("Custom split configuration saved!")
        print(f"suturing processed successful, time taken: {time.time() - start_time}")

    finally:
        # Critical: Ensure all background video writing threads finish before exiting.
        # Without this, the last few videos might not be saved to disk.
        print("Stopping image writer and ensuring all data is persisted...")
        dataset.stop_image_writer()
        print("Image writer stopped.")


def main(
    data_path: Path = Path("/path/to/dataset"),
    repo_id: str = "jchen396/openh_test",
    *,
    push_to_hub: bool = False,
):
    """Entry point for the dataset conversion script.

    Validates input arguments and initiates the conversion process.

    Args:
        data_path (Path): Path to the root of the dataset (default: "/path/to/dataset").
        repo_id (str): Target Hugging Face repository ID (default: "jchen396/openh_test").
        push_to_hub (bool, optional): Whether to upload to Hub. Defaults to False.
    """
    if not data_path.exists():
        print(f"Error: The provided path does not exist: {data_path}")
        print("Please provide a valid path to your data.")
        return

    if repo_id == "your-username/your-dataset-name":
        print(
            "Warning: Using the default repo_id. Please specify your own with --repo-id."
        )

    convert_data_to_lerobot(data_path, repo_id, push_to_hub=push_to_hub)


if __name__ == "__main__":
    tyro.cli(main)
