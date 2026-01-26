#!/usr/bin/env python3
"""
merge_actions_to_videos.py

This script processes data collected with the dVRK during cautery tasks.
Each "cautery_tissue" folder contains multiple action runs. Each run has
four modalities of image frames (endo_psm1, endo_psm2, left_img_dir,
right_img_dir) and a corresponding ee_csv.csv file representing kinematic
data for that run.

For each cautery_tissue folder, the script:
1. Concatenates frames from each run into continuous video streams, one
   video per modality.
2. Merges all ee_csv.csv files vertically (raw append, no additional columns).
3. Saves all outputs into a newly created "videos" subfolder.
4. Applies overwrite behavior: skip generating a modality video if it exists
   unless overwrite=True.
5. Resizes frames within each modality to match the first frame of the first
   run to ensure consistent dimensions throughout concatenation.
6. Assumes that frames within a modality are naturally ordered by filename.

Outputs per tissue:
    videos/
        endo_psm1.mp4
        endo_psm2.mp4
        left_img_dir.mp4
        right_img_dir.mp4
        ee_csv.csv

"""

import os
import cv2
import csv
from typing import List, Dict, Optional
import argparse


def list_tissue_dirs(root_dir: str) -> List[str]:
    """List candidate cautery tissue directories.

    Args:
        root_dir (str): Path to the root directory containing tissue folders.

    Returns:
        List[str]: Sorted list of tissue directory paths.
    """
    dirs: List[str] = []
    for entry in os.listdir(root_dir):
        path = os.path.join(root_dir, entry)
        if os.path.isdir(path) and entry.startswith("cautery_tissue"):
            dirs.append(path)
    return sorted(dirs)


def list_run_dirs(tissue_dir: str) -> List[str]:
    """List run directories (timestamped) within a tissue folder.

    Args:
        tissue_dir (str): Path to a tissue folder.

    Returns:
        List[str]: Sorted list of run directory paths.
    """
    runs: List[str] = []
    for entry in os.listdir(tissue_dir):
        path = os.path.join(tissue_dir, entry)
        # Filtering by presence of modalities rather than checking timestamp pattern
        # because timestamp string format may vary across datasets.
        if os.path.isdir(path):
            if all(os.path.isdir(os.path.join(path, m)) for m in ["endo_psm1", "endo_psm2", "left_img_dir", "right_img_dir"]):
                runs.append(path)
    return sorted(runs)


def read_csv(filepath: str) -> List[List[str]]:
    """Read a CSV file into a list of rows (as list of fields).

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        List[List[str]]: Parsed CSV rows.

    Notes:
        - This function does not enforce header consistency across runs.
        - If header consistency is required for downstream modeling, a higher-
          level validation step should be added outside this function.
    """
    rows: List[List[str]] = []
    with open(filepath, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)
    return rows


def write_csv(filepath: str, rows: List[List[str]]) -> None:
    """Write rows to a CSV file.

    Args:
        filepath (str): Output file path.
        rows (List[List[str]]): Rows to write.
    """
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def collect_frames(run_dir: str, modality: str) -> List[str]:
    """Collect frame filenames for a modality within a run.

    Args:
        run_dir (str): Path to the run directory.
        modality (str): One of the four image modalities.

    Returns:
        List[str]: Sorted list of frame paths.
    """
    modality_dir = os.path.join(run_dir, modality)
    if not os.path.isdir(modality_dir):
        return []
    frames = sorted(
        [os.path.join(modality_dir, f) for f in os.listdir(modality_dir)
         if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    )
    return frames


def build_video_for_modality(
    run_dirs: List[str],
    modality: str,
    output_path: str,
    overwrite: bool = False,
    fps: int = 30
) -> None:
    """Concatenate frames from multiple runs into a single video for one modality.

    Args:
        run_dirs (List[str]): List of run directory paths.
        modality (str): The modality to process (e.g., 'endo_psm1').
        output_path (str): Final output path for the video file.
        overwrite (bool): If False, skip if file already exists.
        fps (int): Frames per second.

    Notes:
        - Resizes all frames to match the first frame encountered for this modality.
        - Skips runs without frames.
        - Pixel format is inferred from OpenCV reads (BGR).
        - Video codec is mp4v for broad compatibility.
    """
    if os.path.exists(output_path) and not overwrite:
        print(f"[INFO] Skipping '{output_path}' (already exists, overwrite=False)")
        return

    all_frame_paths: List[str] = []
    for run_dir in run_dirs:
        frames = collect_frames(run_dir, modality)
        # If a run has no frames for this modality, silently skip to avoid errors.
        all_frame_paths.extend(frames)

    if not all_frame_paths:
        print(f"[WARN] No frames found for modality {modality}. Not creating video.")
        return

    # Read first frame to establish consistent size.
    first_frame = cv2.imread(all_frame_paths[0])
    if first_frame is None:
        print(f"[ERROR] Failed to load first frame for modality {modality}.")
        return

    height, width, _ = first_frame.shape

    # Initialize video writer.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_path in all_frame_paths:
        frame = cv2.imread(frame_path)
        if frame is None:
            # Non-fatal: skip corrupted or non-readable frames rather than breaking the entire pipeline.
            print(f"[WARN] Could not read frame: {frame_path}. Skipping.")
            continue

        # Resize to ensure consistent output dimension.
        frame_resized = cv2.resize(frame, (width, height))
        writer.write(frame_resized)

    writer.release()
    print(f"[INFO] Created video: {output_path}")


def merge_csvs(run_dirs: List[str], output_csv_path: str) -> None:
    """Merge ee_csv.csv files from multiple runs by vertical append.

    Args:
        run_dirs (List[str]): List of run directory paths.
        output_csv_path (str): Path to write merged CSV.

    Notes:
        - Raw append behavior (Option B).
        - No new columns introduced.
        - First CSV's header preserved; subsequent headers stripped to avoid duplication.
        - If consistent headers are required across runs, validation must occur externally.
    """
    merged_rows: List[List[str]] = []
    header: Optional[List[str]] = None

    for idx, run_dir in enumerate(run_dirs):
        csv_path = os.path.join(run_dir, "ee_csv.csv")
        if not os.path.exists(csv_path):
            print(f"[WARN] Missing ee_csv.csv in {run_dir}. Skipping.")
            continue

        rows = read_csv(csv_path)
        if not rows:
            print(f"[WARN] Empty CSV in {run_dir}. Skipping.")
            continue

        # Preserve the first header only; strip subsequent headers to avoid duplication.
        if idx == 0:
            header = rows[0]
            merged_rows.append(header)
            merged_rows.extend(rows[1:])
        else:
            # Strip header row from subsequent runs.
            if header is not None and rows[0] == header:
                merged_rows.extend(rows[1:])
            else:
                # If headers differ, we still append but warn.
                print(f"[WARN] Header mismatch in {csv_path}. Appending raw data.")
                merged_rows.extend(rows)

    if merged_rows:
        write_csv(output_csv_path, merged_rows)
        print(f"[INFO] Created merged CSV: {output_csv_path}")
    else:
        print(f"[WARN] No CSV data merged. Output not created: {output_csv_path}")


def process_root(
    root_dir: str,
    overwrite: bool = False,
    fps: int = 30
) -> None:
    """Process all tissue folders within the root directory.

    Args:
        root_dir (str): Root directory containing cautery_tissue folders.
        overwrite (bool): Overwrite existing outputs.
        fps (int): Frames per second for output videos.

    Raises:
        FileNotFoundError: If root_dir does not exist.
    """
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Root directory does not exist: {root_dir}")

    tissues = list_tissue_dirs(root_dir)
    if not tissues:
        print("[WARN] No tissue directories found.")
        return

    modalities = ["endo_psm1", "endo_psm2", "left_img_dir", "right_img_dir"]

    for tissue_dir in tissues:
        print(f"[INFO] Processing tissue folder: {tissue_dir}")

        run_dirs = list_run_dirs(tissue_dir)
        if not run_dirs:
            print(f"[WARN] No runs found in {tissue_dir}. Skipping.")
            continue

        videos_dir = os.path.join(tissue_dir, "videos")
        os.makedirs(videos_dir, exist_ok=True)

        # Build videos per modality.
        for modality in modalities:
            output_path = os.path.join(videos_dir, f"{modality}.mp4")
            build_video_for_modality(
                run_dirs=run_dirs,
                modality=modality,
                output_path=output_path,
                overwrite=overwrite,
                fps=fps
            )

        # Merge CSVs.
        output_csv_path = os.path.join(videos_dir, "ee_csv.csv")
        if not os.path.exists(output_csv_path) or overwrite:
            merge_csvs(run_dirs, output_csv_path)
        else:
            print(f"[INFO] Skipping CSV merge (exists, overwrite=False): {output_csv_path}")


def main() -> None:
    """Entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(description="Merge dVRK action runs into videos + merged CSV.")
    parser.add_argument("root_dir", type=str, help="Path to root cautery directory.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for video output.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    args = parser.parse_args()

    process_root(root_dir=args.root_dir, overwrite=args.overwrite, fps=args.fps)

if __name__ == "__main__":
    main()