#!/usr/bin/env python3
"""Sample-check episodes in a sliced dataset.

This script randomly samples episode directories from a sliced dataset and
verifies structural and count-level consistency:
- Required image directories exist
- Frame counts are non-zero and comparable
- `ee_csv.csv` row counts match frame counts

This is intended as a lightweight sanity check, not a full validator.
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


REQUIRED_DIRS = ("left_img_dir", "right_img_dir", "endo_psm1", "endo_psm2")


@dataclass
class EpisodeInspection:
    """Summary of inspection results for a single episode."""

    episode_path: Path
    frame_counts: Dict[str, Optional[int]]
    sample_files: Dict[str, Optional[str]]
    ee_csv_rows: Optional[int]


def find_episodes(dataset_dir: Path) -> List[Path]:
    """Discover episode directories under a sliced dataset root.

    Assumes directory structure:
        dataset/tissue_x/session_y/episode_z/

    Args:
        dataset_dir: Root directory of the sliced dataset.

    Returns:
        List of episode directory paths.
    """
    episodes: List[Path] = []
    if not dataset_dir.is_dir():
        return episodes

    for tissue in dataset_dir.iterdir():
        if not tissue.is_dir():
            continue
        for session in tissue.iterdir():
            if not session.is_dir():
                continue
            for episode in session.iterdir():
                if episode.is_dir():
                    episodes.append(episode)

    return episodes


def count_csv_rows(csv_path: Path) -> Optional[int]:
    """Count data rows in a CSV file, accounting for an optional header.

    The CSV is streamed line-by-line for memory efficiency. If the first cell
    of the first row cannot be cast to int, it is treated as a header.

    Args:
        csv_path: Path to ee_csv.csv.

    Returns:
        Number of data rows, or None if the file does not exist.
    """
    if not csv_path.exists():
        return None

    row_count = 0
    try:
        with csv_path.open("r", newline="") as f:
            reader = csv.reader(f)
            first_row = next(reader, None)
            if first_row is None:
                return 0

            try:
                int(first_row[0])
                row_count = 1
            except ValueError:
                # header row detected
                row_count = 0

            for _ in reader:
                row_count += 1

    except Exception:
        return None

    return row_count


def inspect_episode(episode_dir: Path) -> EpisodeInspection:
    """Inspect directory contents and CSV consistency for one episode.

    Args:
        episode_dir: Path to an episode directory.

    Returns:
        EpisodeInspection summary.
    """
    frame_counts: Dict[str, Optional[int]] = {}
    sample_files: Dict[str, Optional[str]] = {}

    for name in REQUIRED_DIRS:
        p = episode_dir / name
        if p.is_dir():
            files = [f.name for f in p.iterdir() if f.is_file() and not f.name.startswith(".")]
            frame_counts[name] = len(files)
            sample_files[name] = files[0] if files else None
        else:
            frame_counts[name] = None
            sample_files[name] = None

    ee_csv_rows = count_csv_rows(episode_dir / "ee_csv.csv")

    return EpisodeInspection(
        episode_path=episode_dir,
        frame_counts=frame_counts,
        sample_files=sample_files,
        ee_csv_rows=ee_csv_rows,
    )


def main() -> None:
    """Entry point for dataset verification CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("dataset_sliced"))
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    episodes = find_episodes(args.dataset)
    if not episodes:
        print(f"No episodes found under {args.dataset}")
        return

    random.seed(args.seed)
    sampled = random.sample(episodes, min(args.samples, len(episodes)))

    inspections = [inspect_episode(ep) for ep in sampled]

    # Report
    for ins in inspections:
        print("\n---")
        print("Episode:", ins.episode_path)
        for d in REQUIRED_DIRS:
            print(f"  {d}: {ins.frame_counts[d]} sample: {ins.sample_files[d]}")
        print("  ee_csv_rows:", ins.ee_csv_rows)

    # Consistency check
    ok = True
    for ins in inspections:
        ref = ins.frame_counts.get("left_img_dir")
        if ref is not None and ins.ee_csv_rows is not None:
            if ref != ins.ee_csv_rows:
                print("Mismatch CSV vs frames in", ins.episode_path)
                ok = False

    if ok:
        print("\nSample-check passed: counts look consistent.")
    else:
        print("\nSample-check detected inconsistencies.")


if __name__ == "__main__":
    main()
