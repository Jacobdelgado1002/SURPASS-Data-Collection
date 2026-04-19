#!/usr/bin/env python3
"""
reformat_data.py — Full pipeline for restructuring surgical robot datasets.

Supports two operating modes:

**Annotation Mode** (``--annotations-dir`` provided):
    Legacy workflow from ``dvrk_lerobot_converter_v2.1.py``
    (filter → plan/slice → trim → restructure).  Requires JSON annotations
    with ``affordance_range`` to slice long recording sessions.

**Direct Mode** (``--annotations-dir`` omitted):
    For the new folder structure where each action is collected as an
    individual recording (created by ``create_cautery_folder_structure.py``).
    Each action folder already IS a single episode — no annotation slicing
    is needed.  The pipeline becomes: filter → discover → trim → restructure.

    Expected input structure (``--input`` at dataset level)::

        Cholecystectomy/
          Tissue#1/
            Jacob/                              ← collector
              unzipping/                        ← phase
                1_grabbing_gallbladder_right/   ← action (= 1 episode)
                  left_img_dir/ right_img_dir/ endo_psm1/ endo_psm2/ ee_csv.csv
          Tissue#2/
            ...

Pipeline Stages (Annotation Mode):
    1. **Filter & Synchronise** — ``run_filter_episodes()``
    2. **Plan Affordance Slices** — ``plan_episodes()``
    3. **Trim Stationary Frames** (optional)
    4. **Restructure** — copy with normalised names/timestamps

Pipeline Stages (Direct Mode):
    1. **Filter & Synchronise** — ``run_filter_episodes()``
    2. **Discover Episodes** — recursively find action directories
    3. **Trim Stationary Frames** (optional)
    4. **Restructure** — copy with normalised names/timestamps

Data Integrity:
    - Original data is NEVER modified.  Output is always a fresh copy.
    - Hardlinks are optional (``--use-hardlink``) for speed when source and
      destination reside on the same filesystem.

Example CLI (Annotation Mode):
    python reformat_data.py \\
        --input  Cholecystectomy/tissues \\
        --annotations-dir Cholecystectomy/annotations \\
        --output restructured_output

Example CLI (Direct Mode — no annotations):
    python reformat_data.py \\
        --input  Cholecystectomy \\
        --output restructured_output

Programmatic usage:
    from reformat_data import run_pipeline, PipelineConfig

    # Annotation mode
    cfg = PipelineConfig(
        source_dir=Path("Cholecystectomy/tissues"),
        annotations_dir=Path("Cholecystectomy/annotations"),
        output_dir=Path("restructured_output"),
    )
    report = run_pipeline(cfg)

    # Direct mode (no annotations_dir)
    cfg = PipelineConfig(
        source_dir=Path("Cholecystectomy"),
        output_dir=Path("restructured_output"),
    )
    report = run_pipeline(cfg)
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Import project modules — add parent paths so un-installed scripts resolve.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_SYNC_SCRIPTS_DIR = str(_SCRIPT_DIR.parent / "sync_image_kinematics")
if _SYNC_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SYNC_SCRIPTS_DIR)

_POST_PROCESSING_DIR = str(_SCRIPT_DIR)
if _POST_PROCESSING_DIR not in sys.path:
    sys.path.insert(0, _POST_PROCESSING_DIR)

from filter_episodes import run_filter_episodes  # noqa: E402
from remove_stationary_frames import (  # noqa: E402
    MOTION_COLUMNS,
    compute_deltas,
    find_trim_range,
)
from slice_affordance import (  # noqa: E402
    extract_timestamp as sa_extract_timestamp,
    list_sorted_frames,
    plan_episodes,
)

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
try:
    from surpass_data_collection.logger_config import get_logger

    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        _handler = logging.StreamHandler()
        _handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(_handler)
        logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LEFT_IMG_DIR: str = "left_img_dir"
RIGHT_IMG_DIR: str = "right_img_dir"
ENDO_PSM1_DIR: str = "endo_psm1"
ENDO_PSM2_DIR: str = "endo_psm2"
CSV_FILE: str = "ee_csv.csv"

# Mapping from camera directory to output filename suffix
CAMERA_MODALITIES: Dict[str, str] = {
    LEFT_IMG_DIR: "_left",
    RIGHT_IMG_DIR: "_right",
    ENDO_PSM1_DIR: "_psm1",
    ENDO_PSM2_DIR: "_psm2",
}

# Camera suffix patterns for list_sorted_frames()
CAMERA_SUFFIXES: Dict[str, str] = {
    LEFT_IMG_DIR: "_left.jpg",
    RIGHT_IMG_DIR: "_right.jpg",
    ENDO_PSM1_DIR: "_psm1.jpg",
    ENDO_PSM2_DIR: "_psm2.jpg",
}

VALID_IMAGE_EXTENSIONS: frozenset = frozenset((".jpg", ".jpeg", ".png"))


# =============================================================================
# DATA STRUCTURES
# =============================================================================
@dataclass
class PipelineConfig:
    """All configuration for the restructuring pipeline."""

    source_dir: Path
    output_dir: Path
    annotations_dir: Optional[Path] = None
    dataset_name: str = "Cholecystectomy"
    fps: int = 30
    workers: int = 4
    trim_threshold: float = 1e-4
    no_trim: bool = False
    use_hardlink: bool = False
    dry_run: bool = False
    sort_by: str = "name"


@dataclass
class EpisodeResult:
    """Outcome of processing a single episode."""

    episode_id: str
    tissue: str
    action: str
    output_path: str
    source_session: str
    frame_range: Tuple[int, int] = (0, 0)
    num_frames: int = 0
    csv_rows: int = 0
    success: bool = False
    skipped: bool = False
    error: str = ""
    duration_s: float = 0.0


@dataclass
class PipelineReport:
    """Summary of the full pipeline run."""

    total_planned: int = 0
    total_processed: int = 0
    total_skipped: int = 0
    total_errors: int = 0
    total_frames_copied: int = 0
    total_csv_rows: int = 0
    trim_stats: Dict[str, int] = field(default_factory=dict)
    episode_results: List[EpisodeResult] = field(default_factory=list)
    elapsed_s: float = 0.0


@dataclass
class DirectEpisodeInfo:
    """Metadata for a single episode discovered in the direct structure.

    In the direct structure each action folder is a self-contained episode.
    Path hierarchy: ``Tissue#N / collector / phase / action``.
    """

    source_dir: Path
    tissue_label: str  # e.g. "Tissue#1"
    tissue_num: int  # e.g. 1
    collector: str  # e.g. "Jacob"
    phase: str  # e.g. "unzipping"
    action: str  # e.g. "1_grabbing_gallbladder_right"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def _copy_or_hardlink(src: Path, dst: Path, use_hardlink: bool) -> bool:
    """Copy or hardlink a single file.  Returns True on success."""
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if use_hardlink:
            try:
                if dst.exists():
                    dst.unlink()
                os.link(src, dst)
            except OSError:
                # Fallback to copy if hardlink fails (cross-device, etc.)
                shutil.copy2(src, dst)
        else:
            shutil.copy2(src, dst)
        return True
    except Exception as exc:
        logger.error("Failed to copy %s → %s: %s", src, dst, exc)
        return False


def _validate_episode_source(session_dir: Path) -> Tuple[bool, str]:
    """Validate that a filtered session directory has all required modalities."""
    errors: List[str] = []
    for cam_dir_name in CAMERA_MODALITIES:
        cam_dir = session_dir / cam_dir_name
        if not cam_dir.exists():
            errors.append(f"Missing directory: {cam_dir_name}")
        elif not any(cam_dir.glob("*.jpg")):
            errors.append(f"No images in: {cam_dir_name}")
    csv_path = session_dir / CSV_FILE
    if not csv_path.exists():
        errors.append(f"Missing CSV: {CSV_FILE}")
    if errors:
        return False, "; ".join(errors)
    return True, "Valid"


def _normalize_timestamps_for_csv(
    csv_path: Path, start_row: int, end_row: int, fps: int
) -> Tuple[Path, int]:
    """Slice CSV rows [start_row:end_row+1] and normalise timestamps.

    Writes the sliced, normalised CSV to a new file at *csv_path*.

    Returns:
        (csv_path, number_of_rows_written)
    """
    # Ensure parent exists
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    dt = 1.0 / fps
    rows_written = 0

    return csv_path, rows_written  # placeholder; actual logic below


def _build_episode_output_path(
    output_dir: Path,
    dataset_name: str,
    tissue_num: int,
    action_name: str,
    episode_index: int,
) -> Path:
    """Construct the canonical output path for an episode.

    output / dataset_name / tissue_N / action_subdir / episode_NNN
    """
    tissue_label = f"tissue_{tissue_num}"

    # Dynamically map action name to numbered subdirectory
    # The number is stable per-action alphabetically
    action_subdir = action_name.lower().strip()

    return output_dir / dataset_name / tissue_label / action_subdir / f"episode_{episode_index:03d}"


# =============================================================================
# STAGE 1: FILTER & SYNCHRONISE
# =============================================================================


def stage_filter(config: PipelineConfig) -> Path:
    """Run multi-camera synchronisation.

    Returns:
        Path to the filtered cache directory.
    """
    filtered_dir = config.output_dir / "_filtered_cache"
    max_time_diff_ms = float(config.fps)

    logger.info(
        "Stage 1: Filtering & synchronising episodes (threshold=%.1fms)...",
        max_time_diff_ms,
    )

    rc = run_filter_episodes(
        source_dir=str(config.source_dir),
        out_dir=str(filtered_dir),
        max_time_diff=max_time_diff_ms,
        min_images=10,
        dry_run=False,
        overwrite=False,
        use_hardlink=True,
    )
    if rc != 0:
        raise RuntimeError(f"Filtering stage failed with exit code {rc}")

    logger.info("  Stage 1 complete.")
    return filtered_dir


# =============================================================================
# STAGE 2: PLAN AFFORDANCE SLICES
# =============================================================================


def stage_plan(
    config: PipelineConfig, filtered_dir: Path
) -> List[Tuple[Path, Path, Path, Path, int, int]]:
    """Plan episode extraction from annotations.

    Returns:
        List of (annotation_path, ref_session_dir, src_session_dir,
                 dst_base_dir, start_frame, end_frame) tuples.
    """
    logger.info("Stage 2: Planning affordance slices...")

    planned = plan_episodes(
        source_data_path=config.source_dir,
        annotations_path=config.annotations_dir,
        out_dir=Path("_unused"),  # dst is rebuilt in stage 4
        source_dataset_dir=filtered_dir,
    )

    logger.info("  Planned %d episodes", len(planned))

    if not planned:
        raise RuntimeError(
            "No episodes planned. Check that annotations, source, and "
            "filtered data paths are correct."
        )

    return planned


# =============================================================================
# STAGE 2.5: TRIM STATIONARY FRAMES
# =============================================================================


def stage_trim(
    planned: List[Tuple[Path, Path, Path, Path, int, int]],
    threshold: float,
) -> Tuple[List[Tuple[Path, Path, Path, Path, int, int]], Dict[str, int]]:
    """Adjust planned episode indices to remove stationary head/tail.

    Ported directly from ``ConversionWorker._trim_stationary_episodes``.

    Returns:
        (updated_planned, stats_dict)
    """
    logger.info(
        "Stage 2.5: Trimming stationary frames (threshold=%.1e)...", threshold
    )

    # Group episodes by source session directory
    groups: Dict[Path, List[Tuple[int, int, int, Path]]] = defaultdict(list)
    for i, (ann, ref, src, dst, start, end) in enumerate(planned):
        groups[src].append((i, start, end, ref))

    trimmed_planned = list(planned)
    stats = {"total": len(planned), "trimmed": 0, "frames_removed": 0}

    for src_dir, episodes_in_session in groups.items():
        csv_path = src_dir / CSV_FILE
        if not csv_path.exists():
            continue

        try:
            full_deltas, _ = compute_deltas(csv_path)
            if full_deltas is None:
                continue
        except Exception as exc:
            logger.warning(
                "Could not compute deltas for %s: %s", csv_path, exc
            )
            continue

        # Load source timestamps for alignment
        src_left_dir = src_dir / LEFT_IMG_DIR
        src_files = list_sorted_frames(src_left_dir, CAMERA_SUFFIXES[LEFT_IMG_DIR])
        if not src_files:
            continue
        src_timestamps = np.array(
            [sa_extract_timestamp(f) for f in src_files], dtype=np.int64
        )

        for list_idx, start, end, ref_dir in episodes_in_session:
            # 1. Align ref bounds → src bounds
            ref_left_dir = ref_dir / LEFT_IMG_DIR
            ref_files = list_sorted_frames(
                ref_left_dir, CAMERA_SUFFIXES[LEFT_IMG_DIR]
            )
            if not ref_files:
                continue

            s_clamped = max(0, min(start, len(ref_files) - 1))
            e_clamped = max(0, min(end, len(ref_files) - 1))
            ts_start = sa_extract_timestamp(ref_files[s_clamped])
            ts_end = sa_extract_timestamp(ref_files[e_clamped])

            src_start = int(np.searchsorted(src_timestamps, ts_start, side="left"))
            src_end = int(np.searchsorted(src_timestamps, ts_end, side="right")) - 1
            src_start = max(0, min(src_start, len(src_files) - 1))
            src_end = max(0, min(src_end, len(src_files) - 1))

            if src_end <= src_start:
                continue

            # 2. Extract deltas and trim
            d_start = src_start
            d_end = min(src_end, len(full_deltas))
            if d_end <= d_start:
                continue

            ep_deltas = full_deltas[d_start:d_end]
            ep_n_rows = src_end - src_start + 1

            trim_start, trim_end = find_trim_range(
                ep_deltas, ep_n_rows, threshold
            )

            if trim_start == 0 and trim_end == ep_n_rows - 1:
                continue

            trimmed_src_start = src_start + trim_start
            trimmed_src_end = src_start + trim_end

            if trimmed_src_end - trimmed_src_start + 1 < 10:
                continue

            # 3. Map back to ref bounds
            ts_trimmed_start = src_timestamps[trimmed_src_start]
            ts_trimmed_end = src_timestamps[trimmed_src_end]

            ref_timestamps = np.array(
                [sa_extract_timestamp(f) for f in ref_files], dtype=np.int64
            )
            new_start = int(
                np.searchsorted(ref_timestamps, ts_trimmed_start, side="left")
            )
            new_end = (
                int(np.searchsorted(ref_timestamps, ts_trimmed_end, side="right"))
                - 1
            )
            new_start = max(0, min(new_start, len(ref_files) - 1))
            new_end = max(0, min(new_end, len(ref_files) - 1))

            if new_start != start or new_end != end:
                ann, ref, src, dst, _, _ = trimmed_planned[list_idx]
                trimmed_planned[list_idx] = (ann, ref, src, dst, new_start, new_end)
                removed = ep_n_rows - (trimmed_src_end - trimmed_src_start + 1)
                stats["trimmed"] += 1
                stats["frames_removed"] += removed

    logger.info(
        "  Trimmed %d/%d episodes, removed %d total frames",
        stats["trimmed"],
        stats["total"],
        stats["frames_removed"],
    )
    return trimmed_planned, stats


# =============================================================================
# STAGE 3: RESTRUCTURE
# =============================================================================


def _assign_action_numbers(
    planned: List[Tuple[Path, Path, Path, Path, int, int]],
) -> Dict[str, str]:
    """Extract unique action names from planned episodes and assign numbered
    subdirectory names (``1_grasp``, ``2_dissect``, etc.).

    Action names are read from the annotation JSON ``action`` field via the
    destination path structure produced by ``plan_episodes``, which encodes
    the action in ``dst.parent.name`` (e.g., ``1_grasp``).

    Returns:
        Mapping from raw action name (lowercase) → numbered subdirectory name.
    """
    raw_actions: set[str] = set()
    for ann_path, _, _, dst, _, _ in planned:
        # plan_episodes already encodes action as dst.parent.name (e.g. "1_grasp")
        subtask_dir_name = dst.parent.name
        # Strip leading number prefix if present: "1_grasp" → "grasp"
        parts = subtask_dir_name.split("_", 1)
        if len(parts) == 2 and parts[0].isdigit():
            action_name = parts[1]
        else:
            action_name = subtask_dir_name
        raw_actions.add(action_name.lower().strip())

    # Sort alphabetically and assign sequential numbers
    sorted_actions = sorted(raw_actions)
    return {
        action: f"{idx}_{action}" for idx, action in enumerate(sorted_actions, start=1)
    }


def _extract_action_from_dst(dst: Path) -> str:
    """Extract the raw action name from a destination path built by plan_episodes.

    ``dst.parent.name`` is e.g. ``"1_grasp"`` → returns ``"grasp"``.
    """
    subtask_dir_name = dst.parent.name
    parts = subtask_dir_name.split("_", 1)
    if len(parts) == 2 and parts[0].isdigit():
        return parts[1].lower().strip()
    return subtask_dir_name.lower().strip()


def _extract_tissue_num_from_dst(dst: Path) -> int:
    """Extract tissue number from the destination path built by plan_episodes.

    ``dst.parent.parent.name`` is e.g. ``"Jacob_tissue1"`` or ``"tissue_1"``.
    """
    tissue_label = dst.parent.parent.name
    # Match patterns like "Jacob_tissue1", "tissue_1", "tissue1"
    match = re.search(r"tissue_?(\d+)", tissue_label, re.IGNORECASE)
    if match:
        return int(match.group(1))
    logger.warning("Could not extract tissue number from %s, defaulting to 0", tissue_label)
    return 0


def _process_single_episode(
    config: PipelineConfig,
    ref_session_dir: Path,
    src_session_dir: Path,
    start_idx: int,
    end_idx: int,
    output_episode_dir: Path,
    action_name: str,
    episode_id: str,
) -> EpisodeResult:
    """Process a single planned episode: align, slice, copy, normalise.

    This mirrors ``_process_planned_episode`` from the converter but writes
    to the canonical directory structure instead of LeRobot.
    """
    result = EpisodeResult(
        episode_id=episode_id,
        tissue=output_episode_dir.parent.parent.name,
        action=action_name,
        output_path=str(output_episode_dir),
        source_session=str(src_session_dir),
    )
    t_start = time.time()

    try:
        # ------------------------------------------------------------------
        # 1. Timestamp alignment: reference indices → source indices
        # ------------------------------------------------------------------
        ref_left_dir = ref_session_dir / LEFT_IMG_DIR
        ref_files = list_sorted_frames(ref_left_dir, CAMERA_SUFFIXES[LEFT_IMG_DIR])
        if not ref_files:
            result.error = f"Reference directory empty: {ref_left_dir}"
            result.skipped = True
            return result

        s_clamped = max(0, min(start_idx, len(ref_files) - 1))
        e_clamped = max(0, min(end_idx, len(ref_files) - 1))
        ts_start = sa_extract_timestamp(ref_files[s_clamped])
        ts_end = sa_extract_timestamp(ref_files[e_clamped])

        src_left_dir = src_session_dir / LEFT_IMG_DIR
        src_files = list_sorted_frames(src_left_dir, CAMERA_SUFFIXES[LEFT_IMG_DIR])
        if not src_files:
            result.error = f"Source directory empty: {src_left_dir}"
            result.skipped = True
            return result

        src_timestamps = np.array(
            [sa_extract_timestamp(f) for f in src_files], dtype=np.int64
        )
        new_start = int(np.searchsorted(src_timestamps, ts_start, side="left"))
        new_end = int(np.searchsorted(src_timestamps, ts_end, side="right")) - 1
        new_start = max(0, min(new_start, len(src_files) - 1))
        new_end = max(0, min(new_end, len(src_files) - 1))

        total_frames = new_end - new_start + 1
        if total_frames <= 0:
            result.error = (
                f"Empty slice after alignment: ref {start_idx}-{end_idx} "
                f"→ src {new_start}-{new_end}"
            )
            result.skipped = True
            return result

        result.frame_range = (new_start, new_end)

        logger.info(
            "  Aligned ref[%d:%d] → src[%d:%d] (%d frames)",
            start_idx,
            end_idx,
            new_start,
            new_end,
            total_frames,
        )

        # ------------------------------------------------------------------
        # 2. Validate source episode
        # ------------------------------------------------------------------
        is_valid, err_msg = _validate_episode_source(src_session_dir)
        if not is_valid:
            result.error = f"Validation failed: {err_msg}"
            result.skipped = True
            return result

        # ------------------------------------------------------------------
        # 3. Build path lists for ALL cameras in the filtered session
        # ------------------------------------------------------------------
        camera_files: Dict[str, List[str]] = {}
        camera_files[LEFT_IMG_DIR] = src_files

        for cam_dir, cam_suffix in CAMERA_SUFFIXES.items():
            if cam_dir == LEFT_IMG_DIR:
                continue
            cam_files = list_sorted_frames(src_session_dir / cam_dir, cam_suffix)
            if not cam_files:
                result.error = f"Missing {cam_dir} frames in {src_session_dir}"
                result.skipped = True
                return result
            camera_files[cam_dir] = cam_files

        # ------------------------------------------------------------------
        # 4. Verify cross-modality alignment (frame counts should match
        #    in filtered data, but verify defensively)
        # ------------------------------------------------------------------
        min_cam_count = min(len(files) for files in camera_files.values())
        if new_end >= min_cam_count:
            logger.warning(
                "  Clamping end index from %d to %d (min camera frames: %d)",
                new_end,
                min_cam_count - 1,
                min_cam_count,
            )
            new_end = min_cam_count - 1
            total_frames = new_end - new_start + 1
            if total_frames <= 0:
                result.error = "No frames after clamping to min camera count"
                result.skipped = True
                return result

        # ------------------------------------------------------------------
        # 5. Copy frames with normalised names
        # ------------------------------------------------------------------
        if config.dry_run:
            result.num_frames = total_frames
            result.success = True
            result.duration_s = time.time() - t_start
            logger.info("  [DRY RUN] Would copy %d frames to %s", total_frames, output_episode_dir)
            return result

        frames_copied = 0
        for cam_dir, cam_suffix_name in CAMERA_MODALITIES.items():
            cam_out_dir = output_episode_dir / cam_dir
            cam_out_dir.mkdir(parents=True, exist_ok=True)
            cam_files_list = camera_files[cam_dir]

            for out_idx in range(total_frames):
                src_idx = new_start + out_idx
                if src_idx >= len(cam_files_list):
                    logger.warning(
                        "  Frame index %d out of range for %s (%d files)",
                        src_idx,
                        cam_dir,
                        len(cam_files_list),
                    )
                    continue

                src_file = src_session_dir / cam_dir / cam_files_list[src_idx]
                ext = src_file.suffix.lower()
                dst_file = cam_out_dir / f"frame{out_idx:06d}{cam_suffix_name}{ext}"

                if not src_file.exists():
                    logger.warning("  Source file missing: %s", src_file)
                    continue

                if _copy_or_hardlink(src_file, dst_file, config.use_hardlink):
                    frames_copied += 1

        result.num_frames = frames_copied // len(CAMERA_MODALITIES)

        # ------------------------------------------------------------------
        # 6. Slice CSV and normalise timestamps
        # ------------------------------------------------------------------
        csv_src = src_session_dir / CSV_FILE
        csv_dst = output_episode_dir / CSV_FILE

        try:
            df = pd.read_csv(csv_src)
        except Exception as exc:
            result.error = f"Failed to read CSV {csv_src}: {exc}"
            result.skipped = True
            # Clean up partially copied frames
            if output_episode_dir.exists():
                shutil.rmtree(output_episode_dir)
            return result

        # Slice rows to match the frame range
        if new_end + 1 > len(df):
            logger.warning(
                "  CSV has %d rows but need row %d; clamping",
                len(df),
                new_end,
            )
            new_end_csv = min(new_end, len(df) - 1)
        else:
            new_end_csv = new_end

        sliced_df = df.iloc[new_start : new_end_csv + 1].copy()
        sliced_df.reset_index(drop=True, inplace=True)

        # Normalise timestamps: replace first column with relative seconds
        dt = 1.0 / config.fps
        timestamp_col = sliced_df.columns[0]
        sliced_df[timestamp_col] = [
            f"{i * dt:.4f}" for i in range(len(sliced_df))
        ]

        csv_dst.parent.mkdir(parents=True, exist_ok=True)
        sliced_df.to_csv(csv_dst, index=False)
        result.csv_rows = len(sliced_df)

        # ------------------------------------------------------------------
        # 7. Final validation
        # ------------------------------------------------------------------
        expected_frames = result.num_frames
        if expected_frames != result.csv_rows:
            logger.warning(
                "  Frame/CSV mismatch: %d frames vs %d CSV rows in %s",
                expected_frames,
                result.csv_rows,
                output_episode_dir,
            )

        result.success = True
        result.duration_s = time.time() - t_start

    except Exception as exc:
        result.error = str(exc)
        result.duration_s = time.time() - t_start
        logger.exception("Error processing episode %s: %s", episode_id, exc)

    return result


def stage_restructure(
    config: PipelineConfig,
    planned: List[Tuple[Path, Path, Path, Path, int, int]],
) -> PipelineReport:
    """Core restructuring stage: copy and reformat data into canonical structure.

    Processes each planned episode sequentially (image I/O is the bottleneck,
    and parallelism at the file level risks disk thrashing).
    """
    logger.info("Stage 3: Restructuring %d episodes...", len(planned))
    report = PipelineReport(total_planned=len(planned))

    # Assign numbered action subdirectory names dynamically
    action_map = _assign_action_numbers(planned)
    logger.info("  Discovered actions: %s", action_map)

    # Build per-(tissue, action) episode counters for sequential naming
    episode_counters: Dict[Tuple[int, str], int] = defaultdict(int)

    start_time = time.time()

    for i, (ann, ref, src, dst, start_frame, end_frame) in enumerate(planned):
        # Extract metadata from the planned tuple
        action_name = _extract_action_from_dst(dst)
        tissue_num = _extract_tissue_num_from_dst(dst)
        action_subdir = action_map.get(action_name, f"0_{action_name}")

        # Increment episode counter for this (tissue, action)
        counter_key = (tissue_num, action_name)
        episode_counters[counter_key] += 1
        episode_index = episode_counters[counter_key]

        # Build output path
        output_episode_dir = (
            config.output_dir
            / config.dataset_name
            / f"tissue_{tissue_num}"
            / action_subdir
            / f"episode_{episode_index:03d}"
        )

        episode_id = f"tissue_{tissue_num}/{action_subdir}/episode_{episode_index:03d}"

        logger.info(
            "\n[%d/%d] Episode: %s  (frames %d–%d)",
            i + 1,
            len(planned),
            episode_id,
            start_frame,
            end_frame,
        )

        result = _process_single_episode(
            config=config,
            ref_session_dir=ref,
            src_session_dir=src,
            start_idx=start_frame,
            end_idx=end_frame,
            output_episode_dir=output_episode_dir,
            action_name=action_name,
            episode_id=episode_id,
        )

        report.episode_results.append(result)

        if result.success:
            report.total_processed += 1
            report.total_frames_copied += result.num_frames
            report.total_csv_rows += result.csv_rows
            logger.info(
                "[SUCCESS] %s — %d frames, %d CSV rows (%.1fs)",
                episode_id,
                result.num_frames,
                result.csv_rows,
                result.duration_s,
            )
        elif result.skipped:
            report.total_skipped += 1
            logger.warning("[SKIPPED] %s — %s", episode_id, result.error)
        else:
            report.total_errors += 1
            logger.error("[ERROR] %s — %s", episode_id, result.error)

    report.elapsed_s = time.time() - start_time
    return report


# =============================================================================
# DIRECT MODE: DISCOVER → TRIM → RESTRUCTURE
# =============================================================================


def _discover_direct_episodes(root_dir: Path) -> List[DirectEpisodeInfo]:
    """Walk *root_dir* and discover all direct-mode episodes.

    A valid episode is a directory containing ``ee_csv.csv`` and at least
    one camera subdirectory (``left_img_dir``).

    Path hierarchy expected::

        root_dir / Tissue#N / collector / phase / action /
            left_img_dir/ right_img_dir/ endo_psm1/ endo_psm2/ ee_csv.csv

    Returns:
        Sorted list of :class:`DirectEpisodeInfo`.
    """
    _TISSUE_RE = re.compile(r"Tissue#?(\d+)", re.IGNORECASE)

    episodes: List[DirectEpisodeInfo] = []

    for csv_file in root_dir.rglob(CSV_FILE):
        episode_dir = csv_file.parent

        # Must have at least the left camera directory
        if not (episode_dir / LEFT_IMG_DIR).is_dir():
            continue
        # Must have at least one image
        if not any((episode_dir / LEFT_IMG_DIR).glob("*.jpg")):
            continue

        # Parse path components relative to root
        try:
            rel = episode_dir.relative_to(root_dir)
        except ValueError:
            continue

        parts = rel.parts

        # Locate the Tissue#N component
        tissue_idx = -1
        tissue_label = ""
        tissue_num = 0
        for i, part in enumerate(parts):
            m = _TISSUE_RE.match(part)
            if m:
                tissue_idx = i
                tissue_label = part
                tissue_num = int(m.group(1))
                break

        if tissue_idx < 0:
            logger.debug(
                "Skipping %s — no Tissue#N component found in path", episode_dir
            )
            continue

        # Parts after Tissue#N: collector / phase / action
        remaining = parts[tissue_idx + 1 :]

        if len(remaining) >= 3:
            collector = remaining[0]
            phase = remaining[1]
            action = remaining[2]
        elif len(remaining) == 2:
            collector = ""
            phase = remaining[0]
            action = remaining[1]
        elif len(remaining) == 1:
            collector = ""
            phase = ""
            action = remaining[0]
        else:
            collector = ""
            phase = ""
            action = episode_dir.name

        episodes.append(
            DirectEpisodeInfo(
                source_dir=episode_dir,
                tissue_label=tissue_label,
                tissue_num=tissue_num,
                collector=collector,
                phase=phase,
                action=action,
            )
        )

    episodes.sort(key=lambda e: (e.tissue_num, e.collector, e.phase, e.action))
    logger.info("Discovered %d direct episodes under %s", len(episodes), root_dir)
    return episodes


def stage_trim_direct(
    episodes: List[DirectEpisodeInfo],
    threshold: float,
) -> Tuple[List[Tuple[DirectEpisodeInfo, int, int]], Dict[str, int]]:
    """Compute stationary-frame trim ranges for each direct episode.

    Returns:
        (episodes_with_trims, stats_dict) where each element of the list
        is ``(episode_info, trim_start, trim_end)``.
    """
    logger.info(
        "Trimming stationary frames (threshold=%.1e) on %d episodes...",
        threshold,
        len(episodes),
    )

    results: List[Tuple[DirectEpisodeInfo, int, int]] = []
    stats = {"total": len(episodes), "trimmed": 0, "frames_removed": 0}

    for ep in episodes:
        csv_path = ep.source_dir / CSV_FILE

        # Determine total frame count from left camera
        left_dir = ep.source_dir / LEFT_IMG_DIR
        frame_files = list_sorted_frames(left_dir, CAMERA_SUFFIXES[LEFT_IMG_DIR])
        n_frames = len(frame_files)

        if n_frames < 10:
            results.append((ep, 0, max(0, n_frames - 1)))
            continue

        try:
            deltas, _ = compute_deltas(csv_path)
            if deltas is None:
                results.append((ep, 0, n_frames - 1))
                continue
        except Exception:
            results.append((ep, 0, n_frames - 1))
            continue

        trim_start, trim_end = find_trim_range(deltas, n_frames, threshold)

        if trim_end - trim_start + 1 < 10:
            # Trimmed too short — keep original
            results.append((ep, 0, n_frames - 1))
            continue

        if trim_start > 0 or trim_end < n_frames - 1:
            removed = n_frames - (trim_end - trim_start + 1)
            stats["trimmed"] += 1
            stats["frames_removed"] += removed

        results.append((ep, trim_start, trim_end))

    logger.info(
        "  Trimmed %d/%d episodes, removed %d total frames",
        stats["trimmed"],
        stats["total"],
        stats["frames_removed"],
    )
    return results, stats


def _process_direct_episode(
    config: PipelineConfig,
    episode_info: DirectEpisodeInfo,
    trim_start: int,
    trim_end: int,
    output_episode_dir: Path,
    episode_id: str,
) -> EpisodeResult:
    """Process a single direct-mode episode: validate, copy, normalise.

    Unlike ``_process_single_episode`` (annotation mode), this does NOT
    perform annotation-based timestamp alignment.  The episode directory
    already contains exactly one self-contained recording.
    """
    result = EpisodeResult(
        episode_id=episode_id,
        tissue=episode_info.tissue_label,
        action=episode_info.action,
        output_path=str(output_episode_dir),
        source_session=str(episode_info.source_dir),
    )
    t_start = time.time()

    try:
        # ------------------------------------------------------------------
        # 1. Validate source episode
        # ------------------------------------------------------------------
        is_valid, err_msg = _validate_episode_source(episode_info.source_dir)
        if not is_valid:
            result.error = f"Validation failed: {err_msg}"
            result.skipped = True
            return result

        # ------------------------------------------------------------------
        # 2. Build frame lists for all cameras
        # ------------------------------------------------------------------
        camera_files: Dict[str, List[str]] = {}
        for cam_dir, cam_suffix in CAMERA_SUFFIXES.items():
            cam_files = list_sorted_frames(
                episode_info.source_dir / cam_dir, cam_suffix
            )
            if not cam_files:
                result.error = f"Missing {cam_dir} frames in {episode_info.source_dir}"
                result.skipped = True
                return result
            camera_files[cam_dir] = cam_files

        # ------------------------------------------------------------------
        # 3. Clamp trim range to actual camera frame counts
        # ------------------------------------------------------------------
        min_cam_count = min(len(files) for files in camera_files.values())
        actual_end = min(trim_end, min_cam_count - 1)
        total_frames = actual_end - trim_start + 1

        if total_frames <= 0:
            result.error = "No frames after trimming/clamping"
            result.skipped = True
            return result

        result.frame_range = (trim_start, actual_end)

        # ------------------------------------------------------------------
        # 4. Dry-run shortcut
        # ------------------------------------------------------------------
        if config.dry_run:
            result.num_frames = total_frames
            result.success = True
            result.duration_s = time.time() - t_start
            logger.info(
                "  [DRY RUN] Would copy %d frames to %s",
                total_frames,
                output_episode_dir,
            )
            return result

        # ------------------------------------------------------------------
        # 5. Copy frames with normalised names
        # ------------------------------------------------------------------
        frames_copied = 0
        for cam_dir, cam_suffix_name in CAMERA_MODALITIES.items():
            cam_out_dir = output_episode_dir / cam_dir
            cam_out_dir.mkdir(parents=True, exist_ok=True)
            cam_files_list = camera_files[cam_dir]

            for out_idx in range(total_frames):
                src_idx = trim_start + out_idx
                if src_idx >= len(cam_files_list):
                    logger.warning(
                        "  Frame index %d out of range for %s (%d files)",
                        src_idx,
                        cam_dir,
                        len(cam_files_list),
                    )
                    continue

                src_file = (
                    episode_info.source_dir / cam_dir / cam_files_list[src_idx]
                )
                ext = src_file.suffix.lower()
                dst_file = (
                    cam_out_dir / f"frame{out_idx:06d}{cam_suffix_name}{ext}"
                )

                if not src_file.exists():
                    logger.warning("  Source file missing: %s", src_file)
                    continue

                if _copy_or_hardlink(src_file, dst_file, config.use_hardlink):
                    frames_copied += 1

        result.num_frames = frames_copied // max(len(CAMERA_MODALITIES), 1)

        # ------------------------------------------------------------------
        # 6. Slice CSV and normalise timestamps
        # ------------------------------------------------------------------
        csv_src = episode_info.source_dir / CSV_FILE
        csv_dst = output_episode_dir / CSV_FILE

        try:
            df = pd.read_csv(csv_src)
        except Exception as exc:
            result.error = f"Failed to read CSV {csv_src}: {exc}"
            result.skipped = True
            if output_episode_dir.exists():
                shutil.rmtree(output_episode_dir)
            return result

        actual_csv_end = min(actual_end, len(df) - 1)
        sliced_df = df.iloc[trim_start : actual_csv_end + 1].copy()
        sliced_df.reset_index(drop=True, inplace=True)

        # Normalise timestamps to relative seconds
        dt = 1.0 / config.fps
        timestamp_col = sliced_df.columns[0]
        sliced_df[timestamp_col] = [
            f"{i * dt:.4f}" for i in range(len(sliced_df))
        ]

        csv_dst.parent.mkdir(parents=True, exist_ok=True)
        sliced_df.to_csv(csv_dst, index=False)
        result.csv_rows = len(sliced_df)

        # ------------------------------------------------------------------
        # 7. Final validation
        # ------------------------------------------------------------------
        if result.num_frames != result.csv_rows:
            logger.warning(
                "  Frame/CSV mismatch: %d frames vs %d CSV rows in %s",
                result.num_frames,
                result.csv_rows,
                output_episode_dir,
            )

        result.success = True
        result.duration_s = time.time() - t_start

    except Exception as exc:
        result.error = str(exc)
        result.duration_s = time.time() - t_start
        logger.exception("Error processing episode %s: %s", episode_id, exc)

    return result


def stage_restructure_direct(
    config: PipelineConfig,
    episodes_with_trims: List[Tuple[DirectEpisodeInfo, int, int]],
) -> PipelineReport:
    """Restructure direct-mode episodes into the canonical output format."""
    logger.info("Restructuring %d episodes...", len(episodes_with_trims))
    report = PipelineReport(total_planned=len(episodes_with_trims))

    # Per-(tissue, collector, phase, action) episode counter
    episode_counters: Dict[Tuple[int, str, str, str], int] = defaultdict(int)

    start_time = time.time()

    for i, (ep, trim_start, trim_end) in enumerate(episodes_with_trims):
        counter_key = (ep.tissue_num, ep.collector, ep.phase, ep.action)
        episode_counters[counter_key] += 1
        episode_index = episode_counters[counter_key]

        # Build output path: dataset / tissue_N / [collector/] [phase/] action / episode_NNN
        tissue_label = f"tissue_{ep.tissue_num}"
        path_parts: List[str] = [tissue_label]
        if ep.collector:
            path_parts.append(ep.collector)
        if ep.phase:
            path_parts.append(ep.phase)
        path_parts.append(ep.action)
        path_parts.append(f"episode_{episode_index:03d}")

        output_episode_dir = config.output_dir / config.dataset_name
        for part in path_parts:
            output_episode_dir = output_episode_dir / part

        episode_id = str(
            output_episode_dir.relative_to(
                config.output_dir / config.dataset_name
            )
        )

        logger.info(
            "\n[%d/%d] Episode: %s  (frames %d–%d)",
            i + 1,
            len(episodes_with_trims),
            episode_id,
            trim_start,
            trim_end,
        )

        result = _process_direct_episode(
            config=config,
            episode_info=ep,
            trim_start=trim_start,
            trim_end=trim_end,
            output_episode_dir=output_episode_dir,
            episode_id=episode_id,
        )

        report.episode_results.append(result)

        if result.success:
            report.total_processed += 1
            report.total_frames_copied += result.num_frames
            report.total_csv_rows += result.csv_rows
            logger.info(
                "[SUCCESS] %s — %d frames, %d CSV rows (%.1fs)",
                episode_id,
                result.num_frames,
                result.csv_rows,
                result.duration_s,
            )
        elif result.skipped:
            report.total_skipped += 1
            logger.warning("[SKIPPED] %s — %s", episode_id, result.error)
        else:
            report.total_errors += 1
            logger.error("[ERROR] %s — %s", episode_id, result.error)

    report.elapsed_s = time.time() - start_time
    return report


def run_pipeline_direct(config: PipelineConfig) -> PipelineReport:
    """Execute the direct-mode restructuring pipeline.

    Stages:
        1. Filter & synchronise
        2. Discover episodes (no annotation planning)
        3. Trim stationary frames (optional)
        4. Restructure
    """
    overall_start = time.time()

    if not config.source_dir.exists():
        raise ValueError(f"Source directory not found: {config.source_dir}")

    n_stages = 3 if config.no_trim else 4
    logger.info("=" * 60)
    logger.info(
        "Dataset Restructuring Pipeline — Direct Mode (%d stages)", n_stages
    )
    logger.info("  Source:      %s", config.source_dir)
    logger.info("  Output:      %s", config.output_dir)
    logger.info("  Dataset:     %s", config.dataset_name)
    logger.info("  FPS:         %d", config.fps)
    logger.info(
        "  Trim:        %s",
        "disabled" if config.no_trim else f"threshold={config.trim_threshold:.1e}",
    )
    logger.info("  Hardlink:    %s", config.use_hardlink)
    logger.info("  Dry run:     %s", config.dry_run)
    logger.info("=" * 60)

    # ---------------------------------------------------------------
    # Stage 1: Filter & synchronise
    # ---------------------------------------------------------------
    filtered_dir = stage_filter(config)

    # ---------------------------------------------------------------
    # Stage 2: Discover episodes
    # ---------------------------------------------------------------
    logger.info("Stage 2/%d: Discovering direct episodes...", n_stages)
    episodes = _discover_direct_episodes(filtered_dir)

    if not episodes:
        raise RuntimeError(
            "No episodes found. Verify the input directory contains "
            "Tissue#N / collector / phase / action folders with camera "
            "directories and ee_csv.csv."
        )

    # ---------------------------------------------------------------
    # Stage 2.5: Trim stationary frames (optional)
    # ---------------------------------------------------------------
    trim_stats: Dict[str, int] = {}
    if not config.no_trim:
        stage_label = f"Stage 3/{n_stages}"
        logger.info("%s: Trimming stationary frames...", stage_label)
        episodes_with_trims, trim_stats = stage_trim_direct(
            episodes, config.trim_threshold
        )
    else:
        # No trim: use full frame ranges
        episodes_with_trims = []
        for ep in episodes:
            left_dir = ep.source_dir / LEFT_IMG_DIR
            frame_files = list_sorted_frames(
                left_dir, CAMERA_SUFFIXES[LEFT_IMG_DIR]
            )
            n_frames = len(frame_files)
            episodes_with_trims.append((ep, 0, max(0, n_frames - 1)))

    # ---------------------------------------------------------------
    # Stage 3/4: Restructure
    # ---------------------------------------------------------------
    final_stage = n_stages
    logger.info("Stage %d/%d: Restructuring...", final_stage, n_stages)
    report = stage_restructure_direct(config, episodes_with_trims)
    report.trim_stats = trim_stats

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    report.elapsed_s = time.time() - overall_start
    _log_report(report)

    return report


# =============================================================================
# MAIN PIPELINE
# =============================================================================


def run_pipeline(config: PipelineConfig) -> PipelineReport:
    """Execute the full restructuring pipeline.

    Dispatches to the **direct** pipeline when ``config.annotations_dir``
    is ``None``, or the **annotation** pipeline when it is set.

    Args:
        config: Pipeline configuration.

    Returns:
        PipelineReport with per-episode results and summary statistics.
    """
    # ---------- Direct Mode ----------
    if config.annotations_dir is None:
        logger.info("No annotations directory provided — using Direct Mode.")
        return run_pipeline_direct(config)

    # ---------- Annotation Mode ----------
    overall_start = time.time()

    # Validate inputs early
    if not config.source_dir.exists():
        raise ValueError(f"Source directory not found: {config.source_dir}")
    if not config.annotations_dir.exists():
        raise ValueError(
            f"Annotations directory not found: {config.annotations_dir}"
        )

    n_stages = 3 if config.no_trim else 4
    logger.info("=" * 60)
    logger.info("Dataset Restructuring Pipeline — Annotation Mode (%d stages)", n_stages)
    logger.info("  Source:      %s", config.source_dir)
    logger.info("  Annotations: %s", config.annotations_dir)
    logger.info("  Output:      %s", config.output_dir)
    logger.info("  Dataset:     %s", config.dataset_name)
    logger.info("  FPS:         %d", config.fps)
    logger.info("  Trim:        %s", "disabled" if config.no_trim else f"threshold={config.trim_threshold:.1e}")
    logger.info("  Hardlink:    %s", config.use_hardlink)
    logger.info("  Dry run:     %s", config.dry_run)
    logger.info("=" * 60)

    # ---------------------------------------------------------------
    # Stage 1: Filter & synchronise
    # ---------------------------------------------------------------
    filtered_dir = stage_filter(config)

    # ---------------------------------------------------------------
    # Stage 2: Plan affordance slices
    # ---------------------------------------------------------------
    planned = stage_plan(config, filtered_dir)

    # ---------------------------------------------------------------
    # Stage 2.5: Trim stationary frames (optional)
    # ---------------------------------------------------------------
    trim_stats: Dict[str, int] = {}
    if not config.no_trim:
        planned, trim_stats = stage_trim(planned, config.trim_threshold)

    # ---------------------------------------------------------------
    # Stage 3: Restructure
    # ---------------------------------------------------------------
    report = stage_restructure(config, planned)
    report.trim_stats = trim_stats

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    report.elapsed_s = time.time() - overall_start
    _log_report(report)

    return report


def _log_report(report: PipelineReport) -> None:
    """Print a structured summary of the pipeline run."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info("  Planned:    %d episodes", report.total_planned)
    logger.info("  Processed:  %d episodes", report.total_processed)
    logger.info("  Skipped:    %d episodes", report.total_skipped)
    logger.info("  Errors:     %d episodes", report.total_errors)
    logger.info("  Frames:     %d total frames copied", report.total_frames_copied)
    logger.info("  CSV rows:   %d total rows written", report.total_csv_rows)
    if report.trim_stats:
        logger.info(
            "  Trimming:   %d/%d episodes trimmed, %d frames removed",
            report.trim_stats.get("trimmed", 0),
            report.trim_stats.get("total", 0),
            report.trim_stats.get("frames_removed", 0),
        )
    logger.info("  Duration:   %.1f seconds", report.elapsed_s)
    logger.info("=" * 60)

    # Report any skipped/failed episodes
    skipped = [r for r in report.episode_results if r.skipped]
    failed = [r for r in report.episode_results if not r.success and not r.skipped]

    if skipped:
        logger.warning("")
        logger.warning("SKIPPED EPISODES (%d):", len(skipped))
        for r in skipped:
            logger.warning("  %s — %s", r.episode_id, r.error)

    if failed:
        logger.error("")
        logger.error("FAILED EPISODES (%d):", len(failed))
        for r in failed:
            logger.error("  %s — %s", r.episode_id, r.error)


# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description=(
            "Restructure and transform a surgical robot dataset into the "
            "canonical directory format.\n\n"
            "Two modes are supported:\n"
            "  Annotation Mode: provide --annotations-dir for JSON-based slicing.\n"
            "  Direct Mode:     omit --annotations-dir when each action folder\n"
            "                   is already a single episode (auto-detected)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example usage (Annotation Mode):\n"
            "  python reformat_data.py \\\n"
            "    --input Cholecystectomy/tissues \\\n"
            "    --annotations-dir Cholecystectomy/annotations \\\n"
            "    --output restructured_output\n"
            "\n"
            "Example usage (Direct Mode):\n"
            "  python reformat_data.py \\\n"
            "    --input Cholecystectomy \\\n"
            "    --output restructured_output\n"
        ),
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to raw data directory (e.g. Cholecystectomy).",
    )
    parser.add_argument(
        "--annotations-dir",
        required=False,
        type=Path,
        default=None,
        help=(
            "Path to annotations directory for annotation-based slicing. "
            "If omitted, Direct Mode is used (each action folder = 1 episode)."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output directory for the restructured dataset.",
    )
    parser.add_argument(
        "--dataset-name",
        default="Cholecystectomy",
        help="Name of the output dataset folder (default: Cholecystectomy).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target FPS for timestamp normalisation (default: 30).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4).",
    )
    parser.add_argument(
        "--trim-threshold",
        type=float,
        default=1e-4,
        help="Stationary frame trim threshold (default: 1e-4).",
    )
    parser.add_argument(
        "--no-trim",
        action="store_true",
        help="Disable stationary frame trimming.",
    )
    parser.add_argument(
        "--use-hardlink",
        action="store_true",
        help="Use hardlinks instead of copying files (faster if same volume).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without copying files.",
    )
    parser.add_argument(
        "--sort-by",
        choices=("name", "mtime"),
        default="name",
        help="Frame sort method (default: name).",
    )

    args = parser.parse_args()

    # Validate paths
    if not args.input.exists():
        logger.error("Source directory not found: %s", args.input)
        return 1

    if args.annotations_dir is not None and not args.annotations_dir.exists():
        logger.error(
            "Annotations directory not found: %s", args.annotations_dir
        )
        return 1

    config = PipelineConfig(
        source_dir=args.input,
        output_dir=args.output,
        annotations_dir=args.annotations_dir,
        dataset_name=args.dataset_name,
        fps=args.fps,
        workers=args.workers,
        trim_threshold=args.trim_threshold,
        no_trim=args.no_trim,
        use_hardlink=args.use_hardlink,
        dry_run=args.dry_run,
        sort_by=args.sort_by,
    )

    try:
        report = run_pipeline(config)
        if report.total_errors > 0:
            return 2
        return 0
    except ValueError as ve:
        logger.error("Invalid input: %s", ve)
        return 1
    except RuntimeError as re_err:
        logger.error("Pipeline error: %s", re_err)
        return 3
    except Exception as exc:
        logger.exception("Unhandled error: %s", exc)
        return 4


if __name__ == "__main__":
    sys.exit(main())