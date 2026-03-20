#!/usr/bin/env python3
"""
remove_stationary_frames.py

Trim stationary segments from the start and end of sliced surgical-robot
episodes.  Designed to run **after** slicing (``slice_affordance.py``) and
**before** LeRobot conversion (``dvrk_lerobot_converter_v2.1.py``).

The Problem:
    In a 10-second surgical video, the PSMs are often stationary for the first
    ~3 s and last ~3 s.  Feeding these near-constant action signals to an ACT
    model increases the variance of episode lengths, adds low-information frames,
    and depresses overall action standard deviation — hurting training.

Algorithm (per episode):
    1. Load ``ee_csv.csv`` and extract joint-space columns that fully define
       the actual PSM state:
           psm1_js[0..5], psm1_jaw, psm2_js[0..5], psm2_jaw   (14 values)
    2. Compute the per-frame L2 norm of the first-order difference
       (delta between consecutive rows) across these 14 columns.
    3. **Start trim** — scan forward to find the first frame whose delta
       exceeds ``--threshold``.
    4. **End trim** — scan backward to find the last frame whose delta
       exceeds ``--threshold``.  The trimmed range is ``[trim_start, trim_end]``
       (inclusive).
    5. If the trimmed episode has fewer than ``--min-episode-length`` frames,
       skip it (do not trim to near-zero).
    6. Overwrite ``ee_csv.csv`` with only the kept rows, and delete image
       files outside the kept range from every camera directory.

Programmatic API:
    from remove_stationary_frames import run_remove_stationary_frames

    stats = run_remove_stationary_frames(
        base_dir=Path("dataset_sliced"),
        threshold=1e-4,
        min_episode_length=10,
        workers=4,
        dry_run=False,
    )
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Try the project logger; fall back to stdlib logging if not installed.
# ---------------------------------------------------------------------------
try:
    from surpass_data_collection.logger_config import get_logger

    logger = get_logger(__name__)
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KINEMATIC_CSV_NAME: str = "ee_csv.csv"

# Joint-space columns that fully define the actual PSM end-effector state.
# 6 joint angles per arm + 1 jaw angle per arm = 14 values.
MOTION_COLUMNS: List[str] = [
    "psm1_js[0]",
    "psm1_js[1]",
    "psm1_js[2]",
    "psm1_js[3]",
    "psm1_js[4]",
    "psm1_js[5]",
    "psm1_jaw",
    "psm2_js[0]",
    "psm2_js[1]",
    "psm2_js[2]",
    "psm2_js[3]",
    "psm2_js[4]",
    "psm2_js[5]",
    "psm2_jaw",
    # Cartesian poses (for safer motion detection in SO(3) space)
    "psm1_pose.orientation.x",
    "psm1_pose.orientation.y",
    "psm1_pose.orientation.z",
    "psm1_pose.orientation.w",
    "psm2_pose.orientation.x",
    "psm2_pose.orientation.y",
    "psm2_pose.orientation.z",
    "psm2_pose.orientation.w",
]

# Camera modality directories (must stay in sync with slice_affordance.py)
CAMERA_DIRS: List[str] = [
    "left_img_dir",
    "right_img_dir",
    "endo_psm1",
    "endo_psm2",
]

# Image extensions to consider for deletion (lowercase, with dot)
_IMAGE_EXTS = frozenset((".jpg", ".jpeg", ".png"))

# Defaults
DEFAULT_THRESHOLD: float = 1e-4
DEFAULT_MIN_EPISODE_LENGTH: int = 10
DEFAULT_WORKERS: int = 4


# ---------------------------------------------------------------------------
# Episode discovery
# ---------------------------------------------------------------------------


def discover_episodes(base_dir: Path) -> List[Path]:
    """Walk *base_dir* and return every directory containing ``ee_csv.csv``.

    Results are sorted for deterministic ordering.

    Uses ``os.scandir`` recursion for faster traversal than ``os.walk``
    on large directory trees (avoids stat-ing non-directory entries).
    """
    csv_name = KINEMATIC_CSV_NAME
    episodes: List[Path] = []

    def _scan(path: str) -> None:
        try:
            with os.scandir(path) as it:
                has_csv = False
                subdirs: List[str] = []
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        subdirs.append(entry.path)
                    elif entry.name == csv_name:
                        has_csv = True
                if has_csv:
                    episodes.append(Path(path))
                for d in subdirs:
                    _scan(d)
        except PermissionError:
            pass

    _scan(str(base_dir))
    episodes.sort()
    logger.info(f"Discovered {len(episodes)} episodes under {base_dir}")
    return episodes


# ---------------------------------------------------------------------------
# Motion detection
# ---------------------------------------------------------------------------


def compute_deltas(csv_path: Path) -> Tuple[Optional[np.ndarray], int]:
    """Read *csv_path* and return the per-frame L2 delta array.

    Uses ``pandas.read_csv`` (C engine) for ~5-10× faster parsing than
    ``numpy.loadtxt``.  Only the 14 motion columns are loaded.

    Returns:
        (deltas, n_rows) where *deltas* has shape ``(n_rows - 1,)`` holding
        ``||row[t+1] - row[t]||₂`` over the motion columns, or ``(None, 0)``
        if the file cannot be processed.
    """
    try:
        # Read header to find which motion columns exist
        header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    except Exception as e:
        logger.warning(f"Cannot read CSV header from {csv_path}: {e}")
        return None, 0

    # Select only the columns present in this file
    cols_present = [c for c in MOTION_COLUMNS if c in header]
    if not cols_present:
        logger.warning(f"No motion columns found in {csv_path}")
        return None, 0

    # Load only the required columns via C engine (much faster than np.loadtxt)
    try:
        data = pd.read_csv(
            csv_path,
            usecols=cols_present,
            dtype=np.float64,
            engine="c",
        ).values  # → numpy array, shape (n_rows, n_cols)
    except Exception as e:
        logger.warning(f"Failed to parse {csv_path}: {e}")
        return None, 0

    n_rows = data.shape[0]
    if n_rows < 2:
        return None, n_rows

    # --- Topological Fix: Quaternion Continuity ---
    # Quaternions q and -q represent the SAME rotation. np.diff would see a
    # massive jump (~2.0) if a sign flip occurs at a singularity. We enforce
    # shortest-path continuity (flipping signs if dot product < 0) before diff.
    quat_groups = [
        ["psm1_pose.orientation.x", "psm1_pose.orientation.y", "psm1_pose.orientation.z", "psm1_pose.orientation.w"],
        ["psm2_pose.orientation.x", "psm2_pose.orientation.y", "psm2_pose.orientation.z", "psm2_pose.orientation.w"]
    ]
    for q_group in quat_groups:
        indices = [i for i, c in enumerate(cols_present) if c in q_group]
        if len(indices) == 4:
            # Vectorized sign-flip correction
            q_arr = data[:, indices]  # n_rows, 4
            for i in range(1, n_rows):
                if np.dot(q_arr[i-1], q_arr[i]) < 0:
                    q_arr[i] *= -1.0
            data[:, indices] = q_arr

    # First-order difference  →  L2 norm per frame
    # np.diff + np.linalg.norm is already optimal for contiguous arrays.
    diffs = np.diff(data, axis=0)             # (n_rows-1, n_cols)
    # Einsum for squared L2 norm avoids a temporary allocation that
    # np.linalg.norm creates internally (it does sqrt(sum(x**2))).
    # For 14+ columns × thousands of rows, this is measurably faster.
    deltas = np.sqrt(np.einsum("ij,ij->i", diffs, diffs))  # (n_rows-1,)

    return deltas, n_rows


def find_trim_range(
    deltas: np.ndarray,
    n_rows: int,
    threshold: float,
) -> Tuple[int, int]:
    """Return ``(trim_start, trim_end)`` — the inclusive range of rows to keep.

    *deltas* has length ``n_rows - 1``; ``deltas[t]`` is the motion between
    row *t* and row *t+1*.

    * ``trim_start`` = first *t* where ``deltas[t] > threshold``
    * ``trim_end``   = last *t* where ``deltas[t] > threshold`` **+ 1**
      (so we keep the frame *after* the last moving transition)

    If no frame exceeds the threshold the full range ``(0, n_rows - 1)`` is
    returned (no trim).

    Uses ``np.argmax`` on a boolean mask for O(n) scan with early-out
    semantics, avoiding the allocation of an index array from ``np.where``.
    """
    mask = deltas > threshold

    if not mask.any():
        # No motion detected at all — keep everything
        return 0, n_rows - 1

    # argmax on a bool array returns the index of the first True (O(n) scan)
    trim_start = int(mask.argmax())
    # Reverse scan: last True = len - 1 - argmax(reversed)
    trim_end = int(len(mask) - 1 - mask[::-1].argmax()) + 1

    # Clamp
    trim_end = min(trim_end, n_rows - 1)

    return trim_start, trim_end


# ---------------------------------------------------------------------------
# Trimming
# ---------------------------------------------------------------------------


def _sorted_images(dir_path: str) -> List[str]:
    """Return sorted list of image filenames in *dir_path*.

    Uses ``os.listdir`` (single syscall, returns strings) instead of
    ``Path.iterdir()`` which creates a Path object per entry.
    """
    try:
        names = os.listdir(dir_path)
    except (FileNotFoundError, NotADirectoryError):
        return []
    # Filter + sort in one pass via list comprehension + sort
    imgs = [n for n in names if os.path.splitext(n)[1].lower() in _IMAGE_EXTS]
    imgs.sort()
    return imgs


def trim_episode(
    episode_dir: Path,
    threshold: float = DEFAULT_THRESHOLD,
    min_episode_length: int = DEFAULT_MIN_EPISODE_LENGTH,
    dry_run: bool = False,
) -> Dict:
    """Trim stationary head/tail from a single episode **in-place**.

    Returns a stats dict with keys:
        episode, original_length, trimmed_length,
        start_removed, end_removed, skipped, reason
    """
    csv_path = episode_dir / KINEMATIC_CSV_NAME
    ep_str = str(episode_dir)

    stats: Dict = {
        "episode": ep_str,
        "original_length": 0,
        "trimmed_length": 0,
        "start_removed": 0,
        "end_removed": 0,
        "skipped": True,
        "reason": "",
    }

    if not csv_path.exists():
        stats["reason"] = "missing CSV"
        return stats

    # 1. Compute deltas
    deltas, n_rows = compute_deltas(csv_path)
    stats["original_length"] = n_rows

    if deltas is None or n_rows < 2:
        stats["reason"] = "too few rows"
        return stats

    # 2. Find trim range
    trim_start, trim_end = find_trim_range(deltas, n_rows, threshold)
    trimmed_length = trim_end - trim_start + 1

    stats["start_removed"] = trim_start
    stats["end_removed"] = n_rows - 1 - trim_end
    stats["trimmed_length"] = trimmed_length

    # 3. Check minimum length
    if trimmed_length < min_episode_length:
        stats["reason"] = f"trimmed length {trimmed_length} < min {min_episode_length}"
        return stats

    # 4. Nothing to do?
    if trim_start == 0 and trim_end == n_rows - 1:
        stats["skipped"] = False
        stats["reason"] = "no stationary frames detected"
        return stats

    if dry_run:
        stats["skipped"] = False
        stats["reason"] = "dry-run"
        return stats

    # ---- Actually trim ----
    stats["skipped"] = False
    stats["reason"] = "trimmed"

    # 5a. Trim CSV — use raw binary line slicing (avoids parsing every field)
    csv_str = str(csv_path)
    try:
        with open(csv_str, "rb") as f:
            raw = f.read()

        # Split into lines (preserving line endings for faithful rewrite)
        lines = raw.split(b"\n")

        # Handle optional trailing newline
        if lines and lines[-1] == b"":
            lines.pop()

        # lines[0] = header, lines[1..] = data rows
        # Keep header + rows [trim_start .. trim_end] (0-indexed in data rows)
        header_line = lines[0]
        # data rows are lines[1:], so kept range is lines[1+trim_start : 1+trim_end+1]
        kept = lines[1 + trim_start : 1 + trim_end + 1]

        with open(csv_str, "wb") as f:
            f.write(header_line)
            f.write(b"\n")
            f.write(b"\n".join(kept))
            f.write(b"\n")

        logger.debug(
            f"CSV trimmed: {n_rows} → {len(kept)} rows in {csv_path}"
        )
    except Exception as e:
        logger.error(f"Failed to trim CSV {csv_path}: {e}", exc_info=True)
        stats["reason"] = f"CSV write error: {e}"
        stats["skipped"] = True
        return stats

    # 5b. Trim image directories — use os.unlink on raw strings (no Path overhead)
    for cam_dir_name in CAMERA_DIRS:
        cam_dir_str = os.path.join(ep_str, cam_dir_name)
        images = _sorted_images(cam_dir_str)
        if not images:
            continue

        # Delete images outside [trim_start, trim_end]
        n_deleted = 0
        for fname in images[:trim_start]:
            try:
                os.unlink(os.path.join(cam_dir_str, fname))
                n_deleted += 1
            except OSError:
                pass
        for fname in images[trim_end + 1 :]:
            try:
                os.unlink(os.path.join(cam_dir_str, fname))
                n_deleted += 1
            except OSError:
                pass

        logger.debug(f"  {cam_dir_name}: deleted {n_deleted} images")

    return stats


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


def run_remove_stationary_frames(
    base_dir: Path,
    threshold: float = DEFAULT_THRESHOLD,
    min_episode_length: int = DEFAULT_MIN_EPISODE_LENGTH,
    workers: int = DEFAULT_WORKERS,
    dry_run: bool = False,
) -> List[Dict]:
    """Process all episodes under *base_dir*.

    Returns a list of per-episode stats dicts.
    """
    episodes = discover_episodes(base_dir)
    if not episodes:
        logger.warning(f"No episodes found under {base_dir}")
        return []

    all_stats: List[Dict] = []

    if workers <= 1:
        # Sequential — easier to debug
        for ep in tqdm(episodes, desc="Trimming episodes"):
            s = trim_episode(ep, threshold, min_episode_length, dry_run)
            all_stats.append(s)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    trim_episode, ep, threshold, min_episode_length, dry_run
                ): ep
                for ep in episodes
            }
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Trimming episodes",
            ):
                all_stats.append(fut.result())

    # Sort for deterministic output
    all_stats.sort(key=lambda s: s["episode"])
    return all_stats



