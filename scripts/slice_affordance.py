#!/usr/bin/env python3
"""
Slice original session data into per-action episodes using affordance_range.

Reads JSON annotations under `post_process/*_left_video/annotation/*.json`,
finds `affordance_range.start` and `affordance_range.end` (indices relative to the raw footage),
and creates an output dataset.

Crucially, this script supports slicing a *filtered* or *synchronized* dataset based on annotations
from the *original* raw dataset. It does this by:
1.  Looking up the timestamp of the start/end frames in the RAW dataset.
2.  Finding the corresponding frames in the TARGET dataset that match those timestamps.

This ensures that even if frames were dropped or the frame rate changed, the semantic
start/end points of the action are preserved.

Structure: out/tissue_<n>/<action>/episode_XXX/{left_img_dir,right_img_dir,endo_psm1,endo_psm2,ee_csv.csv}

Run with `--dry_run` to only list planned operations.
"""
import argparse
import csv
import json
import logging
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
from tqdm import tqdm

# Configure top-level logger
logging.basicConfig(level=logging.INFO, 
                    format="%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s")
logger = logging.getLogger("slice_affordance")


# --- utility helpers ---------------------------------------------------------
FRAME_RE = re.compile(r"frame(\d+)")

def episode_exists(dst_base: Path) -> bool:
    """Check whether an episode output directory already exists.

    Args:
        dst_base: Destination episode base directory.

    Returns:
        True if the episode directory exists and is non-empty.
    """
    return dst_base.exists() and any(dst_base.iterdir())

def find_sessions(post_process_dir: Path) -> Iterator[Tuple[Path, int, str]]:
    """Yield discovered post_process session directories.

    Args:
        post_process_dir: Root directory containing post_process entries.

    Yields:
        Tuples of (post_dir_path, tissue_num, session_name).
    """
    if not post_process_dir.is_dir():
        return
    for entry in post_process_dir.iterdir():
        if not entry.is_dir():
            continue
        m = re.match(r"cautery_tissue(\d+)_(.+)_left_video$", entry.name)
        if not m:
            continue
        tissue = int(m.group(1))
        session = m.group(2)
        yield entry, tissue, session


def read_annotation_jsons(annotation_dir: Path) -> Iterator[Path]:
    """Yield all JSON annotation files under annotation_dir.

    Args:
        annotation_dir: Directory to walk for .json files.

    Yields:
        Paths to .json files.
    """
    if not annotation_dir.is_dir():
        return
    for root, _, files in os.walk(annotation_dir):
        for fn in files:
            if fn.lower().endswith(".json"):
                yield Path(root) / fn


def ensure_dir(p: Path) -> None:
    """Create directory `p` if it does not exist (idempotent)."""
    p.mkdir(parents=True, exist_ok=True)


def extract_timestamp(filename: str) -> int:
    """Extracts nanosecond timestamp from image filename.

    Args:
        filename: Image filename like 'frame1756826516968031906_left.jpg'.

    Returns:
        The extracted timestamp in nanoseconds.

    Raises:
        ValueError: If the filename does not match the expected pattern.
    """
    m = FRAME_RE.search(filename)
    if m:
        return int(m.group(1))
    raise ValueError(f"Could not extract timestamp from filename: {filename}")


def _frame_key(fn: str) -> int | str:
    """Return a sort key for filenames: integer frame number/timestamp when present.

    Using a numeric key ensures correct ordering for both `frame0001` (index) 
    and `frame178...` (timestamp) naming conventions.
    """
    m = FRAME_RE.search(fn)
    if m:
        return int(m.group(1))
    return fn


def list_sorted_frames(src_dir: Path, suffix: str) -> List[str]:
    """List and return sorted filenames in `src_dir` ending with `suffix`.

    Args:
        src_dir: Directory containing frame files.
        suffix: Filename suffix (e.g. '_left.jpg').

    Returns:
        Sorted list of filenames (basename strings). Returns [] if dir missing.
    """
    if not src_dir.is_dir():
        return []
    files = [f.name for f in src_dir.iterdir() if f.is_file() and f.name.endswith(suffix)]
    files.sort(key=_frame_key)
    return files


# --- CSV slicing -------------------------------------------------------------
def slice_ee_csv(src_csv: Path, out_csv: Path, start: int, end: int) -> Tuple[int, str]:
    """Slice rows from a CSV according to frame indices and write to out_csv.

    The function streams the input CSV to avoid loading large files wholly into memory.
    It detects presence of a header by attempting to convert the first cell to int.

    Args:
        src_csv: Source CSV path.
        out_csv: Destination CSV path to write sliced rows.
        start: Start index (inclusive) mapped to data row index 0.
        end: End index (inclusive).

    Returns:
        Tuple (number_of_rows_written, status_string). Status can be 'ok', 'missing', or 'empty'.
    """
    if not src_csv.exists():
        return 0, "missing"

    start_idx = max(0, start)
    end_idx = max(0, end)

    written = 0
    try:
        with src_csv.open("r", newline="") as fin:
            reader = csv.reader(fin)
            try:
                first_row = next(reader)
            except StopIteration:
                return 0, "empty"

            # Determine header presence: if first cell casts to int -> it's data
            header = None
            try:
                int(first_row[0])
                # first_row is data
                first_data_row = first_row
                # We want to continue iteration starting from first_data_row then reader
                data_iter = (r for r in ([first_data_row] + list(reader)))
            except Exception:
                header = first_row
                data_iter = reader

            # Write only the slice [start_idx, end_idx] inclusive
            ensure_dir(out_csv.parent)
            with out_csv.open("w", newline="") as fout:
                writer = csv.writer(fout)
                if header is not None:
                    writer.writerow(header)

                # iterate data rows and only write rows in requested range
                for idx, row in enumerate(data_iter):
                    if idx < start_idx:
                        continue
                    if idx > end_idx:
                        break
                    writer.writerow(row)
                    written += 1
    except Exception as e:
        logger.exception("Failed to slice ee_csv %s -> %s: %s", src_csv, out_csv, e)
        return written, "error"
    return written, "ok"


# --- file copying with concurrency & optional hardlinks ---------------------
def copy_or_link(
    src: Path,
    dst: Path,
    use_hardlink: bool = False,
    overwrite: bool = False,
) -> bool:
    """Copy or hardlink a single file from src to dst.

    Args:
        src: Source file path.
        dst: Destination file path.
        use_hardlink: Attempt hardlink before copying.
        overwrite: Whether to overwrite dst if it exists.

    Returns:
        True on success, False otherwise.
    """
    try:
        if dst.exists() and not overwrite:
            return True  # treated as success (already present)

        ensure_dir(dst.parent)

        if use_hardlink:
            try:
                if dst.exists():
                    dst.unlink()
                os.link(src, dst)
                return True
            except Exception:
                pass  # fallback to copy

        if dst.exists() and overwrite:
            dst.unlink()

        shutil.copy2(src, dst)
        return True

    except Exception:
        logger.debug("Failed to copy %s -> %s", src, dst, exc_info=True)
        return False





# --- main orchestration -----------------------------------------------------
def plan_episodes(
    post_dir: Path, 
    cautery_dir: Path, 
    out_dir: Path,
    source_dataset_dir: Optional[Path] = None
) -> List[Tuple[Path, Path, Path, Path, int, int]]:
    """Scan annotations and plan episodes to create.

    Args:
        post_dir: Root post_process directory.
        cautery_dir: Root raw cautery directory (used for reference indices).
        out_dir: Destination root for sliced dataset.
        source_dataset_dir: Root directory of data to actually copy (if None, use cautery_dir).

    Returns:
        List of tuples: (annotation_json, ref_session_dir, source_session_dir, dst_base, start, end)
    """
    planned = []
    counters = {}
    
    # If source location not specified, assume we are slicing the raw data itself (legacy mode)
    if source_dataset_dir is None:
        source_dataset_dir = cautery_dir

    for post_path, tissue_num, session_name in find_sessions(post_dir):
        annotation_dir = post_path / "annotation"
        if not annotation_dir.is_dir():
            continue
            
        # Reference path (RAW data) - used to resolve index N to Timestamp T
        ref_session_dir = cautery_dir / f"cautery_tissue#{tissue_num}" / session_name
        
        # Source path (TARGET data to slice) - used to find frame closest to Timestamp T
        # We assume the directory structure is preserved: tissue_N/session_name
        if source_dataset_dir == cautery_dir:
             src_session_dir = ref_session_dir
        else:
             # Handle possible structural differences. 
             # Filtered dataset: tissue_1/session_name
             # Raw dataset: cautery_tissue#1/session_name
             # We try a few strict patterns, fallback to flexible matching logic can be added if needed.
             # Current assumption: Source matches `tissue_{N}/{session}` pattern from conversion scripts.
             src_session_dir = source_dataset_dir / f"tissue_{tissue_num}" / session_name
             if not src_session_dir.exists():
                 # Fallback: maybe source has same names as raw?
                 src_session_dir = source_dataset_dir / f"cautery_tissue#{tissue_num}" / session_name

        if not ref_session_dir.is_dir():
            logger.warning("Reference session dir not found: %s (skipping)", ref_session_dir)
            continue
            
        if not src_session_dir.is_dir():
             logger.warning("Source session dir not found: %s (skipping)", src_session_dir)
             continue

        for ann_path in read_annotation_jsons(annotation_dir):
            try:
                ann = json.loads(ann_path.read_text())
            except Exception as e:
                logger.warning("Failed to read %s: %s", ann_path, e)
                continue

            ar = ann.get("affordance_range") or ann.get("afforadance_range")
            if not ar:
                logger.warning("No affordance_range in %s", ann_path)
                continue
            try:
                start = int(ar.get("start"))
                end = int(ar.get("end"))
            except Exception:
                logger.warning("Invalid affordance_range in %s", ann_path)
                continue

            key = (tissue_num, session_name)
            idx = counters.get(key, 0) + 1
            counters[key] = idx
            episode_name = f"episode_{idx:03d}"
            dst_base = out_dir / f"tissue_{tissue_num}" / session_name / episode_name
            
            planned.append((ann_path, ref_session_dir, src_session_dir, dst_base, start, end))
            
    return planned


def process_episode(
    ref_session_dir: Path,
    src_session_dir: Path,
    dst_base: Path,
    start_idx: int,
    end_idx: int,
    workers: int,
    use_hardlink: bool,
    overwrite: bool,
) -> Tuple[int, int, int]:
    """Process a single episode with index-to-timestamp alignment.

    1. Reads file list from `ref_session_dir` (raw data) to find `start_timestamp` and `end_timestamp`.
    2. Reads file list from `src_session_dir` (target data).
    3. Finds indices in target data that correspond to start/end timestamps.
    4. Copies files and slices CSV based on new target indices.

    Args:
        ref_session_dir: Directory of original raw session (reference).
        src_session_dir: Directory of data to slice (source).
        dst_base: Destination base directory for this episode.
        start_idx: Start frame index in REFERENCE (RAW) coordinate system.
        end_idx: End frame index in REFERENCE (RAW) coordinate system.
        workers: Number of parallel file-copy workers.
        use_hardlink: Whether to attempt hardlinks instead of copying.
        overwrite: Overwrite existing destination.

    Returns:
        Tuple (copied_count, missing_count, ee_rows_written)
    """
    # 1. Resolve Timestamps from Reference (Raw) Data
    ref_left_dir = ref_session_dir / "left_img_dir"
    ref_files = list_sorted_frames(ref_left_dir, "_left.jpg")
    
    # Safety clamp
    s_idx = max(0, min(start_idx, len(ref_files) - 1))
    e_idx = max(0, min(end_idx, len(ref_files) - 1))
    
    if not ref_files:
        logger.error(f"Reference directory empty: {ref_left_dir}")
        return 0, 0, 0
        
    try:
        t_start = extract_timestamp(ref_files[s_idx])
        t_end = extract_timestamp(ref_files[e_idx])
    except ValueError as e:
        logger.error(f"Failed to extract timestamp from reference files: {e}")
        return 0, 0, 0

    # 2. Map Timestamps to Source (Target/Filtered) Data Indices
    src_left_dir = src_session_dir / "left_img_dir"
    src_files = list_sorted_frames(src_left_dir, "_left.jpg")
    
    if not src_files:
        logger.error(f"Source directory empty: {src_left_dir}")
        return 0, 0, 0

    # Extract all source timestamps efficiently
    # Optimization: If filenames are standard 'frame<Timestamp>...', this is fast
    try:
        src_timestamps = np.array([extract_timestamp(f) for f in src_files])
    except ValueError:
        logger.error(f"Source files in {src_left_dir} have invalid timestamp format")
        return 0, 0, 0

    # Use binary search to find the closest insertion points
    new_start_idx = int(np.searchsorted(src_timestamps, t_start, side="left"))
    new_end_idx = int(np.searchsorted(src_timestamps, t_end, side="right")) - 1

    # Clamp to valid range
    new_start_idx = max(0, min(new_start_idx, len(src_files) - 1))
    new_end_idx = max(0, min(new_end_idx, len(src_files) - 1))
    
    # Validation: Ensure we picked frames reasonably close in time?
    # For now, we trust the nearest match logic (searchsorted).

    # 3. Setup Paths for Copying
    left_src = src_session_dir / "left_img_dir"
    right_src = src_session_dir / "right_img_dir"
    psm1_src = src_session_dir / "endo_psm1"
    psm2_src = src_session_dir / "endo_psm2"
    ee_csv_src = src_session_dir / "ee_csv.csv"

    left_dst = dst_base / "left_img_dir"
    right_dst = dst_base / "right_img_dir"
    psm1_dst = dst_base / "endo_psm1"
    psm2_dst = dst_base / "endo_psm2"
    ee_out = dst_base / "ee_csv.csv"

    # 4. Prepare File Lists based on NEW indices
    # Optimization: We already have src_files (left), just need to list others if they exist
    right_files = list_sorted_frames(right_src, "_right.jpg")
    psm1_files = list_sorted_frames(psm1_src, "_psm1.jpg")
    psm2_files = list_sorted_frames(psm2_src, "_psm2.jpg")

    indices = range(new_start_idx, new_end_idx + 1)
    
    # Safe list getter helper
    def get_slice(files: List[str], idxs: range) -> List[str]:
         return [files[i] for i in idxs if 0 <= i < len(files)]

    left_to_copy = get_slice(src_files, indices)
    right_to_copy = get_slice(right_files, indices)
    psm1_to_copy = get_slice(psm1_files, indices)
    psm2_to_copy = get_slice(psm2_files, indices)

    # 5. Execute Copy (Unified ThreadPool for entire episode)
    # Collect all copy tasks first
    all_copy_tasks: List[Tuple[Path, Path]] = []
    
    for s_dir, d_dir, names in (
        (left_src, left_dst, left_to_copy),
        (right_src, right_dst, right_to_copy),
        (psm1_src, psm1_dst, psm1_to_copy),
        (psm2_src, psm2_dst, psm2_to_copy),
    ):
        if not names:
            continue
        ensure_dir(d_dir)
        for name in names:
            all_copy_tasks.append((s_dir / name, d_dir / name))

    copied = 0
    missing = 0
    
    # Use one pool for all files in the episode
    if all_copy_tasks:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            future_to_pair = {
                ex.submit(copy_or_link, src, dst, use_hardlink, overwrite): (src, dst) 
                for src, dst in all_copy_tasks
            }
            for fut in as_completed(future_to_pair):
                src, dst = future_to_pair[fut]
                try:
                    ok = fut.result()
                except Exception:
                    ok = False
                if ok:
                    copied += 1
                else:
                    if not src.exists():
                        missing += 1
                        logger.debug("Missing source file: %s", src)
                    else:
                        missing += 1
                        logger.warning("Failed to copy file despite existing: %s", src)

    # 6. Slice CSV using NEW indices
    ee_rows = 0
    status = "skipped"

    if not ee_out.exists() or overwrite:
        ee_rows, status = slice_ee_csv(ee_csv_src, ee_out, new_start_idx, new_end_idx)

    logger.debug(
        "Created %s: [%d..%d]->[%d..%d] (frames=%d, ee=%d) copied=%d %s",
        dst_base.name,
        start_idx, end_idx, new_start_idx, new_end_idx,
        len(left_to_copy), ee_rows, copied, status
    )

    return copied, missing, ee_rows


def main() -> None:
    """Parse CLI args and orchestrate episode planning + execution."""
    p = argparse.ArgumentParser()
    p.add_argument("--post_process_dir", default="post_process", help="post_process root dir")
    p.add_argument("--cautery_dir", default="cautery", help="original cautery dir (reference indices)")
    p.add_argument("--source_dataset_dir", default=None, help="actual dataset to copy/slice (default: same as cautery_dir)")
    p.add_argument("--out_dir", default="dataset_sliced", help="output dataset root")
    p.add_argument("--dry_run", action="store_true", help="only list planned operations")
    p.add_argument("--execute", dest="dry_run", action="store_false")
    p.add_argument("--workers", type=int, default=8, help="number of parallel copy workers (per-dir)")
    p.add_argument(
        "--hardlink",
        action="store_true",
        help="attempt to create hardlinks instead of copying when possible (fast, no extra space)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing episode outputs.",
    )
    p.add_argument(
        "--episode-workers",
        type=int,
        default=None,
        help="number of parallel episode workers (default: min(4, cpu_count))",
    )
    args = p.parse_args()

    post_dir = Path(args.post_process_dir)
    cautery_dir = Path(args.cautery_dir)
    source_dir = Path(args.source_dataset_dir) if args.source_dataset_dir else None
    out_dir = Path(args.out_dir)

    planned = plan_episodes(post_dir, cautery_dir, out_dir, source_dataset_dir=source_dir)
    logger.info("Planned %d episodes to create under: %s", len(planned), out_dir)

    if args.dry_run:
        for ann_path, ref_dir, src_dir, dst_base, start, end in planned:
            logger.info("PLAN: %s\n  Ref: %s\n  Src: %s\n  Out: %s\n  Raw Frames: %d..%d",
                        ann_path.name, ref_dir.name, src_dir.name, dst_base.name, start, end)
        return

    # Execute episodes in parallel
    print(f"Processing {len(planned)} episodes...")
    
    # Use ProcessPoolExecutor for episode-level parallelism
    # We use a default of 4 episode workers if not specified, to avoid memory explosion
    episode_workers = args.episode_workers if args.episode_workers else min(4, os.cpu_count() or 4)
    inner_workers = max(1, args.workers // episode_workers)
    
    logger.info(f"Pool: {episode_workers} workers, {inner_workers} threads/worker.")

    completed_count = 0
    total_copied = 0
    total_missing = 0 # Track globally for summary
    
    with ProcessPoolExecutor(max_workers=episode_workers) as executor:
        futures = {}
        for ann_path, ref_dir, src_dir, dst_base, start, end in planned:
            if dst_base.exists() and not args.overwrite:
                continue
                
            # Heuristic for hardlink in main process
            use_hardlink_arg = False
            if args.hardlink:
                try:
                    # Check against the ACTUAL source, not the reference
                    use_hardlink_arg = src_dir.exists() and (src_dir.stat().st_dev == out_dir.stat().st_dev)
                except Exception:
                    pass

            fut = executor.submit(
                process_episode,
                ref_dir,
                src_dir,
                dst_base,
                start,
                end,
                inner_workers,
                use_hardlink_arg,
                args.overwrite,
            )
            futures[fut] = dst_base

        # Create progress bar
        with tqdm(total=len(futures), desc="Processing episodes", unit="episode") as pbar:
            for fut in as_completed(futures):
                dst = futures[fut]
                completed_count += 1
                try:
                    c, m, er = fut.result()
                    total_copied += c
                    total_missing += m
                    pbar.set_postfix({
                        'copied': total_copied,
                        'missing': total_missing
                    })
                except Exception as e:
                    logger.error(f"Failed to process episode {dst}: {e}")
                    pbar.set_postfix({
                        'copied': total_copied,
                        'missing': total_missing,
                        'error': str(dst.name)
                    })
                finally:
                    pbar.update(1)

    logger.info("Done. Total copied files: %d, Total missing: %d", total_copied, total_missing)


if __name__ == "__main__":
    main()
