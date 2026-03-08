#!/usr/bin/env python
"""
CLI-based converter for DVRK data to LeRobot v2.1 format.

This tool provides a command-line interface to convert DVRK surgical robot datasets
into LeRobot format with timestamp-based alignment across multiple camera views.

Based on dvrk_zarr_to_lerobot.py but with:
- Support for new DVRK data format (timestamp-based filenames)
- All episodes go to training set
- CLI arguments for ease of use

python dvrk_lerobot_converter_v2.1.py --source-dir <insert_path_to_dvrk_data> --annotations-dir <insert_path_to_annotations> \
 --dataset-name <insert_dataset_name> --overwrite
"""

import sys
import os
import re
import glob as glob_module
import logging
import shutil
import time
import types
import argparse
import signal
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import traceback

import cv2
import numpy as np
import pandas as pd
from PIL import Image

# Note: LeRobot imports are delayed until inside ConversionWorker
# to allow setting HF_LEROBOT_HOME before module initialization

# ---------------------------------------------------------------------------
# Import strict synchronization pipeline.
# These scripts are NOT installed as packages, so we add to sys.path.
# ---------------------------------------------------------------------------
_SYNC_SCRIPTS_DIR = str(
    Path(__file__).resolve().parent.parent.parent / "sync_image_kinematics"
)
if _SYNC_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SYNC_SCRIPTS_DIR)

_POST_PROCESSING_DIR = str(
    Path(__file__).resolve().parent.parent.parent / "post_processing"
)
if _POST_PROCESSING_DIR not in sys.path:
    sys.path.insert(0, _POST_PROCESSING_DIR)

from filter_episodes import run_filter_episode, run_filter_episodes
from slice_affordance import plan_episodes, list_sorted_frames, extract_timestamp as sa_extract_timestamp
from remove_stationary_frames import compute_deltas, find_trim_range, MOTION_COLUMNS

# Default LeRobot home for UI display only
DEFAULT_LEROBOT_HOME = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "lerobot")

# =============================================================================
# CONFIGURATION
# =============================================================================
# Available video codecs:
CODEC_OPTIONS = [
    "h264",
    "h264_nvenc",
    "h264_amf",
    "h264_qsv",
    "libsvtav1",
]

# This global is updated from the CLI before conversion starts
VIDEO_CODEC = "h264"

# Intermediate image directory — use a FAST LOCAL drive (NVMe/SSD) to avoid
# bottlenecking encoding on the (possibly slower) dataset output drive.
# Set to None to use the default (inside the dataset directory).
import tempfile
TEMP_IMAGE_DIR = Path(tempfile.gettempdir()) / "lerobot_images"

# Directory names
LEFT_IMG_DIR = "left_img_dir"
RIGHT_IMG_DIR = "right_img_dir"
ENDO_PSM1_DIR = "endo_psm1"
ENDO_PSM2_DIR = "endo_psm2"
CSV_FILE = "ee_csv.csv"

# State and action column names for the CSV
STATES_NAME = [
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

ACTIONS_NAME = [
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

def safe_normalize(q, eps=1e-8):
    norm = np.linalg.norm(q)
    if norm < eps:
        # fallback to identity rotation
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return q / norm


def quat_inverse(q):
    q = safe_normalize(q)
    q_conj = q.copy()
    q_conj[:3] *= -1.0
    return q_conj  # safe since unit


def quat_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dtype=np.float32)


def ensure_quat_continuity(q_prev, q_curr):
    """
    Ensures shortest path by preventing sign flips.
    """
    if np.dot(q_prev, q_curr) < 0:
        return -q_curr
    return q_curr
    

def compute_action_hybrid_rel(action_t, action_next):
    """
    actions: (T, 16) array
    Returns:
        rel_actions: (T, 16) array
    """

    rel_action = np.zeros_like(action_t)

    for psm_arm in range(2):
        base = psm_arm * 8

        # --- current ---
        p_t = action_t[base:base+3]
        q_t = action_t[base+3:base+7]

        # --- next ---
        p_next = action_next[base:base+3]
        q_next = action_next[base+3:base+7]

        # Normalize
        q_t = safe_normalize(q_t)
        q_next = safe_normalize(q_next)

        # Fix sign ambiguity
        q_next = ensure_quat_continuity(q_t, q_next)

        # Relative position (frame 1)
        dp = p_next - p_t

        # Relative quaternion (local frame)
        q_inv = quat_inverse(q_t)
        dq = quat_multiply(q_inv, q_next)
        dq = safe_normalize(dq)

        # Normalize for safety
        dq = dq / np.linalg.norm(dq)

        rel_action[base:base+3] = dp
        rel_action[base+3:base+7] = dq
        rel_action[base+7] = action_t[base+7]

    return rel_action.astype(np.float32)


# =============================================================================
# DATA STRUCTURES
# =============================================================================
@dataclass
class FrameInfo:
    """Information about a single frame"""
    path: Path
    timestamp: int  # nanoseconds


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
_FRAME_TS_RE = re.compile(r'frame(\d+)')


def extract_timestamp(filename: str) -> int:
    """Extract timestamp from filename like 'frame1767971796430639266_psm1.jpg'"""
    match = _FRAME_TS_RE.search(filename)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not extract timestamp from {filename}")


def load_frames_from_dir(dir_path: Path) -> List[FrameInfo]:
    """Load all frames from a directory, sorted by timestamp"""
    frames = []
    if not dir_path.exists():
        return frames
    for f in dir_path.glob("*.jpg"):
        try:
            ts = extract_timestamp(f.name)
            frames.append(FrameInfo(path=f, timestamp=ts))
        except ValueError:
            continue
    return sorted(frames, key=lambda x: x.timestamp)

def validate_episode(episode_path: Path) -> Tuple[bool, str]:
    """Validate episode structure. Returns (is_valid, error_message)"""
    errors = []
    
    left_dir = episode_path / LEFT_IMG_DIR
    if not left_dir.exists():
        errors.append(f"Missing directory: {LEFT_IMG_DIR}")
    elif not list(left_dir.glob("*.jpg")):
        errors.append(f"No images in: {LEFT_IMG_DIR}")
    
    csv_path = episode_path / CSV_FILE
    if not csv_path.exists():
        errors.append(f"Missing CSV: {CSV_FILE}")
    
    if errors:
        return False, "\n".join(errors)
    return True, "Valid"


# =============================================================================
# PARALLEL VIDEO ENCODING (with GPU support)
# =============================================================================
def _encode_video_frames_custom(
    imgs_dir: Path,
    video_path: Path,
    fps: int,
    vcodec: str = "h264_nvenc",
    pix_fmt: str = "yuv420p",
) -> None:
    """Encode intermediate frames (JPEG or PNG) to video.  Supports CPU *and* GPU codecs.

    Unlike LeRobot's built-in ``encode_video_frames`` (which has a hard-
    coded whitelist of three CPU codecs), this accepts any codec known to
    the installed ffmpeg / PyAV build, including:
        h264_nvenc, h264_amf, h264_qsv   (GPU)
        h264, hevc, libsvtav1             (CPU)
    """
    import av

    video_path.parent.mkdir(parents=True, exist_ok=True)

    # Discover intermediate frames — JPEG first, fall back to PNG (legacy)
    digits = "[0-9]" * 6
    input_list = sorted(
        glob_module.glob(str(imgs_dir / f"frame_{digits}.jpg")),
        key=lambda x: int(x.split("_")[-1].split(".")[0]),
    )
    if not input_list:
        input_list = sorted(
            glob_module.glob(str(imgs_dir / f"frame_{digits}.png")),
            key=lambda x: int(x.split("_")[-1].split(".")[0]),
        )
    if not input_list:
        raise FileNotFoundError(f"No images found in {imgs_dir}.")

    dummy = Image.open(input_list[0])
    width, height = dummy.size

    # ---- codec-specific encoder options ----
    is_nvenc = "nvenc" in vcodec
    is_amf = "amf" in vcodec
    is_qsv = "qsv" in vcodec

    if is_nvenc:
        video_options = {"preset": "p4", "rc": "constqp", "qp": "28", "bf": "0", "g": "2"}
    elif is_amf:
        video_options = {"quality": "balanced", "rc": "cqp", "qp_i": "28", "qp_p": "28", "g": "2"}
    elif is_qsv:
        video_options = {"preset": "medium", "global_quality": "28", "g": "2"}
    else:
        # CPU codecs (h264, hevc, libsvtav1)
        video_options = {"crf": "30", "g": "2"}

    logging.getLogger("libav").setLevel(av.logging.ERROR)

    with av.open(str(video_path), "w") as output:
        stream = output.add_stream(vcodec, fps, options=video_options)
        stream.pix_fmt = pix_fmt
        stream.width = width
        stream.height = height

        # Pre-read images with a thread pool so NVENC is never starved
        READ_AHEAD = 32

        def _load_pil(path):
            return Image.open(path).convert("RGB")

        pool = ThreadPoolExecutor(max_workers=8)
        futures = {}
        for i in range(min(READ_AHEAD, len(input_list))):
            futures[i] = pool.submit(_load_pil, input_list[i])

        for i in range(len(input_list)):
            img = futures.pop(i).result()
            # keep pipeline full
            nxt = i + READ_AHEAD
            if nxt < len(input_list):
                futures[nxt] = pool.submit(_load_pil, input_list[nxt])

            frame = av.VideoFrame.from_image(img)
            packet = stream.encode(frame)
            if packet:
                output.mux(packet)

        pool.shutdown(wait=False)

        # Flush encoder
        packet = stream.encode()
        if packet:
            output.mux(packet)


def _parallel_encode_episode_videos(self, episode_index: int) -> None:
    """Drop-in replacement for LeRobotDataset.encode_episode_videos.

    * Encodes all video streams in parallel (ThreadPoolExecutor).
    * Uses the VIDEO_CODEC configured at the top of this script.
    * Supports GPU codecs (h264_nvenc, h264_amf, h264_qsv) that the
      original LeRobot encoder rejects.
    """
    def _encode_one(key):
        video_path = self.root / self.meta.get_video_file_path(episode_index, key)
        img_dir = self._get_image_file_path(
            episode_index=episode_index, image_key=key, frame_index=0
        ).parent
        _encode_video_frames_custom(
            img_dir, video_path, self.fps,
            vcodec=VIDEO_CODEC,
        )
        shutil.rmtree(img_dir)

    with ThreadPoolExecutor(max_workers=len(self.meta.video_keys)) as pool:
        futs = [pool.submit(_encode_one, key) for key in self.meta.video_keys]
        for f in futs:
            f.result()  # propagate exceptions

    # Update video info on first episode (mirrors original behaviour)
    if len(self.meta.video_keys) > 0 and episode_index == 0:
        self.meta.update_video_info()
        from lerobot.datasets.utils import write_info
        write_info(self.meta.info, self.meta.root)


def _patch_parallel_encoding(dataset):
    """Monkey-patch a LeRobotDataset instance to use parallel video encoding."""
    dataset.encode_episode_videos = types.MethodType(
        _parallel_encode_episode_videos, dataset
    )


def _save_image_jpeg(self, image, fpath: Path) -> None:
    """Drop-in replacement for LeRobotDataset._save_image.

    Saves as JPEG instead of PNG (5-10x faster) and uses cv2.imwrite
    which is faster than PIL for numpy arrays.
    """
    fpath = fpath.with_suffix(".jpg")
    if self.image_writer is None:
        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[0] == 3:
                image = image.transpose(1, 2, 0)
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(fpath), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            image.save(str(fpath), quality=95)
    else:
        self.image_writer.save_image(image=image, fpath=fpath)


def _get_image_file_path_jpg(self, episode_index: int, image_key: str, frame_index: int) -> Path:
    """Return .jpg path, using TEMP_IMAGE_DIR (fast local SSD) if configured."""
    from lerobot.datasets.utils import DEFAULT_IMAGE_PATH
    fpath = DEFAULT_IMAGE_PATH.format(
        image_key=image_key, episode_index=episode_index, frame_index=frame_index
    )
    base = TEMP_IMAGE_DIR if TEMP_IMAGE_DIR is not None else self.root
    return (base / fpath).with_suffix(".jpg")


def _patch_jpeg_image_saving(dataset):
    """Monkey-patch a LeRobotDataset to save intermediate images as JPEG.

    This eliminates the slow PNG zlib compression and uses faster cv2.imwrite.
    Also patches the image writer's write_image to handle .jpg properly.
    """
    dataset._save_image = types.MethodType(_save_image_jpeg, dataset)
    dataset._get_image_file_path = types.MethodType(_get_image_file_path_jpg, dataset)

    # Cap the image writer queue to prevent unbounded memory growth.
    # Without this, fast frame processing can queue 60GB+ of numpy arrays
    # before the writer threads flush them to disk.
    # Setting maxsize on the live queue works because Queue.put() checks it dynamically.
    if dataset.image_writer is not None and hasattr(dataset.image_writer, "queue"):
        dataset.image_writer.queue.maxsize = 128  # ~128 images ≈ ~200 MB max

    # Patch the image writer's worker to use cv2 for JPEG (faster than PIL)
    if dataset.image_writer is not None:
        import lerobot.datasets.image_writer as iw
        _original_write_image = iw.write_image

        def _write_image_fast(image, fpath: Path):
            fpath = Path(fpath)
            # If the file was pre-placed (hardlink from source), skip writing
            if fpath.exists():
                return
            if fpath.suffix.lower() in (".jpg", ".jpeg"):
                if isinstance(image, np.ndarray):
                    if image.ndim == 3 and image.shape[0] == 3:
                        image = image.transpose(1, 2, 0)
                    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    fpath.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(fpath), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                elif isinstance(image, Image.Image):
                    fpath.parent.mkdir(parents=True, exist_ok=True)
                    image.save(str(fpath), quality=95)
                else:
                    _original_write_image(image, fpath)
            else:
                _original_write_image(image, fpath)

        iw.write_image = _write_image_fast


# =============================================================================
# CONVERSION WORKER
# =============================================================================
class ConversionWorker:
    """Worker class for running the conversion synchronously."""
    
    def __init__(self, source_path: Path, output_dir: Path, dataset_name: str,
                 psm1_tool: str, psm2_tool: str, fps: int,
                 annotations_dir: Optional[Path] = None,
                 trim_threshold: float = 1e-4):
        self.source_path = source_path
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.psm1_tool = psm1_tool
        self.psm2_tool = psm2_tool
        self.fps = fps
        self.annotations_dir = annotations_dir
        self.trim_threshold = trim_threshold
        self.cancelled = False
    
    def cancel(self, signum=None, frame=None):
        print("\n[!] Cancellation requested by user. Terminating soon...")
        self.cancelled = True
    
    def run(self):
        try:
            self._run_conversion()
        except Exception as e:
            print(f"Error: {str(e)}\n{traceback.format_exc()}", file=sys.stderr)
            sys.exit(1)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format a duration in seconds to a human-readable string."""
        seconds = int(seconds)
        if seconds < 60:
            return f"{seconds}s"
        minutes, secs = divmod(seconds, 60)
        if minutes < 60:
            return f"{minutes}m {secs:02d}s"
        hours, mins = divmod(minutes, 60)
        return f"{hours}h {mins:02d}m {secs:02d}s"
    
    def _run_conversion(self):
        # Set HF_LEROBOT_HOME BEFORE importing LeRobot modules
        os.environ["HF_LEROBOT_HOME"] = str(self.output_dir)
        
        # Now import LeRobot (this will use the custom HF_LEROBOT_HOME)
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.datasets.utils import write_info

        self._run_pipeline_conversion(LeRobotDataset, write_info)

    def _get_image_shapes(self, episodes: List[Path]) -> Tuple[Tuple, Tuple]:
            """Get image shapes from the first valid episode"""
            for episode_path in episodes:
                left_dir = episode_path / LEFT_IMG_DIR
                endo_dir = episode_path / ENDO_PSM1_DIR
                
                # Get endoscope shape
                endo_shape = (540, 960, 3)  # default
                left_images = list(left_dir.glob("*.jpg")) if left_dir.exists() else []
                if left_images:
                    img = Image.open(left_images[0])
                    arr = np.array(img)
                    endo_shape = arr.shape[:2] + (3,)
                
                # Get wrist camera shape
                wrist_shape = (480, 640, 3)  # default
                endo_images = list(endo_dir.glob("*.jpg")) if endo_dir.exists() else []
                if endo_images:
                    img = Image.open(endo_images[0])
                    arr = np.array(img)
                    wrist_shape = arr.shape[:2] + (3,)
                
                return endo_shape, wrist_shape
            
            return (540, 960, 3), (480, 640, 3)
    # -----------------------------------------------------------------
    # Dataset creation helpers (shared by pipeline and flat modes)
    # -----------------------------------------------------------------

    def _create_dataset(self, LeRobotDataset, img_shape_endo, img_shape_wrist):
        """Create a fresh LeRobot dataset with the standard feature schema."""
        dataset = LeRobotDataset.create(
            repo_id=self.dataset_name,
            use_videos=True,
            robot_type="dvrk",
            fps=self.fps,
            features={
                "observation.images.endoscope.left": {
                    "dtype": "video",
                    "shape": img_shape_endo,
                    "names": ["height", "width", "channel"],
                },
                "observation.images.endoscope.right": {
                    "dtype": "video",
                    "shape": img_shape_endo,
                    "names": ["height", "width", "channel"],
                },
                "observation.images.wrist.left": {
                    "dtype": "video",
                    "shape": img_shape_wrist,
                    "names": ["height", "width", "channel"],
                },
                "observation.images.wrist.right": {
                    "dtype": "video",
                    "shape": img_shape_wrist,
                    "names": ["height", "width", "channel"],
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": (len(STATES_NAME),),
                    "names": [STATES_NAME],
                },
                "action": {
                    "dtype": "float32",
                    "shape": (len(ACTIONS_NAME),),
                    "names": [ACTIONS_NAME],
                },
                "action_hybrid_relative": {
                    "dtype": "float32",
                    "shape": (len(ACTIONS_NAME),),
                    "names": [ACTIONS_NAME],
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
            image_writer_processes=0,
            image_writer_threads=16,
            tolerance_s=0.1,
            batch_encoding_size=1,  # Encode after each episode for crash resilience
        )
        _patch_parallel_encoding(dataset)
        _patch_jpeg_image_saving(dataset)
        print(f"  Image writer: 16 threads (JPEG)  |  Video encoding: parallel, codec={VIDEO_CODEC}")
        return dataset



    # -----------------------------------------------------------------
    # Pipeline mode: filter → plan → (trim) → convert
    # -----------------------------------------------------------------

    def _trim_stationary_episodes(self, planned):
        """Adjust planned episode indices to remove stationary head/tail.

        Groups episodes by reference session to load each CSV only once.
        Returns (trimmed_planned, stats_dict).
        """
        from collections import defaultdict
        import csv as csv_mod

        # Group episode indices by ref_session_dir
        groups = defaultdict(list)  # ref_dir → [(list_index, start, end), ...]
        for i, (ann, ref, src, dst, start, end) in enumerate(planned):
            groups[ref].append((i, start, end))

        trimmed_planned = list(planned)  # shallow copy of the list
        stats = {"total": len(planned), "trimmed": 0, "frames_removed": 0}

        for ref_dir, episodes_in_session in groups.items():
            csv_path = ref_dir / CSV_FILE
            if not csv_path.exists():
                continue

            # Load CSV and compute deltas using optimized pandas engine
            try:
                full_deltas, _ = compute_deltas(csv_path)
                if full_deltas is None:
                    continue
            except Exception as e:
                print(f"  Warning: could not compute deltas for {csv_path}: {e}", file=sys.stderr)
                continue

            # Trim each episode's range within this session
            for list_idx, start, end in episodes_in_session:
                # Extract the delta slice for this episode
                # deltas[i] = ||row[i+1] - row[i]||, so for rows [start:end]
                # we need deltas[start : end-1]  (end-1 transitions)
                d_start = max(start, 0)
                d_end = min(end, len(full_deltas))  # deltas has len(data)-1 entries
                if d_end <= d_start:
                    continue

                ep_deltas = full_deltas[d_start:d_end]
                ep_n_rows = end - start + 1

                trim_start, trim_end = find_trim_range(ep_deltas, ep_n_rows, self.trim_threshold)

                # No motion detected, or no trim needed
                if trim_start == 0 and trim_end == ep_n_rows - 1:
                    continue

                new_start = start + trim_start
                new_end   = start + trim_end

                if new_end - new_start + 1 < 10:  # min episode length
                    continue

                if new_start != start or new_end != end:
                    ann, ref, src, dst, _, _ = trimmed_planned[list_idx]
                    trimmed_planned[list_idx] = (ann, ref, src, dst, new_start, new_end)
                    removed = ep_n_rows - (new_end - new_start + 1)
                    stats["trimmed"] += 1
                    stats["frames_removed"] += removed

        return trimmed_planned, stats

    def _run_pipeline_conversion(self, LeRobotDataset, write_info):
        """Full pipeline: run_filter_episodes → plan_episodes → accelerated convert."""

        # ---------------------------------------------------------------
        # Stage 1: Filter and synchronise episodes
        # ---------------------------------------------------------------
        filtered_dir = self.output_dir / "_filtered_cache"
        max_time_diff_ms = self.fps

        print(
            f"Stage 1/{4 if self.trim_threshold > 0 else 3}: Filtering & synchronising episodes "
            f"(threshold={max_time_diff_ms:.1f}ms)..."
        )
        rc = run_filter_episodes(
            source_dir=str(self.source_path),
            out_dir=str(filtered_dir),
            max_time_diff=max_time_diff_ms,
            min_images=10,
            dry_run=False,
            overwrite=False,
            use_hardlink=True,
        )
        if rc != 0:
            print("Filtering stage failed. Check logs.", file=sys.stderr)
            sys.exit(rc)
        print("  Filtering complete.")

        if self.cancelled:
            print("Cancelled by user")
            sys.exit(1)

        # ---------------------------------------------------------------
        # Stage 2: Plan affordance-based episode slices
        # ---------------------------------------------------------------
        n_stages = 4 if self.trim_threshold > 0 else 3
        print(f"Stage 2/{n_stages}: Planning affordance slices...")
        planned = plan_episodes(
            self.annotations_dir,
            self.source_path,             # raw data = cautery reference dir
            Path("_unused"),              # out_dir placeholder (we don't copy)
            source_dataset_dir=filtered_dir,
        )
        print(f"  Planned {len(planned)} episodes")

        if not planned:
            print(
                "No episodes planned. Check that annotations, cautery, and "
                "filtered data paths are correct.", file=sys.stderr
            )
            sys.exit(1)

        if self.cancelled:
            print("Cancelled by user")
            sys.exit(1)

        # ---------------------------------------------------------------
        # Stage 2.5: Trim stationary frames (optional)
        # ---------------------------------------------------------------
        if self.trim_threshold > 0:
            print(
                f"Stage 3/{n_stages}: Trimming stationary frames "
                f"(threshold={self.trim_threshold})..."
            )
            planned, trim_stats = self._trim_stationary_episodes(planned)
            print(
                f"  Trimmed {trim_stats['trimmed']}/{trim_stats['total']} "
                f"episodes, removed {trim_stats['frames_removed']} total frames"
            )

            if self.cancelled:
                print("Cancelled by user")
                sys.exit(1)

        # ---------------------------------------------------------------
        # Stage N: Accelerated conversion to LeRobot
        # ---------------------------------------------------------------
        print(f"Stage {n_stages}/{n_stages}: Converting to LeRobot format...")
        output_path = self.output_dir / self.dataset_name
        print(f"Output path: {output_path}")
        if output_path.exists():
            print(f"Removing existing dataset at {output_path}")
            shutil.rmtree(output_path)
        # Also clear TEMP_IMAGE_DIR to prevent stale hardlinks from a previous run
        if TEMP_IMAGE_DIR is not None:
            temp_images = TEMP_IMAGE_DIR / "images"
            if temp_images.exists():
                shutil.rmtree(temp_images)
                print(f"Cleared stale temp images: {temp_images}")
        # Get image shapes from first planned episode
        first_src_session = planned[0][2]  # src_session_dir
        img_shape_endo, img_shape_wrist = self._get_image_shapes([first_src_session])
        print(f"Endoscope image shape: {img_shape_endo}")
        print(f"Wrist camera image shape: {img_shape_wrist}")
        print("Creating LeRobot dataset...")
        dataset = self._create_dataset(LeRobotDataset, img_shape_endo, img_shape_wrist)

        start_time = time.time()
        successful_episodes = 0
        perfect_count = 0
        recovery_count = 0
        episode_times = []  # Track per-episode durations for ETA

        for i, (ann, ref, src, dst, start, end) in enumerate(planned):
            if self.cancelled:
                print("Conversion cancelled by user")
                sys.exit(1)

            # Derive subtask text from destination path structure
            # dst = out_dir / tissue_N / subtask_dir / episode_NNN
            subtask_dir_name = dst.parent.name          # e.g. "1_grasp"
            subtask_text = " ".join(subtask_dir_name.split("_")[1:])  # "grasp"
            is_recovery = "recovery" in subtask_text.lower()
            if is_recovery:
                subtask_text = subtask_text.replace(" recovery", "").replace("recovery", "").strip()

            print(f"\n[{i+1}/{len(planned)}] Episode: {dst.parent.name}/{dst.name}")

            try:
                t_ep_start = time.time()
                self._process_planned_episode(dataset, ref, src, start, end, subtask_text)
                t_proc_elapsed = time.time() - t_ep_start
                print(
                    f"    → Frame processing complete ({t_proc_elapsed:.1f}s). "
                    f"Encoding videos ({VIDEO_CODEC}, parallel)..."
                )
                t_enc = time.time()
                dataset.save_episode()
                t_enc_elapsed = time.time() - t_enc
                t_ep_total = time.time() - t_ep_start
                successful_episodes += 1
                episode_times.append(t_ep_total)
                if is_recovery:
                    recovery_count += 1
                else:
                    perfect_count += 1
                print(
                    f"✓ {dst.parent.name}/{dst.name} saved — "
                    f"processing: {t_proc_elapsed:.1f}s, encoding: {t_enc_elapsed:.1f}s, "
                    f"total: {t_ep_total:.1f}s  [task: \"{subtask_text}\"]"
                )

                # Compute and print ETA
                elapsed = time.time() - start_time
                completed_count = i + 1
                remaining_count = len(planned) - (i + 1)
                if completed_count > 0 and remaining_count > 0:
                    # Use recent episodes (last 5) for better estimate
                    recent = episode_times[-5:]
                    avg_time = sum(recent) / len(recent)
                    eta_secs = avg_time * remaining_count
                    elapsed_str = self._format_duration(elapsed)
                    eta_str = self._format_duration(eta_secs)
                    print(f"Elapsed: {elapsed_str}  |  ETA: ~{eta_str} remaining")
                elif remaining_count == 0:
                    elapsed_str = self._format_duration(elapsed)
                    print(f"Elapsed: {elapsed_str}  |  Done!")
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                print(f"✗ Error processing {dst.name}:\n  {error_details}", file=sys.stderr)
                try:
                    dataset.clear_episode_buffer()
                except:
                    pass

        # Final progress
        elapsed_str = self._format_duration(time.time() - start_time)
        print(f"Elapsed: {elapsed_str}  |  Complete")

        # Write train/val/test + perfect/recovery splits
        total_episodes = successful_episodes
        if total_episodes > 0:
            train_count = int(0.8 * total_episodes)
            val_count = int(0.1 * total_episodes)
            dataset.meta.info["splits"] = {
                "train": f"0:{train_count}",
                "val": f"{train_count}:{train_count + val_count}",
                "test": f"{train_count + val_count}:{total_episodes}",
                "perfect": f"0:{perfect_count}",
                "recovery": f"{perfect_count}:{perfect_count + recovery_count}",
            }
            write_info(dataset.meta.info, dataset.root)
            print(
                f"Split configuration saved — "
                f"train: {train_count}, val: {val_count}, "
                f"test: {total_episodes - train_count - val_count} | "
                f"perfect: {perfect_count}, recovery: {recovery_count}"
            )

        # Clean up intermediate images
        print("Cleaning up intermediate images...")
        self._cleanup_all_images(dataset)

        elapsed = time.time() - start_time
        print(f"\n{'='*50}")
        print("Conversion complete!")
        print(f"Total episodes: {successful_episodes}/{len(planned)}")
        print(f"Time elapsed: {elapsed:.1f} seconds")
        print(f"Dataset saved to: {output_path}")

    def _process_planned_episode(self, dataset, ref_session_dir: Path,
                                   src_session_dir: Path, start_idx: int,
                                   end_idx: int, subtask_text: str):
        """Process a single affordance-sliced episode from pre-filtered data.

        Reads frames directly from the *filtered session* directory,
        using timestamp alignment to map reference (cautery) frame indices
        to source (filtered) frame indices.

        Since the filtered data is already multi-camera synced by
        ``run_filter_episodes``, we only need to locate the affordance
        slice window — no additional sync step is needed.
        """
        t_start = time.time()

        # ------------------------------------------------------------------
        # 1. Timestamp alignment: reference indices → source indices
        # ------------------------------------------------------------------
        ref_left_dir = ref_session_dir / LEFT_IMG_DIR
        ref_files = list_sorted_frames(ref_left_dir, "_left.jpg")
        if not ref_files:
            raise ValueError(f"Reference directory empty: {ref_left_dir}")

        s_clamped = max(0, min(start_idx, len(ref_files) - 1))
        e_clamped = max(0, min(end_idx, len(ref_files) - 1))
        ts_start = sa_extract_timestamp(ref_files[s_clamped])
        ts_end   = sa_extract_timestamp(ref_files[e_clamped])

        src_left_dir = src_session_dir / LEFT_IMG_DIR
        src_files = list_sorted_frames(src_left_dir, "_left.jpg")
        if not src_files:
            raise ValueError(f"Source directory empty: {src_left_dir}")

        src_timestamps = np.array(
            [sa_extract_timestamp(f) for f in src_files], dtype=np.int64
        )
        new_start = int(np.searchsorted(src_timestamps, ts_start, side="left"))
        new_end   = int(np.searchsorted(src_timestamps, ts_end,   side="right")) - 1
        new_start = max(0, min(new_start, len(src_files) - 1))
        new_end   = max(0, min(new_end,   len(src_files) - 1))

        total_frames = new_end - new_start + 1
        if total_frames <= 0:
            raise ValueError(
                f"Empty slice after alignment: ref {start_idx}-{end_idx} "
                f"→ src {new_start}-{new_end}"
            )

        print(
            f"  Aligned ref[{start_idx}:{end_idx}] → src[{new_start}:{new_end}] "
            f"({total_frames} frames)"
        )
        t_align = time.time()

        # ------------------------------------------------------------------
        # 2. Build path lists for ALL cameras in the filtered session
        # ------------------------------------------------------------------
        left_paths = [src_left_dir / f for f in src_files]

        right_files = list_sorted_frames(src_session_dir / RIGHT_IMG_DIR, "_right.jpg")
        right_paths = (
            [src_session_dir / RIGHT_IMG_DIR / f for f in right_files]
            if right_files else None
        )

        psm1_files = list_sorted_frames(src_session_dir / ENDO_PSM1_DIR, "_psm1.jpg")
        psm1_paths = (
            [src_session_dir / ENDO_PSM1_DIR / f for f in psm1_files]
            if psm1_files else None
        )

        psm2_files = list_sorted_frames(src_session_dir / ENDO_PSM2_DIR, "_psm2.jpg")
        psm2_paths = (
            [src_session_dir / ENDO_PSM2_DIR / f for f in psm2_files]
            if psm2_files else None
        )

        # ------------------------------------------------------------------
        # 3. Load CSV & pre-compute state/action matrices (sliced)
        # ------------------------------------------------------------------
        # CSV lives in the original raw data dir, not in the filtered cache
        csv_path = ref_session_dir / CSV_FILE
        df = pd.read_csv(csv_path)
        state_cols  = [c for c in STATES_NAME  if c in df.columns]
        action_cols = [c for c in ACTIONS_NAME if c in df.columns]
        states_matrix  = df[state_cols].values.astype(np.float32)
        actions_matrix = df[action_cols].values.astype(np.float32)

        # Pre-compute hybrid relative actions for the entire CSV
        hybrid_rel_matrix = np.zeros_like(actions_matrix)
        for t in range(len(actions_matrix) - 1):
            hybrid_rel_matrix[t] = compute_action_hybrid_rel(
                actions_matrix[t], actions_matrix[t + 1]
            )
        # Last frame: zero-delta (identity rotation, zero translation, same jaw)
        hybrid_rel_matrix[-1] = compute_action_hybrid_rel(
            actions_matrix[-1], actions_matrix[-1]
        )

        t_prep = time.time()
        print(
            f"  Processing {total_frames} frames "
            f"(align: {t_align - t_start:.1f}s, prep: {t_prep - t_align:.1f}s)..."
        )

        # ------------------------------------------------------------------
        # 4. Pre-place source JPEGs & build frames with dummy images
        #    (Eliminates JPEG decode → re-encode → re-decode round-trip)
        # ------------------------------------------------------------------
        from lerobot.datasets.utils import DEFAULT_IMAGE_PATH

        fps_inv = 1.0 / self.fps
        t_add_total = 0.0

        # Determine the episode_index LeRobot will use for file paths
        ep_idx_for_paths = dataset.meta.total_episodes
        _temp_base = TEMP_IMAGE_DIR if TEMP_IMAGE_DIR is not None else dataset.root

        # Map image keys to their source path lists
        camera_path_map = {
            "observation.images.endoscope.left":  left_paths,
            "observation.images.endoscope.right": right_paths,
            "observation.images.wrist.right":     psm1_paths,
            "observation.images.wrist.left":      psm2_paths,
        }

        # Get image shapes from dataset features for dummy arrays
        _dummy_cache = {}  # image_key -> dummy numpy array
        for img_key in camera_path_map:
            feat = dataset.features[img_key]
            shape = tuple(feat["shape"])  # (c, h, w) or (h, w, c)
            _dummy_cache[img_key] = np.zeros(shape, dtype=np.uint8)

        # Pre-place source JPEGs as hardlinks at the paths LeRobot expects.
        # This lets ffmpeg read the original files directly during encoding,
        # completely skipping the decode→numpy→encode→write cycle.
        t_preplace_start = time.time()
        preplace_count = 0
        preplace_fallback_count = 0
        import shutil as _shutil

        for img_key, paths in camera_path_map.items():
            if not paths:
                continue
            for out_idx in range(total_frames):
                src_idx = new_start + out_idx
                if src_idx >= len(paths):
                    continue
                source_jpg = paths[src_idx]
                target_rel = DEFAULT_IMAGE_PATH.format(
                    image_key=img_key,
                    episode_index=ep_idx_for_paths,
                    frame_index=out_idx,
                )
                target = (_temp_base / target_rel).with_suffix(".jpg")
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    try:
                        os.link(str(source_jpg), str(target))
                    except (OSError, NotImplementedError):
                        _shutil.copy2(str(source_jpg), str(target))
                        preplace_fallback_count += 1
                    preplace_count += 1

        t_preplace = time.time()
        print(
            f"  Pre-placed {preplace_count} source JPEGs as hardlinks "
            f"({t_preplace - t_preplace_start:.1f}s)"
            + (f" ({preplace_fallback_count} fell back to copy)" if preplace_fallback_count else "")
        )

        # Build and add frames using lightweight dummy image arrays.
        # The dummy arrays pass LeRobot's shape validation, and _write_image_fast
        # skips writing because the file already exists on disk.
        for frame_idx in range(total_frames):
            if frame_idx > 0 and frame_idx % 500 == 0:
                avg_add = (t_add_total / frame_idx) * 1000
                print(
                    f"    -> {frame_idx}/{total_frames} "
                    f"(avg add_frame={avg_add:.1f}ms)"
                )

            src_idx = new_start + frame_idx
            frame = {
                "observation.state": states_matrix[src_idx],
                "action":            actions_matrix[src_idx],
                "action_hybrid_relative": hybrid_rel_matrix[src_idx],
                "instruction.text":  subtask_text,
                "observation.meta.tool.psm1": self.psm1_tool,
                "observation.meta.tool.psm2": self.psm2_tool,
            }
            # Add dummy images for each available camera
            for img_key, paths in camera_path_map.items():
                if paths and src_idx < len(paths):
                    frame[img_key] = _dummy_cache[img_key]

            t0 = time.time()
            dataset.add_frame(frame, task=subtask_text, timestamp=frame_idx * fps_inv)
            t_add_total += time.time() - t0

        t_end = time.time()
        print(
            f"    -> {total_frames}/{total_frames} frames (100%) "
            f"| total: {t_end - t_start:.1f}s "
            f"(align: {t_align - t_start:.1f}s, "
            f"prep: {t_prep - t_align:.1f}s, "
            f"pre-place: {t_preplace - t_preplace_start:.1f}s, "
            f"add_frame: {t_add_total:.1f}s)"
        )

    
    def _cleanup_all_images(self, dataset):
        """Remove ALL intermediate images after all video encoding is complete"""
        for images_dir in [dataset.root / "images",
                           TEMP_IMAGE_DIR / "images" if TEMP_IMAGE_DIR else None]:
            try:
                if images_dir and images_dir.exists():
                    shutil.rmtree(images_dir)
                    print(f"  Cleaned up: {images_dir}")
            except Exception as e:
                print(f"  Warning: Could not clean up {images_dir}: {str(e)}", file=sys.stderr)

# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Accelerated DVRK to LeRobot v2.1 Converter (CLI)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--source-dir", type=Path, required=True,
        help="Path to the raw DVRK data directory."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(DEFAULT_LEROBOT_HOME),
        help="Path to the output directory where the dataset will be saved."
    )
    parser.add_argument(
        "--dataset-name", type=str, required=True,
        help="Name of the output LeRobot dataset."
    )
    parser.add_argument(
        "--annotations-dir", type=Path, required=True,
        help="Path to the annotations directory (enables pipeline mode)."
    )
    parser.add_argument(
        "--psm1-tool", type=str, default="Permanent Cautery Hook",
        help="Name of the PSM1 tool."
    )
    parser.add_argument(
        "--psm2-tool", type=str, default="Prograsp Forceps",
        help="Name of the PSM2 tool."
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Frames per second for the output dataset."
    )
    parser.add_argument(
        "--codec", type=str, choices=CODEC_OPTIONS, default="h264",
        help="Video codec to use for encoding."
    )
    parser.add_argument(
        "--clear-cache", action="store_true",
        help="Clear the filtered cache directory before running."
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite the output dataset if it already exists."
    )

    parser.add_argument(
        "--trim-threshold", type=float, default=1e-4,
        help=(
            "Joint-space L2 delta threshold for trimming stationary frames "
            "from episode start/end. Set to 0 to disable trimming."
        ),
    )

    args = parser.parse_args()

    # Handle clear cache
    cache_dir = args.output_dir / "_filtered_cache"
    if args.clear_cache:
        if cache_dir.exists():
            print(f"Clearing cache directory: {cache_dir}")
            shutil.rmtree(cache_dir)
        else:
            print(f"Cache directory does not exist, nothing to clear: {cache_dir}")

    # Check for existing dataset if not overwriting
    dataset_path = args.output_dir / args.dataset_name
    if dataset_path.exists() and not args.overwrite:
        print(
            f"Error: Dataset already exists at {dataset_path}. "
            "Use --overwrite to replace it.",
            file=sys.stderr
        )
        sys.exit(1)

    # Set the global video codec
    global VIDEO_CODEC
    VIDEO_CODEC = args.codec

    print("=" * 60)
    print("DVRK to LeRobot Converter Configuration:")
    print(f"  Source Dir:      {args.source_dir}")
    print(f"  Annotations Dir: {args.annotations_dir}")
    print(f"  Output Dir:      {args.output_dir}")
    print(f"  Dataset Name:    {args.dataset_name}")
    print(f"  FPS:             {args.fps}")
    print(f"  Codec:           {VIDEO_CODEC}")
    print("=" * 60)

    # Validate source directory
    if not args.source_dir.exists() or not args.source_dir.is_dir():
        print(f"Error: Source directory does not exist: {args.source_dir}", file=sys.stderr)
        sys.exit(1)
        
    if args.annotations_dir and (not args.annotations_dir.exists() or not args.annotations_dir.is_dir()):
        print(f"Error: Annotations directory does not exist: {args.annotations_dir}", file=sys.stderr)
        sys.exit(1)

    worker = ConversionWorker(
        source_path=args.source_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        psm1_tool=args.psm1_tool,
        psm2_tool=args.psm2_tool,
        fps=args.fps,
        annotations_dir=args.annotations_dir,
        trim_threshold=args.trim_threshold,
    )

    # Handle SIGINT (Ctrl+C) gracefully
    signal.signal(signal.SIGINT, worker.cancel)

    worker.run()


if __name__ == "__main__":
    main()
