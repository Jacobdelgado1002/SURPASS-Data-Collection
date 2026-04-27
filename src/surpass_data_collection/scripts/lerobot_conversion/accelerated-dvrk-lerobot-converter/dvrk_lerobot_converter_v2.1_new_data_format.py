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
from typing import List, Optional, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
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
from slice_affordance import list_sorted_frames, extract_timestamp as sa_extract_timestamp
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
class DirectEpisodeInfo:
    """Metadata for a single episode discovered in the direct structure."""
    source_dir: Path
    tissue_label: str  
    tissue_num: int  
    collector: str  
    phase: str  
    action: str  


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
_FRAME_TS_RE = re.compile(r'frame(\d+)')
_NEW_FORMAT_RE = re.compile(r'frame\d+_(?:left|right|psm1|psm2)_(\d+)_(\d+)\.jpg')


def extract_timestamp(filename: str) -> int:
    """Extract nanosecond timestamp from frame filename.

    Supports both old ('frame{ts}_{cam}.jpg') and new
    ('frame{seq}_{cam}_{sec}_{nsec}.jpg') formats.
    """
    # Try new format first (more specific)
    new_match = _NEW_FORMAT_RE.search(filename)
    if new_match:
        return int(new_match.group(1)) * 1_000_000_000 + int(new_match.group(2))

    # Fall back to old format
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
    
    # Check all required/expected camera directories
    for cam_dir_name in [LEFT_IMG_DIR, RIGHT_IMG_DIR, ENDO_PSM1_DIR, ENDO_PSM2_DIR]:
        cam_dir = episode_path / cam_dir_name
        if not cam_dir.exists():
            errors.append(f"Missing directory: {cam_dir_name}")
        elif not list(cam_dir.glob("*.jpg")):
            errors.append(f"No images in: {cam_dir_name}")
    
    csv_path = episode_path / CSV_FILE
    if not csv_path.exists():
        errors.append(f"Missing CSV: {CSV_FILE}")
    
    if errors:
        return False, "; ".join(errors)
    return True, "Valid"

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
        print(f"Failed to copy {src} → {dst}: {exc}", file=sys.stderr)
        return False

def _discover_direct_episodes(root_dir: Path) -> List[DirectEpisodeInfo]:
    """Walk *root_dir* and discover all direct-mode episodes.

    A valid episode is a directory containing ``ee_csv.csv`` and at least
    one camera subdirectory (``left_img_dir``).

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
    return episodes


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
    
    def __init__(self, source_data_path: Path, output_dir: Path, dataset_name: str,
                 psm1_tool: str, psm2_tool: str, fps: int,
                 trim_threshold: float = 1e-4, use_hardlink: bool = True, no_trim: bool = False):
        self.source_data_path = source_data_path
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.psm1_tool = psm1_tool
        self.psm2_tool = psm2_tool
        self.fps = fps
        self.trim_threshold = trim_threshold
        self.use_hardlink = use_hardlink
        self.no_trim = no_trim
        self.cancelled = False
    
    def cancel(self, signum=None, frame=None):
        print("\n[!] Cancellation requested by user. Terminating soon...")
        self.cancelled = True
    
    def run(self):
        try:
            self._run_conversion()
        except Exception as e:
            print(f"[ERROR] {str(e)}\n{traceback.format_exc()}", file=sys.stderr)
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

    # Image shape should alwyas be the same for all episodes. That is why there are default values.
    # If you are getting errors about image shapes, it is likely because the default values are incorrect.
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
                
                if left_images and endo_images:
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
    # Pipeline mode: filter -> plan -> (trim) -> convert
    # -----------------------------------------------------------------

    def _trim_stationary_episodes_direct(self, episodes: List[DirectEpisodeInfo]) -> Tuple[List[Tuple[DirectEpisodeInfo, int, int]], Dict[str, int]]:
        """Compute stationary-frame trim ranges for each direct episode."""
        print(f"Trimming stationary frames (threshold={self.trim_threshold:.1e}) on {len(episodes)} episodes...")
        
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

            trim_start, trim_end = find_trim_range(deltas, n_frames, self.trim_threshold)

            if trim_end - trim_start + 1 < 10:
                # Trimmed too short — keep original
                results.append((ep, 0, n_frames - 1))
                continue

            if trim_start > 0 or trim_end < n_frames - 1:
                removed = n_frames - (trim_end - trim_start + 1)
                stats["trimmed"] += 1
                stats["frames_removed"] += removed

            results.append((ep, trim_start, trim_end))

        print(f"  Trimmed {stats['trimmed']}/{stats['total']} episodes, removed {stats['frames_removed']} total frames")
        return results, stats

    def _process_direct_episode(self, episode_info: DirectEpisodeInfo, trim_start: int, trim_end: int, output_episode_dir: Path, episode_id: str) -> EpisodeResult:
        result = EpisodeResult(
            episode_id=episode_id,
            tissue=episode_info.tissue_label,
            action=episode_info.action,
            output_path=str(output_episode_dir),
            source_session=str(episode_info.source_dir),
        )
        t_start = time.time()
        try:
            is_valid, err_msg = validate_episode(episode_info.source_dir)
            if not is_valid:
                result.error = f"Validation failed: {err_msg}"
                result.skipped = True
                return result

            camera_files: Dict[str, List[str]] = {}
            for cam_dir, cam_suffix in CAMERA_SUFFIXES.items():
                cam_files = list_sorted_frames(episode_info.source_dir / cam_dir, cam_suffix)
                if not cam_files:
                    result.error = f"Missing {cam_dir} frames in {episode_info.source_dir}"
                    result.skipped = True
                    return result
                camera_files[cam_dir] = cam_files

            min_cam_count = min(len(files) for files in camera_files.values())
            actual_end = min(trim_end, min_cam_count - 1)
            total_frames = actual_end - trim_start + 1

            if total_frames <= 0:
                result.error = "No frames after trimming/clamping"
                result.skipped = True
                return result

            result.frame_range = (trim_start, actual_end)

            frames_copied = 0
            for cam_dir, cam_suffix_name in CAMERA_MODALITIES.items():
                cam_out_dir = output_episode_dir / cam_dir
                cam_out_dir.mkdir(parents=True, exist_ok=True)
                cam_files_list = camera_files[cam_dir]

                for out_idx in range(total_frames):
                    src_idx = trim_start + out_idx
                    if src_idx >= len(cam_files_list):
                        continue

                    src_file = episode_info.source_dir / cam_dir / cam_files_list[src_idx]
                    ext = src_file.suffix.lower()
                    dst_file = cam_out_dir / f"frame{out_idx:06d}{cam_suffix_name}{ext}"

                    if not src_file.exists():
                        continue

                    if _copy_or_hardlink(src_file, dst_file, self.use_hardlink):
                        frames_copied += 1

            result.num_frames = frames_copied // max(len(CAMERA_MODALITIES), 1)

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

            dt = 1.0 / self.fps
            timestamp_col = sliced_df.columns[0]
            sliced_df[timestamp_col] = [f"{i * dt:.4f}" for i in range(len(sliced_df))]

            csv_dst.parent.mkdir(parents=True, exist_ok=True)
            sliced_df.to_csv(csv_dst, index=False)
            result.csv_rows = len(sliced_df)

            result.success = True
            result.duration_s = time.time() - t_start

        except Exception as exc:
            result.error = str(exc)
            result.duration_s = time.time() - t_start
            print(f"Error processing episode {episode_id}: {exc}", file=sys.stderr)

        return result

    def _restructure_direct(self, episodes_with_trims: List[Tuple[DirectEpisodeInfo, int, int]], output_base: Path):
        """Restructure direct-mode episodes into the canonical output format."""
        print(f"Restructuring {len(episodes_with_trims)} episodes...")
        episode_counters: Dict[Tuple[int, str, str, str], int] = defaultdict(int)

        results = []
        for i, (ep, trim_start, trim_end) in enumerate(episodes_with_trims):
            if self.cancelled:
                print("Conversion cancelled by user")
                sys.exit(1)

            counter_key = (ep.tissue_num, ep.collector, ep.phase, ep.action)
            episode_counters[counter_key] += 1
            episode_index = episode_counters[counter_key]

            tissue_label = f"tissue_{ep.tissue_num}"
            path_parts: List[str] = [tissue_label]
            if ep.collector:
                path_parts.append(ep.collector)
            if ep.phase:
                path_parts.append(ep.phase)
            path_parts.append(ep.action)
            path_parts.append(f"episode_{episode_index:03d}")

            output_episode_dir = output_base
            for part in path_parts:
                output_episode_dir = output_episode_dir / part

            episode_id = str(output_episode_dir.relative_to(output_base))

            print(f"\n[{i+1}/{len(episodes_with_trims)}] Restructuring: {episode_id} (frames {trim_start}–{trim_end})")

            result = self._process_direct_episode(
                episode_info=ep,
                trim_start=trim_start,
                trim_end=trim_end,
                output_episode_dir=output_episode_dir,
                episode_id=episode_id,
            )
            results.append(result)

            if result.success:
                print(f"    [SUCCESS] {result.num_frames} frames, {result.csv_rows} CSV rows ({result.duration_s:.1f}s)")
            elif result.skipped:
                print(f"    [SKIPPED] {result.error}")
            else:
                print(f"    [ERROR] {result.error}", file=sys.stderr)

        return results

    def _convert_restructured_to_lerobot(self, dataset, restructured_dir: Path):
        """Walk the restructured output and ingest into LeRobot format."""
        episodes_paths = []
        for csv_file in restructured_dir.rglob(CSV_FILE):
            if (csv_file.parent / LEFT_IMG_DIR).is_dir():
                episodes_paths.append(csv_file.parent)
        
        episodes_paths.sort()

        successful_episodes = 0
        perfect_count = 0
        recovery_count = 0
        episode_times = []
        start_time = time.time()

        for i, ep_dir in enumerate(episodes_paths):
            if self.cancelled:
                print("Conversion cancelled by user")
                sys.exit(1)
            
            action_name = ep_dir.parent.name
            subtask_text = action_name.replace("_", " ")
            parts = subtask_text.split(" ", 1)
            if len(parts) == 2 and parts[0].isdigit():
                subtask_text = parts[1]
                
            is_recovery = "recovery" in subtask_text.lower()
            if is_recovery:
                subtask_text = subtask_text.replace(" recovery", "").replace("recovery", "").strip()

            print(f"\n[{i+1}/{len(episodes_paths)}] Converting to LeRobot: {ep_dir.relative_to(restructured_dir)}")

            try:
                t_ep_start = time.time()
                self._ingest_single_episode(dataset, ep_dir, subtask_text)
                t_proc_elapsed = time.time() - t_ep_start
                print(f"    -> Frame processing complete ({t_proc_elapsed:.1f}s). Encoding videos ({VIDEO_CODEC}, parallel)...")
                
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
                print(f"[SUCCESS] Saved — processing: {t_proc_elapsed:.1f}s, encoding: {t_enc_elapsed:.1f}s, total: {t_ep_total:.1f}s [task: \"{subtask_text}\"]")

                elapsed = time.time() - start_time
                completed_count = i + 1
                remaining_count = len(episodes_paths) - (i + 1)
                if completed_count > 0 and remaining_count > 0:
                    recent = episode_times[-5:]
                    avg_time = sum(recent) / len(recent)
                    eta_secs = avg_time * remaining_count
                    print(f"Elapsed: {self._format_duration(elapsed)}  |  ETA: ~{self._format_duration(eta_secs)} remaining")

            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                print(f"✗ Error processing {ep_dir.name}:\n  {error_details}", file=sys.stderr)
                try:
                    dataset.clear_episode_buffer()
                except:
                    pass

        return successful_episodes, perfect_count, recovery_count

    def _ingest_single_episode(self, dataset, ep_dir: Path, subtask_text: str):
        """Read an already restructured episode and ingest frames into LeRobot."""
        t_start = time.time()
        
        left_files = list_sorted_frames(ep_dir / LEFT_IMG_DIR, CAMERA_SUFFIXES[LEFT_IMG_DIR])
        right_files = list_sorted_frames(ep_dir / RIGHT_IMG_DIR, CAMERA_SUFFIXES[RIGHT_IMG_DIR])
        psm1_files = list_sorted_frames(ep_dir / ENDO_PSM1_DIR, CAMERA_SUFFIXES[ENDO_PSM1_DIR])
        psm2_files = list_sorted_frames(ep_dir / ENDO_PSM2_DIR, CAMERA_SUFFIXES[ENDO_PSM2_DIR])

        left_paths = [ep_dir / LEFT_IMG_DIR / f for f in left_files]
        right_paths = [ep_dir / RIGHT_IMG_DIR / f for f in right_files]
        psm1_paths = [ep_dir / ENDO_PSM1_DIR / f for f in psm1_files]
        psm2_paths = [ep_dir / ENDO_PSM2_DIR / f for f in psm2_files]

        csv_path = ep_dir / CSV_FILE
        df = pd.read_csv(csv_path)
        state_cols  = [c for c in STATES_NAME  if c in df.columns]
        action_cols = [c for c in ACTIONS_NAME if c in df.columns]
        states_matrix  = df[state_cols].values.astype(np.float32)
        actions_matrix = df[action_cols].values.astype(np.float32)

        hybrid_rel_matrix = np.zeros_like(actions_matrix)
        for t in range(len(actions_matrix) - 1):
            hybrid_rel_matrix[t] = compute_action_hybrid_rel(actions_matrix[t], actions_matrix[t + 1])
        hybrid_rel_matrix[-1] = compute_action_hybrid_rel(actions_matrix[-1], actions_matrix[-1])

        total_frames = len(left_files)
        fps_inv = 1.0 / self.fps
        t_add_total = 0.0

        ep_idx_for_paths = dataset.meta.total_episodes
        _temp_base = TEMP_IMAGE_DIR if TEMP_IMAGE_DIR is not None else dataset.root

        camera_path_map = {
            "observation.images.endoscope.left":  left_paths,
            "observation.images.endoscope.right": right_paths,
            "observation.images.wrist.right":     psm1_paths,
            "observation.images.wrist.left":      psm2_paths,
        }

        from lerobot.datasets.utils import DEFAULT_IMAGE_PATH
        _dummy_cache = {}
        for img_key in camera_path_map:
            feat = dataset.features[img_key]
            shape = tuple(feat["shape"])
            _dummy_cache[img_key] = np.zeros(shape, dtype=np.uint8)

        t_preplace_start = time.time()
        preplace_count = 0
        preplace_fallback_count = 0
        import shutil as _shutil

        for img_key, paths in camera_path_map.items():
            if not paths:
                continue
            for out_idx in range(total_frames):
                if out_idx >= len(paths):
                    continue
                source_jpg = paths[out_idx]
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
        print(f"  Pre-placed {preplace_count} source JPEGs as hardlinks "
              f"({t_preplace - t_preplace_start:.1f}s)"
              + (f" ({preplace_fallback_count} fell back to copy)" if preplace_fallback_count else ""))

        for frame_idx in range(total_frames):
            if frame_idx > 0 and frame_idx % 500 == 0:
                avg_add = (t_add_total / frame_idx) * 1000
                print(f"    -> {frame_idx}/{total_frames} (avg add_frame={avg_add:.1f}ms)")

            frame = {
                "observation.state": states_matrix[frame_idx],
                "action":            actions_matrix[frame_idx],
                "action_hybrid_relative": hybrid_rel_matrix[frame_idx],
                "instruction.text":  subtask_text,
                "observation.meta.tool.psm1": self.psm1_tool,
                "observation.meta.tool.psm2": self.psm2_tool,
            }
            for img_key, paths in camera_path_map.items():
                if paths and frame_idx < len(paths):
                    frame[img_key] = _dummy_cache[img_key]

            t0 = time.time()
            dataset.add_frame(frame, task=subtask_text, timestamp=frame_idx * fps_inv)
            t_add_total += time.time() - t0

        t_end = time.time()
        print(f"    -> {total_frames}/{total_frames} frames (100%) | total: {t_end - t_start:.1f}s "
              f"(pre-place: {t_preplace - t_preplace_start:.1f}s, add_frame: {t_add_total:.1f}s)")

    def _run_pipeline_conversion(self, LeRobotDataset, write_info):
        """Full pipeline: Filter -> Discover -> Trim -> Restructure -> Convert to LeRobot."""
        filtered_dir = self.output_dir / "_filtered_cache"
        max_time_diff_ms = float(self.fps)

        print(f"Stage 1/5: Filtering & synchronising episodes (threshold={max_time_diff_ms:.1f}ms)...")
        rc = run_filter_episodes(
            source_dir=str(self.source_data_path),
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

        if self.cancelled: sys.exit(1)

        print("Stage 2/5: Discovering direct episodes...")
        episodes = _discover_direct_episodes(filtered_dir)
        if not episodes:
            print("No episodes found. Verify the input directory structure.", file=sys.stderr)
            sys.exit(1)
        
        if self.cancelled: sys.exit(1)

        print(f"Stage 3/5: Trimming stationary frames (threshold={self.trim_threshold:.1e})...")
        if not self.no_trim:
            episodes_with_trims, trim_stats = self._trim_stationary_episodes_direct(episodes)
        else:
            episodes_with_trims = []
            for ep in episodes:
                left_dir = ep.source_dir / LEFT_IMG_DIR
                frame_files = list_sorted_frames(left_dir, CAMERA_SUFFIXES[LEFT_IMG_DIR])
                episodes_with_trims.append((ep, 0, max(0, len(frame_files) - 1)))
        
        if self.cancelled: sys.exit(1)

        print("Stage 4/5: Restructuring to canonical format...")
        restructured_dir = self.output_dir / f"{self.dataset_name}_restructured"
        if restructured_dir.exists():
            print(f"Removing existing restructured directory at {restructured_dir}")
            shutil.rmtree(restructured_dir)
        
        self._restructure_direct(episodes_with_trims, restructured_dir)

        if self.cancelled: sys.exit(1)

        print("Stage 5/5: Converting to LeRobot format...")
        output_path = self.output_dir / self.dataset_name
        if output_path.exists():
            print(f"Removing existing dataset at {output_path}")
            shutil.rmtree(output_path)
            
        if TEMP_IMAGE_DIR is not None:
            temp_images = TEMP_IMAGE_DIR / "images"
            if temp_images.exists():
                shutil.rmtree(temp_images)

        # Get image shapes from first restructured episode
        first_csv_iter = restructured_dir.rglob(CSV_FILE)
        try:
            first_csv = next(first_csv_iter)
            img_shape_endo, img_shape_wrist = self._get_image_shapes([first_csv.parent])
        except StopIteration:
            print("Error: No CSV files found in the restructured directory.", file=sys.stderr)
            sys.exit(1)
            
        dataset = self._create_dataset(LeRobotDataset, img_shape_endo, img_shape_wrist)

        start_time = time.time()
        successful_episodes, perfect_count, recovery_count = self._convert_restructured_to_lerobot(dataset, restructured_dir)
        
        # Write train/val/test splits
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
            print(f"Split configuration saved — train: {train_count}, val: {val_count}, test: {total_episodes - train_count - val_count}")

        print("Cleaning up intermediate images...")
        self._cleanup_all_images(dataset)

        elapsed = time.time() - start_time
        print(f"\n{'='*50}")
        print("Conversion complete!")
        print(f"Total episodes: {successful_episodes}/{len(episodes_with_trims)}")
        print(f"Time elapsed: {elapsed:.1f} seconds")
        print(f"Dataset saved to: {output_path}")

    
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
        "--no-trim", action="store_true",
        help="Disable stationary frame trimming entirely."
    )
    parser.add_argument(
        "--no-hardlink", action="store_true",
        help="Disable hardlinks and force copying for all frames."
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
    print(f"  Output Dir:      {args.output_dir}")
    print(f"  Dataset Name:    {args.dataset_name}")
    print(f"  FPS:             {args.fps}")
    print(f"  Codec:           {VIDEO_CODEC}")
    print("=" * 60)

    # Validate source directory
    if not args.source_dir.exists() or not args.source_dir.is_dir():
        print(f"Error: Source directory does not exist: {args.source_dir}", file=sys.stderr)
        sys.exit(1)

    worker = ConversionWorker(
        source_data_path=args.source_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        psm1_tool=args.psm1_tool,
        psm2_tool=args.psm2_tool,
        fps=args.fps,
        trim_threshold=args.trim_threshold,
        use_hardlink=not args.no_hardlink,
        no_trim=args.no_trim,
    )

    # Handle SIGINT (Ctrl+C) gracefully
    signal.signal(signal.SIGINT, worker.cancel)

    worker.run()


if __name__ == "__main__":
    main()
