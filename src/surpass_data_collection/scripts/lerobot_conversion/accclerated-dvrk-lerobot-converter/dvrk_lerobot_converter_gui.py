#!/usr/bin/env python
"""
GUI-based converter for DVRK data to LeRobot v2.1 format.

This tool provides a graphical interface to convert DVRK surgical robot datasets
into LeRobot format with timestamp-based alignment across multiple camera views.

Based on dvrk_zarr_to_lerobot.py but with:
- Support for new DVRK data format (timestamp-based filenames)
- All episodes go to training set
- Simple GUI for ease of use
"""

import sys
import os
import re
import glob as glob_module
import logging
import shutil
import time
import types
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import traceback

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QLineEdit, QSpinBox, QMessageBox,
    QProgressBar, QGroupBox, QFileDialog, QFormLayout, QComboBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

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

# Default LeRobot home for UI display only
DEFAULT_LEROBOT_HOME = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "lerobot")

# =============================================================================
# CONFIGURATION
# =============================================================================
# Available video codecs (selectable in the GUI):
CODEC_OPTIONS = {
    "h264 (CPU — works everywhere)":       "h264",
    "hevc (CPU — works everywhere)":       "hevc",
    # "h264_nvenc (NVIDIA GPU — fastest)":    "h264_nvenc",
    # "h264_amf (AMD GPU)":                   "h264_amf",
    # "h264_qsv (Intel Quick Sync)":          "h264_qsv",
    "libsvtav1 (CPU — best compression, VERY SLOW)": "libsvtav1",
}
# DEFAULT_CODEC_LABEL = "h264_nvenc (NVIDIA GPU — fastest)"
DEFAULT_CODEC_LABEL = "h264 (CPU — works everywhere)"

# This global is updated from the GUI before conversion starts
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


def find_closest_frame(target_ts: int, frames: List[FrameInfo]) -> Optional[FrameInfo]:
    """Find the frame with timestamp closest to target (LEGACY — kept for compatibility)"""
    if not frames:
        return None
    return min(frames, key=lambda x: abs(x.timestamp - target_ts))


def find_closest_index_fast(target_ts: int, timestamps: np.ndarray) -> int:
    """Binary-search O(log n) lookup for closest timestamp index."""
    idx = np.searchsorted(timestamps, target_ts)
    if idx == 0:
        return 0
    if idx == len(timestamps):
        return len(timestamps) - 1
    if abs(timestamps[idx] - target_ts) < abs(timestamps[idx - 1] - target_ts):
        return idx
    return idx - 1


def find_closest_csv_index(target_ts: int, timestamps: np.ndarray) -> int:
    """Find the CSV row index with timestamp closest to target using binary search"""
    idx = np.searchsorted(timestamps, target_ts)
    if idx == 0:
        return 0
    if idx == len(timestamps):
        return len(timestamps) - 1
    if abs(timestamps[idx] - target_ts) < abs(timestamps[idx - 1] - target_ts):
        return idx
    return idx - 1


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
        glob_module.glob(str(imgs_dir / f"frame-{digits}.jpg")),
        key=lambda x: int(Path(x).stem.split("-")[-1]),
    )
    if not input_list:
        input_list = sorted(
            glob_module.glob(str(imgs_dir / f"frame-{digits}.png")),
            key=lambda x: int(Path(x).stem.split("-")[-1]),
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


def _patch_save_episode(dataset):
    """Monkey-patch LeRobotDataset.save_episode to use our custom ThreadPoolEncoder
    and properly resolve JPEGs in TEMP_IMAGE_DIR, bypassing the broken multiprocessing
    and PNG-only globbing in v0.4.3.
    """
    _original_save_episode = dataset.save_episode

    def _custom_encode_video(video_key, episode_index):
        import shutil
        import tempfile
        from lerobot.datasets.utils import DEFAULT_IMAGE_PATH
        
        temp_path = Path(tempfile.mkdtemp(dir=dataset.root)) / f"{video_key}_{episode_index:03d}.mp4"
        
        base = TEMP_IMAGE_DIR if TEMP_IMAGE_DIR is not None else dataset.root
        fpath = DEFAULT_IMAGE_PATH.format(image_key=video_key, episode_index=episode_index, frame_index=0)
        img_dir = (base / fpath).parent
        
        _encode_video_frames_custom(
            img_dir, temp_path, dataset.fps, vcodec=VIDEO_CODEC
        )
            
        return temp_path

    def _mock_encode_temp(self, video_key, episode_index):
        return self._precomputed_temp_paths[video_key]

    def _save_episode_wrapper(self, episode_data=None, parallel_encoding=True):
        episode_buffer = episode_data if episode_data is not None else self.episode_buffer
        episode_index = episode_buffer["episode_index"]
        if isinstance(episode_index, list) or isinstance(episode_index, np.ndarray):
            episode_index = int(np.asarray(episode_index).flatten()[0])
        elif isinstance(episode_index, int):
            episode_index = episode_index

        has_video_keys = len(self.meta.video_keys) > 0
        use_batched_encoding = getattr(self, "batch_encoding_size", 1) > 1

        if has_video_keys and not use_batched_encoding:
            num_cameras = len(self.meta.video_keys)
            
            with ThreadPoolExecutor(max_workers=max(1, num_cameras)) as pool:
                futures = {
                    pool.submit(_custom_encode_video, key, episode_index): key 
                    for key in self.meta.video_keys
                }
                results = {}
                for f in futures:
                    results[futures[f]] = f.result()
            
            self._precomputed_temp_paths = results
            
            _orig_temp = self._encode_temporary_episode_video
            self._encode_temporary_episode_video = types.MethodType(_mock_encode_temp, self)
            
            try:
                _original_save_episode(episode_data=episode_data, parallel_encoding=False)
            finally:
                self._encode_temporary_episode_video = _orig_temp
                if hasattr(self, "_precomputed_temp_paths"):
                    del self._precomputed_temp_paths
        else:
            _original_save_episode(episode_data=episode_data, parallel_encoding=parallel_encoding)

    dataset.save_episode = types.MethodType(_save_episode_wrapper, dataset)


def _save_image_jpeg(self, image, fpath: Path, compress_level: int = 1) -> None:
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
        self.image_writer.save_image(image=image, fpath=fpath, compress_level=compress_level)


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

        def _write_image_fast(image, fpath: Path, compress_level: int = 1):
            fpath = Path(fpath)
            # HARDLINK OPTIMIZATION: skip writing if the file was pre-placed
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
                    _original_write_image(image, fpath, compress_level=compress_level)
            else:
                _original_write_image(image, fpath, compress_level=compress_level)

        iw.write_image = _write_image_fast


# =============================================================================
# CONVERSION WORKER THREAD
# =============================================================================
class ConversionWorker(QThread):
    """Worker thread for running the conversion without blocking the UI"""
    
    progress = pyqtSignal(int, int)  # current, total
    log_message = pyqtSignal(str)
    episode_started = pyqtSignal(str)
    eta_update = pyqtSignal(str)  # formatted ETA string
    finished_signal = pyqtSignal(bool, str)  # success, message
    
    def __init__(self, source_path: Path, output_dir: Path, dataset_name: str,
                 psm1_tool: str, psm2_tool: str, fps: int,
                 annotations_dir: Optional[Path] = None,
                 resume_mode: bool = False, skip_count: int = 0):
        super().__init__()
        self.source_path = source_path
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.psm1_tool = psm1_tool
        self.psm2_tool = psm2_tool
        self.fps = fps
        self.annotations_dir = annotations_dir
        self.resume_mode = resume_mode
        self.skip_count = skip_count
        self.cancelled = False
    
    def cancel(self):
        self.cancelled = True
    
    def run(self):
        try:
            self._run_conversion()
        except Exception as e:
            self.finished_signal.emit(False, f"Error: {str(e)}\n{traceback.format_exc()}")

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

        if self.annotations_dir:
            self._run_pipeline_conversion(LeRobotDataset, write_info)
        else:
            self._run_flat_conversion(LeRobotDataset, write_info)

    # -----------------------------------------------------------------
    # Dataset creation helpers (shared by pipeline and flat modes)
    # -----------------------------------------------------------------

    def _create_dataset(self, LeRobotDataset, img_shape_endo, img_shape_wrist, output_path: Path):
        """Create a fresh LeRobot dataset with the standard feature schema."""
        dataset = LeRobotDataset.create(
            repo_id=self.dataset_name,
            root=output_path,
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
            vcodec=VIDEO_CODEC,
        )
        _patch_save_episode(dataset)
        _patch_jpeg_image_saving(dataset)
        self.log_message.emit(f"  Image writer: 16 threads (JPEG)  |  Video encoding: parallel, codec={VIDEO_CODEC}")
        return dataset

    def _init_resume_dataset(self, LeRobotDataset, output_path):
        """Resume an existing dataset, cleaning up partial data."""
        self.log_message.emit(f"Resuming: {self.skip_count} episodes already completed")

        # Clean up stale intermediate images
        for stale_dir in [output_path / "images",
                          TEMP_IMAGE_DIR / "images" if TEMP_IMAGE_DIR else None]:
            if stale_dir and stale_dir.exists():
                self.log_message.emit(f"Cleaning up stale intermediate images in {stale_dir}...")
                shutil.rmtree(stale_dir)

        # Remove partial files for the first incomplete episode
        partial_ep = f"episode_{self.skip_count:06d}"
        for parquet in (output_path / "data").glob(f"chunk-*/{partial_ep}.parquet"):
            self.log_message.emit(f"Removing partial parquet: {parquet.name}")
            parquet.unlink()
        videos_dir = output_path / "videos"
        if videos_dir.exists():
            for video_key_dir in videos_dir.iterdir():
                partial_vid = video_key_dir / f"{partial_ep}.mp4"
                if partial_vid.exists():
                    self.log_message.emit(f"Removing partial video: {video_key_dir.name}/{partial_vid.name}")
                    partial_vid.unlink()

        self.log_message.emit("Loading existing dataset...")
        dataset = LeRobotDataset(repo_id=self.dataset_name, root=output_path)
        dataset.start_image_writer(num_processes=0, num_threads=16)
        _patch_save_episode(dataset)
        _patch_jpeg_image_saving(dataset)
        self.log_message.emit(f"  Image writer: 16 threads (JPEG)  |  Video encoding: parallel, codec={VIDEO_CODEC}")
        return dataset

    # -----------------------------------------------------------------
    # Pipeline mode: filter → plan → convert
    # -----------------------------------------------------------------

    def _run_pipeline_conversion(self, LeRobotDataset, write_info):
        """Full pipeline: run_filter_episodes → plan_episodes → accelerated convert."""

        # ---------------------------------------------------------------
        # Stage 1: Filter and synchronise episodes
        # ---------------------------------------------------------------
        filtered_dir = self.output_dir / "_filtered_cache"
        max_time_diff_ms = self.fps

        if filtered_dir.exists():
            self.log_message.emit("Stage 1/3: Using cached filtered data (delete _filtered_cache to re-run)")
        else:
            self.log_message.emit(
                f"Stage 1/3: Filtering & synchronising episodes "
                f"(threshold={max_time_diff_ms:.1f}ms)..."
            )
            rc = run_filter_episodes(
                source_dir=str(self.source_path),
                out_dir=str(filtered_dir),
                max_time_diff=max_time_diff_ms,
                min_images=10,
                dry_run=False,
                overwrite=False,
            )
            if rc != 0:
                self.finished_signal.emit(False, "Filtering stage failed. Check logs.")
                return
            self.log_message.emit("  Filtering complete.")

        if self.cancelled:
            self.finished_signal.emit(False, "Cancelled by user")
            return

        # ---------------------------------------------------------------
        # Stage 2: Plan affordance-based episode slices
        # ---------------------------------------------------------------
        self.log_message.emit("Stage 2/3: Planning affordance slices...")
        planned = plan_episodes(
            self.annotations_dir,
            self.source_path,             # raw data = cautery reference dir
            Path("_unused"),              # out_dir placeholder (we don't copy)
            source_dataset_dir=filtered_dir,
        )
        self.log_message.emit(f"  Planned {len(planned)} episodes")

        if not planned:
            self.finished_signal.emit(
                False,
                "No episodes planned. Check that annotations, cautery, and "
                "filtered data paths are correct."
            )
            return

        if self.cancelled:
            self.finished_signal.emit(False, "Cancelled by user")
            return

        # ---------------------------------------------------------------
        # Stage 3: Accelerated conversion to LeRobot
        # ---------------------------------------------------------------
        self.log_message.emit("Stage 3/3: Converting to LeRobot format...")
        output_path = self.output_dir / self.dataset_name
        self.log_message.emit(f"Output path: {output_path}")
        if self.resume_mode:
            dataset = self._init_resume_dataset(LeRobotDataset, output_path)
        else:
            if output_path.exists():
                self.log_message.emit(f"Removing existing dataset at {output_path}")
                shutil.rmtree(output_path)
            # Get image shapes from first planned episode
            first_src_session = planned[0][2]  # src_session_dir
            img_shape_endo, img_shape_wrist = self._get_image_shapes([first_src_session])
            self.log_message.emit(f"Endoscope image shape: {img_shape_endo}")
            self.log_message.emit(f"Wrist camera image shape: {img_shape_wrist}")
            self.log_message.emit("Creating LeRobot dataset...")
            dataset = self._create_dataset(LeRobotDataset, img_shape_endo, img_shape_wrist, output_path)

        start_time = time.time()
        successful_episodes = self.skip_count if self.resume_mode else 0
        perfect_count = 0
        recovery_count = 0
        valid_seen = 0
        episode_times = []  # Track per-episode durations for ETA

        for i, (ann, ref, src, dst, start, end) in enumerate(planned):
            if self.cancelled:
                self.log_message.emit("Conversion cancelled by user")
                self.finished_signal.emit(False, "Cancelled by user")
                return

            # Derive subtask text from destination path structure
            # dst = out_dir / tissue_N / subtask_dir / episode_NNN
            subtask_dir_name = dst.parent.name          # e.g. "1_grasp"
            subtask_text = " ".join(subtask_dir_name.split("_")[1:])  # "grasp"
            is_recovery = "recovery" in subtask_text.lower()
            if is_recovery:
                subtask_text = subtask_text.replace(" recovery", "").replace("recovery", "").strip()

            self.progress.emit(i, len(planned))
            self.episode_started.emit(f"{dst.parent.name}/{dst.name}")

            # Skip already-converted episodes during resume
            if self.resume_mode and valid_seen < self.skip_count:
                valid_seen += 1
                self.log_message.emit(f"⏭ Skipping (already converted): {dst.name}")
                continue

            try:
                t_ep_start = time.time()
                self._process_planned_episode(dataset, ref, src, start, end, subtask_text)
                t_proc_elapsed = time.time() - t_ep_start
                self.log_message.emit(
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
                self.log_message.emit(
                    f"✓ {dst.parent.name}/{dst.name} saved — "
                    f"processing: {t_proc_elapsed:.1f}s, encoding: {t_enc_elapsed:.1f}s, "
                    f"total: {t_ep_total:.1f}s  [task: \"{subtask_text}\"]"
                )

                # Compute and emit ETA
                elapsed = time.time() - start_time
                completed_count = i + 1 - (self.skip_count if self.resume_mode else 0)
                remaining_count = len(planned) - (i + 1)
                if completed_count > 0 and remaining_count > 0:
                    # Use recent episodes (last 5) for better estimate
                    recent = episode_times[-5:]
                    avg_time = sum(recent) / len(recent)
                    eta_secs = avg_time * remaining_count
                    elapsed_str = self._format_duration(elapsed)
                    eta_str = self._format_duration(eta_secs)
                    self.eta_update.emit(
                        f"Elapsed: {elapsed_str}  |  ETA: ~{eta_str} remaining"
                    )
                elif remaining_count == 0:
                    elapsed_str = self._format_duration(elapsed)
                    self.eta_update.emit(f"Elapsed: {elapsed_str}  |  Done!")
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                self.log_message.emit(f"✗ Error processing {dst.name}:")
                self.log_message.emit(f"  {error_details}")
                try:
                    dataset.clear_episode_buffer()
                except:
                    pass

        # Final progress
        self.progress.emit(len(planned), len(planned))
        elapsed_str = self._format_duration(time.time() - start_time)
        self.eta_update.emit(f"Elapsed: {elapsed_str}  |  Complete")

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
            self.log_message.emit(
                f"Split configuration saved — "
                f"train: {train_count}, val: {val_count}, "
                f"test: {total_episodes - train_count - val_count} | "
                f"perfect: {perfect_count}, recovery: {recovery_count}"
            )

        # Clean up intermediate images
        self.log_message.emit("Cleaning up intermediate images...")
        self._cleanup_all_images(dataset)

        # Flush parquet files for v3.0
        self.log_message.emit("Finalizing dataset metadata...")
        dataset.finalize()

        elapsed = time.time() - start_time
        self.log_message.emit(f"\n{'='*50}")
        self.log_message.emit("Conversion complete!")
        self.log_message.emit(f"Total episodes: {successful_episodes}/{len(planned)}")
        self.log_message.emit(f"Time elapsed: {elapsed:.1f} seconds")
        self.log_message.emit(f"Dataset saved to: {output_path}")

        self.finished_signal.emit(
            True,
            f"Successfully converted {successful_episodes} episodes\nOutput: {output_path}"
        )

    # -----------------------------------------------------------------
    # Legacy flat-directory mode (no annotations)
    # -----------------------------------------------------------------

    def _run_flat_conversion(self, LeRobotDataset, write_info):
        """Original flat-directory conversion (source/episode_xxx/...)."""

        # Find all episode directories
        episodes = sorted([d for d in self.source_path.iterdir() if d.is_dir()],
                         key=lambda x: x.name)

        if not episodes:
            self.finished_signal.emit(False, "No episode directories found in source path")
            return

        self.log_message.emit(f"Found {len(episodes)} episodes to convert")

        output_path = self.output_dir / self.dataset_name
        self.log_message.emit(f"Output path: {output_path}")

        if self.resume_mode:
            dataset = self._init_resume_dataset(LeRobotDataset, output_path)
        else:
            if output_path.exists():
                self.log_message.emit(f"Removing existing dataset at {output_path}")
                shutil.rmtree(output_path)
            img_shape_endo, img_shape_wrist = self._get_image_shapes(episodes)
            self.log_message.emit(f"Endoscope image shape: {img_shape_endo}")
            self.log_message.emit(f"Wrist camera image shape: {img_shape_wrist}")
            self.log_message.emit("Creating LeRobot dataset...")
            dataset = self._create_dataset(LeRobotDataset, img_shape_endo, img_shape_wrist, output_path)

        # Process each episode
        start_time = time.time()
        successful_episodes = self.skip_count if self.resume_mode else 0
        valid_seen = 0

        # Derive a default task text from the source directory name
        default_task_text = self.source_path.name.replace("_", " ")

        for i, episode_path in enumerate(episodes):
            if self.cancelled:
                self.log_message.emit("Conversion cancelled by user")
                self.finished_signal.emit(False, "Cancelled by user")
                return

            self.progress.emit(i, len(episodes))
            self.episode_started.emit(episode_path.name)

            # Validate episode
            is_valid, error_msg = validate_episode(episode_path)
            if not is_valid:
                self.log_message.emit(f"⚠ Skipping invalid episode {episode_path.name}: {error_msg}")
                continue

            # Skip already-converted episodes during resume
            if self.resume_mode and valid_seen < self.skip_count:
                valid_seen += 1
                self.log_message.emit(f"⏭ Skipping (already converted): {episode_path.name}")
                continue

            try:
                t_ep_start = time.time()
                self._process_episode(dataset, episode_path, default_task_text)
                t_proc_elapsed = time.time() - t_ep_start
                self.log_message.emit(
                    f"    → Frame processing complete ({t_proc_elapsed:.1f}s). "
                    f"Encoding videos ({VIDEO_CODEC}, parallel)..."
                )
                t_enc = time.time()
                dataset.save_episode()
                t_enc_elapsed = time.time() - t_enc
                t_ep_total = time.time() - t_ep_start
                successful_episodes += 1
                self.log_message.emit(
                    f"✓ Episode {episode_path.name} saved — "
                    f"processing: {t_proc_elapsed:.1f}s, encoding: {t_enc_elapsed:.1f}s, "
                    f"total: {t_ep_total:.1f}s"
                )
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                self.log_message.emit(f"✗ Error processing {episode_path.name}:")
                self.log_message.emit(f"  {error_details}")
                try:
                    dataset.clear_episode_buffer()
                except:
                    pass  # Ignore cleanup errors

        # Final progress update
        self.progress.emit(len(episodes), len(episodes))

        # Write splits
        total_episodes = successful_episodes
        if total_episodes > 0:
            dataset.meta.info["splits"] = {
                "train": f"0:{total_episodes}",
                "val": "0:0",
                "test": "0:0",
            }
            write_info(dataset.meta.info, dataset.root)
            self.log_message.emit(f"Split configuration saved (train: {total_episodes}, val: 0, test: 0)")

        # Clean up ALL intermediate images at the end
        self.log_message.emit("Cleaning up intermediate images...")
        self._cleanup_all_images(dataset)

        # Flush parquet files for v3.0
        self.log_message.emit("Finalizing dataset metadata...")
        dataset.finalize()

        elapsed = time.time() - start_time
        self.log_message.emit(f"\n{'='*50}")
        self.log_message.emit(f"Conversion complete!")
        self.log_message.emit(f"Total episodes: {successful_episodes}/{len(episodes)}")
        self.log_message.emit(f"Time elapsed: {elapsed:.1f} seconds")
        self.log_message.emit(f"Dataset saved to: {output_path}")

        self.finished_signal.emit(True, f"Successfully converted {successful_episodes} episodes\nOutput: {output_path}")
    
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
    
    @staticmethod
    def _load_image_cv2(path: Path) -> np.ndarray:
        """Load a JPEG image via OpenCV and return as RGB numpy array."""
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Failed to read image: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

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

        self.log_message.emit(
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

        t_prep = time.time()
        self.log_message.emit(
            f"  Processing {total_frames} frames "
            f"(align: {t_align - t_start:.1f}s, prep: {t_prep - t_align:.1f}s)..."
        )

        # ------------------------------------------------------------------
        # 4. Pre-place JPEGs as hardlinks and add zero'd dummy arrays
        # ------------------------------------------------------------------
        from lerobot.datasets.utils import DEFAULT_IMAGE_PATH
        import shutil as _shutil

        fps_inv = 1.0 / self.fps
        t_add_total = 0.0

        ep_idx_for_paths = dataset.episode_buffer["episode_index"]
        _temp_base = TEMP_IMAGE_DIR if TEMP_IMAGE_DIR is not None else dataset.root

        camera_path_map = {
            "observation.images.endoscope.left":  left_paths,
        }
        if right_paths: camera_path_map["observation.images.endoscope.right"] = right_paths
        if psm1_paths: camera_path_map["observation.images.wrist.right"] = psm1_paths
        if psm2_paths: camera_path_map["observation.images.wrist.left"] = psm2_paths

        # Cache dummy arrays
        _dummy_cache = {}
        for img_key in camera_path_map:
            feat = dataset.features[img_key]
            shape = tuple(feat["shape"])
            _dummy_cache[img_key] = np.zeros(shape, dtype=np.uint8)

        t_preplace_start = time.time()
        preplace_count = 0
        preplace_fallback = 0

        for img_key, paths in camera_path_map.items():
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
                    except OSError:
                        _shutil.copy2(str(source_jpg), str(target))
                        preplace_fallback += 1
                    preplace_count += 1

        t_preplace = time.time()
        fallback_msg = f" ({preplace_fallback} fell back to copy)" if preplace_fallback else ""
        self.log_message.emit(
            f"  Pre-placed {preplace_count} source JPEGs as hardlinks "
            f"({t_preplace - t_preplace_start:.1f}s){fallback_msg}"
        )

        for frame_idx in range(total_frames):
            if frame_idx > 0 and frame_idx % 500 == 0:
                avg_add = (t_add_total / frame_idx) * 1000
                self.log_message.emit(
                    f"    -> {frame_idx}/{total_frames} "
                    f"(avg add_frame={avg_add:.1f}ms)"
                )

            src_idx = new_start + frame_idx
            frame = {
                "observation.state": states_matrix[src_idx],
                "action":            actions_matrix[src_idx],
                "instruction.text":  subtask_text,
                "observation.meta.tool.psm1": self.psm1_tool,
                "observation.meta.tool.psm2": self.psm2_tool,
                "task": subtask_text,
            }
            # Inject dummy arrays (the hardlinks skip writing process)
            for img_key in camera_path_map:
                if src_idx < len(camera_path_map[img_key]):
                    frame[img_key] = _dummy_cache[img_key]

            t0 = time.time()
            dataset.add_frame(frame)
            t_add_total += time.time() - t0

        t_end = time.time()
        self.log_message.emit(
            f"    -> {total_frames}/{total_frames} frames (100%) "
            f"| total: {t_end - t_start:.1f}s "
            f"(align: {t_align - t_start:.1f}s, "
            f"prep: {t_prep - t_align:.1f}s, "
            f"pre-place: {t_preplace - t_preplace_start:.1f}s, "
            f"add_frame: {t_add_total:.1f}s)"
        )

    def _process_episode(self, dataset, episode_path: Path, task_text: str = "default_task"):
        """Process a single episode with strict timestamp-based synchronization.

        Delegates sync + filtering to ``run_filter_episode()`` which handles:
        1. Left-camera <-> kinematic sync with outlier removal
        2. Vectorized multi-camera sync (all cameras within threshold)

        The threshold is derived from fps: ``max_time_diff_ms = 1000 / fps``.

        Optimisations preserved:
        - Deep pipeline: 16 worker threads pre-build frame dicts ahead
        - cv2.imread: faster JPEG decoding (releases GIL)
        - Pre-computed state/action matrices
        """
        t_start = time.time()

        # Derive sync threshold from fps: one frame period
        max_time_diff_ms = self.fps

        # ------------------------------------------------------------------
        # 1. Run full sync + multi-camera filtering
        # ------------------------------------------------------------------
        self.log_message.emit(
            f"  Running strict sync (threshold={max_time_diff_ms:.1f}ms)..."
        )
        filt = run_filter_episode(episode_path, max_time_diff_ms)

        if not filt["success"]:
            raise ValueError(f"Sync failed: {filt.get('error', 'Unknown')}")

        valid_left_filenames = filt["valid_left_filenames"]
        kinematics_indices  = filt["kinematics_indices"]
        secondary_indices   = filt["secondary_camera_indices"]

        self.log_message.emit(
            f"  Sync complete: {filt['num_valid']} synced frames "
            f"({filt['outliers_removed']} kinematic outliers, "
            f"{filt['multicam_dropped']} camera mismatches removed)"
        )

        t_sync = time.time()

        # ------------------------------------------------------------------
        # 2. Load frame lists and map filenames to paths
        # ------------------------------------------------------------------
        left_frames  = load_frames_from_dir(episode_path / LEFT_IMG_DIR)
        right_frames = load_frames_from_dir(episode_path / RIGHT_IMG_DIR)
        psm1_frames  = load_frames_from_dir(episode_path / ENDO_PSM1_DIR)
        psm2_frames  = load_frames_from_dir(episode_path / ENDO_PSM2_DIR)

        if not left_frames:
            raise ValueError("No frames in left_img_dir")

        # Map valid left filenames -> indices in left_frames
        left_fname_to_idx = {f.path.name: i for i, f in enumerate(left_frames)}
        final_left_indices = np.array(
            [left_fname_to_idx[fn] for fn in valid_left_filenames],
            dtype=np.int64,
        )

        # ------------------------------------------------------------------
        # 3. Load CSV & pre-compute state/action matrices
        # ------------------------------------------------------------------
        csv_path = episode_path / CSV_FILE
        df = pd.read_csv(csv_path)

        state_cols  = [c for c in STATES_NAME  if c in df.columns]
        action_cols = [c for c in ACTIONS_NAME if c in df.columns]
        states_matrix  = df[state_cols].values.astype(np.float32)
        actions_matrix = df[action_cols].values.astype(np.float32)

        t_prep = time.time()

        total_frames = len(final_left_indices)
        self.log_message.emit(
            f"  Processing {total_frames} synchronized frames "
            f"(sync: {t_sync - t_start:.1f}s, prep: {t_prep - t_sync:.1f}s)..."
        )

        # ------------------------------------------------------------------
        # 4. Pre-place JPEGs as hardlinks and add zero'd dummy arrays
        # ------------------------------------------------------------------
        from lerobot.datasets.utils import DEFAULT_IMAGE_PATH
        import shutil as _shutil

        fps_inv = 1.0 / self.fps
        t_add_total = 0.0

        ep_idx_for_paths = dataset.episode_buffer["episode_index"]
        _temp_base = TEMP_IMAGE_DIR if TEMP_IMAGE_DIR is not None else dataset.root

        # Pre-build path lists for fast indexing
        left_paths  = [f.path for f in left_frames]
        right_paths = [f.path for f in right_frames] if right_frames else None
        psm1_paths  = [f.path for f in psm1_frames] if psm1_frames else None
        psm2_paths  = [f.path for f in psm2_frames] if psm2_frames else None

        # Secondary camera index arrays (already final from run_filter_episode)
        right_idx_arr = secondary_indices.get("right_img_dir")
        psm1_idx_arr  = secondary_indices.get("endo_psm1")
        psm2_idx_arr  = secondary_indices.get("endo_psm2")

        camera_path_map = {}
        # We need to map img_key -> list of mapped source paths for this episode length
        camera_path_map["observation.images.endoscope.left"] = [left_paths[idx] for idx in final_left_indices]

        if right_idx_arr is not None and right_paths:
            camera_path_map["observation.images.endoscope.right"] = [right_paths[idx] for idx in right_idx_arr]
        if psm1_idx_arr is not None and psm1_paths:
            camera_path_map["observation.images.wrist.right"] = [psm1_paths[idx] for idx in psm1_idx_arr]
        if psm2_idx_arr is not None and psm2_paths:
            camera_path_map["observation.images.wrist.left"] = [psm2_paths[idx] for idx in psm2_idx_arr]

        # Cache dummy arrays
        _dummy_cache = {}
        for img_key in camera_path_map:
            feat = dataset.features[img_key]
            shape = tuple(feat["shape"])
            _dummy_cache[img_key] = np.zeros(shape, dtype=np.uint8)

        t_preplace_start = time.time()
        preplace_count = 0
        preplace_fallback = 0

        for img_key, paths in camera_path_map.items():
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
                    except OSError:
                        _shutil.copy2(str(source_jpg), str(target))
                        preplace_fallback += 1
                    preplace_count += 1

        t_preplace = time.time()
        fallback_msg = f" ({preplace_fallback} fell back to copy)" if preplace_fallback else ""
        self.log_message.emit(
            f"  Pre-placed {preplace_count} source JPEGs as hardlinks "
            f"({t_preplace - t_preplace_start:.1f}s){fallback_msg}"
        )

        for frame_idx in range(total_frames):
            # Progress every 500 frames
            if frame_idx > 0 and frame_idx % 500 == 0:
                avg_add = (t_add_total / frame_idx) * 1000
                self.log_message.emit(
                    f"    -> {frame_idx}/{total_frames} "
                    f"(avg add_frame={avg_add:.1f}ms)"
                )

            csv_idx = kinematics_indices[frame_idx]
            frame = {
                "observation.state": states_matrix[csv_idx],
                "action": actions_matrix[csv_idx],
                "instruction.text": task_text,
                "observation.meta.tool.psm1": self.psm1_tool,
                "observation.meta.tool.psm2": self.psm2_tool,
                "task": task_text,
            }
            # Inject dummy arrays (the hardlinks skip writing process)
            for img_key in camera_path_map:
                if frame_idx < len(camera_path_map[img_key]):
                    frame[img_key] = _dummy_cache[img_key]

            t0 = time.time()
            dataset.add_frame(frame)
            t_add_total += time.time() - t0

        t_end = time.time()
        self.log_message.emit(
            f"    -> {total_frames}/{total_frames} frames processed (100%) "
            f"| total: {t_end - t_start:.1f}s "
            f"(sync: {t_sync - t_start:.1f}s, "
            f"prep: {t_prep - t_sync:.1f}s, "
            f"add_frame: {t_add_total:.1f}s, "
            f"pipeline overhead: {(t_end - t_prep) - t_add_total:.1f}s)"
        )
    
    def _cleanup_all_images(self, dataset):
        """Remove ALL intermediate images after all video encoding is complete"""
        for images_dir in [dataset.root / "images",
                           TEMP_IMAGE_DIR / "images" if TEMP_IMAGE_DIR else None]:
            try:
                if images_dir and images_dir.exists():
                    shutil.rmtree(images_dir)
                    self.log_message.emit(f"  Cleaned up: {images_dir}")
            except Exception as e:
                self.log_message.emit(f"  Warning: Could not clean up {images_dir}: {str(e)}")


# =============================================================================
# MAIN WINDOW
# =============================================================================
class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DVRK to LeRobot Converter")
        self.setGeometry(100, 100, 900, 750)
        
        self.source_path: Optional[Path] = None
        self.annotations_path: Optional[Path] = None
        self.output_dir: Path = Path(DEFAULT_LEROBOT_HOME)
        self.worker: Optional[ConversionWorker] = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        
        # Title
        title = QLabel("DVRK → LeRobot v3.0 Converter")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #2c3e50; margin: 10px;")
        main_layout.addWidget(title)
        
        # Source & Pipeline Directory Selection
        source_group = QGroupBox("Pipeline Directories")
        source_vlayout = QVBoxLayout(source_group)

        # Raw data directory
        raw_layout = QHBoxLayout()
        raw_layout.addWidget(QLabel("Raw Data Directory:"))
        self.source_label = QLabel("No directory selected")
        self.source_label.setStyleSheet("color: #888; font-style: italic;")
        raw_layout.addWidget(self.source_label, stretch=1)
        self.select_source_btn = QPushButton("Browse...")
        self.select_source_btn.clicked.connect(self._select_source_directory)
        self.select_source_btn.setStyleSheet("background-color: #3498db; color: white;")
        raw_layout.addWidget(self.select_source_btn)
        source_vlayout.addLayout(raw_layout)

        # Annotations directory (post_process)
        ann_layout = QHBoxLayout()
        ann_layout.addWidget(QLabel("Annotations Dir:"))
        self.annotations_label = QLabel("(optional — enables pipeline mode)")
        self.annotations_label.setStyleSheet("color: #888; font-style: italic;")
        ann_layout.addWidget(self.annotations_label, stretch=1)
        self.select_annotations_btn = QPushButton("Browse...")
        self.select_annotations_btn.clicked.connect(self._select_annotations_directory)
        self.select_annotations_btn.setStyleSheet("background-color: #3498db; color: white;")
        ann_layout.addWidget(self.select_annotations_btn)
        source_vlayout.addLayout(ann_layout)



        main_layout.addWidget(source_group)
        
        # Destination/Repo Settings
        dest_group = QGroupBox("Output Settings")
        dest_layout = QVBoxLayout(dest_group)
        
        # Output directory selection
        output_dir_layout = QHBoxLayout()
        output_dir_layout.addWidget(QLabel("Output Directory:"))
        self.output_dir_label = QLabel(str(DEFAULT_LEROBOT_HOME))
        self.output_dir_label.setStyleSheet("color: #333;")
        output_dir_layout.addWidget(self.output_dir_label, stretch=1)
        
        self.select_output_btn = QPushButton("Browse...")
        self.select_output_btn.clicked.connect(self._select_output_directory)
        self.select_output_btn.setStyleSheet("background-color: #3498db; color: white;")
        output_dir_layout.addWidget(self.select_output_btn)
        dest_layout.addLayout(output_dir_layout)
        
        # Dataset name
        name_layout = QFormLayout()
        self.dataset_name_edit = QLineEdit("Cholecystectomy")
        self.dataset_name_edit.setPlaceholderText("dataset-name")
        name_layout.addRow("Dataset Name:", self.dataset_name_edit)
        dest_layout.addLayout(name_layout)
        
        # Output path preview
        self.output_preview = QLabel("")
        self.output_preview.setStyleSheet("color: #666; font-style: italic;")
        dest_layout.addWidget(self.output_preview)
        self._update_output_preview()
        
        # Connect name change to preview update
        self.dataset_name_edit.textChanged.connect(self._update_output_preview)
        
        main_layout.addWidget(dest_group)
        
        # Metadata Settings
        meta_group = QGroupBox("Metadata")
        meta_layout = QFormLayout(meta_group)
        
        self.psm1_tool_edit = QLineEdit("Permanent Cautery Hook")
        meta_layout.addRow("PSM1 Tool:", self.psm1_tool_edit)
        
        self.psm2_tool_edit = QLineEdit("Prograsp Forceps")
        meta_layout.addRow("PSM2 Tool:", self.psm2_tool_edit)
        
        self.fps_spinbox = QSpinBox()
        self.fps_spinbox.setRange(1, 120)
        self.fps_spinbox.setValue(30)
        meta_layout.addRow("FPS:", self.fps_spinbox)
        
        self.codec_combo = QComboBox()
        for label in CODEC_OPTIONS:
            self.codec_combo.addItem(label)
        self.codec_combo.setCurrentText(DEFAULT_CODEC_LABEL)
        meta_layout.addRow("Video Codec:", self.codec_combo)
        
        main_layout.addWidget(meta_group)
        
        # Progress Section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.episode_label = QLabel("Episode: -")
        self.episode_label.setStyleSheet("font-weight: bold;")
        progress_layout.addWidget(self.episode_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setFormat("%v / %m episodes")
        progress_layout.addWidget(self.progress_bar)
        
        self.eta_label = QLabel("")
        self.eta_label.setStyleSheet(
            "color: #555; font-size: 11px; font-style: italic;"
        )
        progress_layout.addWidget(self.eta_label)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(200)
        self.log_text.setStyleSheet("font-family: monospace; font-size: 11px;")
        progress_layout.addWidget(self.log_text)
        
        main_layout.addWidget(progress_group, stretch=1)
        
        # Action Buttons
        btn_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Conversion")
        self.start_btn.clicked.connect(self._start_conversion)
        self.start_btn.setStyleSheet("""
            background-color: #27ae60; 
            color: white; 
            font-weight: bold;
            padding: 10px 20px;
            font-size: 14px;
        """)
        self.start_btn.setEnabled(False)
        btn_layout.addWidget(self.start_btn)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._cancel_conversion)
        self.cancel_btn.setStyleSheet("""
            background-color: #e74c3c; 
            color: white;
            padding: 10px 20px;
            font-size: 14px;
        """)
        self.cancel_btn.setEnabled(False)
        btn_layout.addWidget(self.cancel_btn)
        
        self.clear_cache_btn = QPushButton("Clear Cache")
        self.clear_cache_btn.clicked.connect(self._clear_cache)
        self.clear_cache_btn.setStyleSheet("""
            background-color: #f39c12;
            color: white;
            padding: 10px 20px;
            font-size: 14px;
        """)
        self.clear_cache_btn.setToolTip(
            "Delete the filtered data cache. Use this if the cache is "
            "corrupted or was created with incorrect settings."
        )
        btn_layout.addWidget(self.clear_cache_btn)
        
        main_layout.addLayout(btn_layout)
    
    def _log(self, message: str):
        """Add message to log"""
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
        QApplication.processEvents()
    
    def _select_source_directory(self):
        """Allow user to select the source data directory"""
        selected_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Source Data Directory",
            str(self.source_path) if self.source_path else "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if not selected_dir:
            return
        
        self.source_path = Path(selected_dir)
        self.source_label.setText(str(self.source_path))
        self.source_label.setStyleSheet("color: black;")
        
        # Count episodes
        episodes = [d for d in self.source_path.iterdir() if d.is_dir()]
        self._log(f"Selected: {self.source_path}")
        self._log(f"Found {len(episodes)} episode directories")
        
        self.start_btn.setEnabled(len(episodes) > 0)

    def _select_annotations_directory(self):
        """Allow user to select the annotations (post_process) directory."""
        selected_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Annotations Directory (post_process)",
            str(self.annotations_path) if self.annotations_path else "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        if not selected_dir:
            return
        self.annotations_path = Path(selected_dir)
        self.annotations_label.setText(str(self.annotations_path))
        self.annotations_label.setStyleSheet("color: black;")
        self._log(f"Annotations dir: {self.annotations_path}")

    
    def _select_output_directory(self):
        """Allow user to select the output directory"""
        selected_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            str(self.output_dir),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if not selected_dir:
            return
        
        self.output_dir = Path(selected_dir)
        self.output_dir_label.setText(str(self.output_dir))
        self._update_output_preview()
    
    def _update_output_preview(self):
        """Update the output path preview label"""
        dataset_name = self.dataset_name_edit.text().strip() or "dataset-name"
        full_path = self.output_dir / dataset_name
        self.output_preview.setText(f"Full output path: {full_path}")
    
    def _count_completed_episodes(self, output_path: Path) -> int:
        """Count fully completed episodes by checking parquet + all 4 video files exist."""
        VIDEO_KEYS = [
            "observation.images.endoscope.left",
            "observation.images.endoscope.right",
            "observation.images.wrist.left",
            "observation.images.wrist.right",
        ]
        idx = 0
        while True:
            ep_name = f"episode_{idx:06d}"
            # Check parquet (could be in any chunk dir)
            parquet_matches = list((output_path / "data").glob(f"chunk-*/{ep_name}.parquet"))
            if not parquet_matches:
                break
            # Check all 4 video files (inside chunk-*/key/ dirs)
            all_videos = all(
                list((output_path / "videos").glob(f"chunk-*/{key}/{ep_name}.mp4"))
                for key in VIDEO_KEYS
            )
            if not all_videos:
                break
            idx += 1
        return idx
    
    def _start_conversion(self):
        """Start the conversion process"""
        if not self.source_path:
            QMessageBox.warning(self, "Error", "Please select a source directory first.")
            return
        
        dataset_name = self.dataset_name_edit.text().strip()
        if not dataset_name:
            QMessageBox.warning(self, "Error", "Please enter a dataset name.")
            return
        
        # Check for existing dataset — offer resume or overwrite
        resume_mode = False
        skip_count = 0
        output_path = self.output_dir / dataset_name
        if output_path.exists():
            completed = self._count_completed_episodes(output_path)
            if completed > 0:
                # Offer resume
                msg = QMessageBox(self)
                msg.setWindowTitle("Resume or Start Over?")
                msg.setText(
                    f"Found {completed} fully completed episodes (0–{completed - 1}) in:\n{output_path}\n\n"
                    f"Resume from episode {completed}?"
                )
                resume_btn = msg.addButton("Resume", QMessageBox.AcceptRole)
                start_over_btn = msg.addButton("Start Over", QMessageBox.DestructiveRole)
                msg.addButton(QMessageBox.Cancel)
                msg.exec_()
                
                clicked = msg.clickedButton()
                if clicked == resume_btn:
                    resume_mode = True
                    skip_count = completed
                elif clicked == start_over_btn:
                    resume_mode = False
                else:
                    return  # Cancel
            else:
                reply = QMessageBox.question(
                    self, "Dataset Exists",
                    f"Dataset already exists at:\n{output_path}\n\nOverwrite?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return
        
        # Set the video codec from the dropdown
        global VIDEO_CODEC
        VIDEO_CODEC = CODEC_OPTIONS[self.codec_combo.currentText()]
        
        # Disable UI during conversion
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.clear_cache_btn.setEnabled(False)
        self.select_source_btn.setEnabled(False)
        self.select_annotations_btn.setEnabled(False)
        self.select_output_btn.setEnabled(False)
        self.dataset_name_edit.setEnabled(False)
        self.psm1_tool_edit.setEnabled(False)
        self.psm2_tool_edit.setEnabled(False)
        self.fps_spinbox.setEnabled(False)
        self.codec_combo.setEnabled(False)
        self.eta_label.setText("")
        
        self.log_text.clear()
        if resume_mode:
            self._log(f"Resuming conversion from episode {skip_count}...")
        else:
            self._log("Starting conversion...")
        
        # Create and start worker thread
        self.worker = ConversionWorker(
            source_path=self.source_path,
            output_dir=self.output_dir,
            dataset_name=dataset_name,
            psm1_tool=self.psm1_tool_edit.text().strip(),
            psm2_tool=self.psm2_tool_edit.text().strip(),
            fps=self.fps_spinbox.value(),
            annotations_dir=self.annotations_path,
            resume_mode=resume_mode,
            skip_count=skip_count,
        )
        
        self.worker.progress.connect(self._on_progress)
        self.worker.log_message.connect(self._log)
        self.worker.episode_started.connect(self._on_episode_started)
        self.worker.eta_update.connect(self._on_eta_update)
        self.worker.finished_signal.connect(self._on_finished)
        
        self.worker.start()
    
    def _cancel_conversion(self):
        """Cancel the conversion process"""
        if self.worker:
            self.worker.cancel()
            self._log("Cancelling... please wait")
    
    def _on_progress(self, current: int, total: int):
        """Update progress bar"""
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(current)
    
    def _on_episode_started(self, episode_name: str):
        """Update episode label"""
        self.episode_label.setText(f"Episode: {episode_name}")
    
    def _on_eta_update(self, eta_text: str):
        """Update the ETA label."""
        self.eta_label.setText(eta_text)

    def _clear_cache(self):
        """Delete the _filtered_cache directory after user confirmation."""
        cache_dir = self.output_dir / "_filtered_cache"

        if not cache_dir.exists():
            QMessageBox.information(
                self, "No Cache",
                "No filtered data cache found.\n\n"
                f"Expected location:\n{cache_dir}"
            )
            return

        reply = QMessageBox.warning(
            self,
            "Clear Filtered Data Cache?",
            "This will delete the cached filtered data. The next conversion "
            "will re-run the filtering stage (Stage 1), which takes longer.\n\n"
            "Use this if:\n"
            "  • The cache was created with incorrect settings\n"
            "  • Data appears corrupted or misaligned\n"
            "  • You changed the source data since the last run\n\n"
            f"Cache location:\n{cache_dir}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            import shutil
            try:
                shutil.rmtree(cache_dir)
                self._log(f"✓ Cleared filtered data cache: {cache_dir}")
                QMessageBox.information(
                    self, "Cache Cleared",
                    "Filtered data cache deleted successfully.\n"
                    "Stage 1 will re-run on the next conversion."
                )
            except Exception as e:
                self._log(f"✗ Failed to clear cache: {e}")
                QMessageBox.critical(
                    self, "Error",
                    f"Failed to delete cache:\n{e}"
                )

    def _on_finished(self, success: bool, message: str):
        """Handle conversion completion"""
        # Re-enable UI
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.clear_cache_btn.setEnabled(True)
        self.select_source_btn.setEnabled(True)
        self.select_annotations_btn.setEnabled(True)
        self.select_output_btn.setEnabled(True)
        self.dataset_name_edit.setEnabled(True)
        self.psm1_tool_edit.setEnabled(True)
        self.psm2_tool_edit.setEnabled(True)
        self.fps_spinbox.setEnabled(True)
        self.codec_combo.setEnabled(True)
        
        self.episode_label.setText("Episode: -")
        
        if success:
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.warning(self, "Conversion Issue", message)
        
        self.worker = None


# =============================================================================
# MAIN
# =============================================================================
def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
