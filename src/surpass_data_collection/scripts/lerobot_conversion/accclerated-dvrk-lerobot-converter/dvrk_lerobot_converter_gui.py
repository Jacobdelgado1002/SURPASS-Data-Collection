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

from filter_episodes import run_filter_episode

# Default LeRobot home for UI display only
DEFAULT_LEROBOT_HOME = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "lerobot")

# =============================================================================
# CONFIGURATION
# =============================================================================
# Available video codecs (selectable in the GUI):
CODEC_OPTIONS = {
    "h264 (CPU — works everywhere)":       "h264",
    "h264_nvenc (NVIDIA GPU — fastest)":    "h264_nvenc",
    "h264_amf (AMD GPU)":                   "h264_amf",
    "h264_qsv (Intel Quick Sync)":          "h264_qsv",
    "libsvtav1 (CPU — best compression, VERY SLOW)": "libsvtav1",
}
DEFAULT_CODEC_LABEL = "h264_nvenc (NVIDIA GPU — fastest)"

# This global is updated from the GUI before conversion starts
VIDEO_CODEC = "h264_nvenc"

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
        if video_path.is_file():
            return  # already encoded (resume case)
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
# CONVERSION WORKER THREAD
# =============================================================================
class ConversionWorker(QThread):
    """Worker thread for running the conversion without blocking the UI"""
    
    progress = pyqtSignal(int, int)  # current, total
    log_message = pyqtSignal(str)
    episode_started = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)  # success, message
    
    def __init__(self, source_path: Path, output_dir: Path, dataset_name: str,
                 task_text: str, psm1_tool: str, psm2_tool: str, fps: int,
                 resume_mode: bool = False, skip_count: int = 0):
        super().__init__()
        self.source_path = source_path
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.task_text = task_text
        self.psm1_tool = psm1_tool
        self.psm2_tool = psm2_tool
        self.fps = fps
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
    
    def _run_conversion(self):
        # Set HF_LEROBOT_HOME BEFORE importing LeRobot modules
        os.environ["HF_LEROBOT_HOME"] = str(self.output_dir)
        
        # Now import LeRobot (this will use the custom HF_LEROBOT_HOME)
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.datasets.utils import write_info
        
        # Find all episode directories
        episodes = sorted([d for d in self.source_path.iterdir() if d.is_dir()], 
                         key=lambda x: x.name)
        
        if not episodes:
            self.finished_signal.emit(False, "No episode directories found in source path")
            return
        
        self.log_message.emit(f"Found {len(episodes)} episodes to convert")
        
        # Output path (will be created directly in the chosen location)
        output_path = self.output_dir / self.dataset_name
        self.log_message.emit(f"Output path: {output_path}")
        
        if self.resume_mode:
            # --- RESUME: clean up partial data, then load existing dataset ---
            self.log_message.emit(f"Resuming: {self.skip_count} episodes already completed")
            
            # Delete stale intermediate images from killed-mid-episode process
            # Check both the dataset dir (old runs) and the local temp dir (new runs)
            for stale_dir in [output_path / "images",
                              TEMP_IMAGE_DIR / "images" if TEMP_IMAGE_DIR else None]:
                if stale_dir and stale_dir.exists():
                    self.log_message.emit(f"Cleaning up stale intermediate images in {stale_dir}...")
                    shutil.rmtree(stale_dir)
            
            # Delete any partial files for the first incomplete episode
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
            
            # Load existing dataset
            self.log_message.emit("Loading existing dataset...")
            dataset = LeRobotDataset(repo_id=self.dataset_name)
            dataset.start_image_writer(num_processes=0, num_threads=16)
            _patch_parallel_encoding(dataset)
            _patch_jpeg_image_saving(dataset)
            self.log_message.emit(f"  Image writer: 16 threads (JPEG)  |  Video encoding: parallel, codec={VIDEO_CODEC}")
        else:
            # --- FRESH: wipe and create new dataset ---
            if output_path.exists():
                self.log_message.emit(f"Removing existing dataset at {output_path}")
                shutil.rmtree(output_path)
            
            # Get image dimensions from first valid episode
            img_shape_endo, img_shape_wrist = self._get_image_shapes(episodes)
            self.log_message.emit(f"Endoscope image shape: {img_shape_endo}")
            self.log_message.emit(f"Wrist camera image shape: {img_shape_wrist}")
            
            # Initialize LeRobot dataset
            self.log_message.emit("Creating LeRobot dataset...")
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
            self.log_message.emit(f"  Image writer: 16 threads (JPEG)  |  Video encoding: parallel, codec={VIDEO_CODEC}")
        
        # Process each episode
        start_time = time.time()
        successful_episodes = self.skip_count if self.resume_mode else 0
        valid_seen = 0
        
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
                self._process_episode(dataset, episode_path)
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
        
        # Note: Video encoding happens automatically via batch_encoding_size=1
        # Each episode is encoded to MP4 right after save_episode() is called
        
        # Write splits - all episodes go to train, val and test are empty
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

    def _process_episode(self, dataset, episode_path: Path):
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
        # 4. Pipelined frame processing
        #    16 workers pre-build frame dicts (image loading + dict assembly)
        #    while the main thread just calls add_frame().
        # ------------------------------------------------------------------
        PIPELINE_DEPTH = 64
        NUM_WORKERS = 16

        fps_inv = 1.0 / self.fps
        t_add_total = 0.0

        # Pre-build path lists for fast indexing
        left_paths  = [f.path for f in left_frames]
        right_paths = [f.path for f in right_frames] if right_frames else None
        psm1_paths  = [f.path for f in psm1_frames] if psm1_frames else None
        psm2_paths  = [f.path for f in psm2_frames] if psm2_frames else None

        # Secondary camera index arrays (already final from run_filter_episode)
        right_idx_arr = secondary_indices.get("right_img_dir")
        psm1_idx_arr  = secondary_indices.get("endo_psm1")
        psm2_idx_arr  = secondary_indices.get("endo_psm2")

        def _build_frame(out_idx):
            """Build a complete frame dict -- runs in worker thread.
            cv2.imread releases the GIL so threads truly run in parallel."""
            left_idx = final_left_indices[out_idx]
            csv_idx  = kinematics_indices[out_idx]

            left_img = self._load_image_cv2(left_paths[left_idx])

            frame = {
                "observation.state": states_matrix[csv_idx],
                "action": actions_matrix[csv_idx],
                "instruction.text": self.task_text,
                "observation.meta.tool.psm1": self.psm1_tool,
                "observation.meta.tool.psm2": self.psm2_tool,
                "observation.images.endoscope.left": left_img,
            }

            if right_idx_arr is not None and right_paths:
                frame["observation.images.endoscope.right"] = self._load_image_cv2(
                    right_paths[right_idx_arr[out_idx]]
                )
            if psm1_idx_arr is not None and psm1_paths:
                frame["observation.images.wrist.right"] = self._load_image_cv2(
                    psm1_paths[psm1_idx_arr[out_idx]]
                )
            if psm2_idx_arr is not None and psm2_paths:
                frame["observation.images.wrist.left"] = self._load_image_cv2(
                    psm2_paths[psm2_idx_arr[out_idx]]
                )
            return frame

        pool = ThreadPoolExecutor(max_workers=NUM_WORKERS)

        # Seed the pipeline with initial batch of futures
        futures = {}
        for i in range(min(PIPELINE_DEPTH, total_frames)):
            futures[i] = pool.submit(_build_frame, i)

        for frame_idx in range(total_frames):
            # Progress every 500 frames
            if frame_idx > 0 and frame_idx % 500 == 0:
                avg_add = (t_add_total / frame_idx) * 1000
                self.log_message.emit(
                    f"    -> {frame_idx}/{total_frames} "
                    f"(avg add_frame={avg_add:.1f}ms)"
                )

            frame = futures.pop(frame_idx).result()

            next_idx = frame_idx + PIPELINE_DEPTH
            if next_idx < total_frames:
                futures[next_idx] = pool.submit(_build_frame, next_idx)

            t0 = time.time()
            dataset.add_frame(frame, task=self.task_text, timestamp=frame_idx * fps_inv)
            t_add_total += time.time() - t0

        pool.shutdown(wait=False)

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
        self.setGeometry(100, 100, 900, 700)
        
        self.source_path: Optional[Path] = None
        self.output_dir: Path = Path(DEFAULT_LEROBOT_HOME)
        self.worker: Optional[ConversionWorker] = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        
        # Title
        title = QLabel("DVRK → LeRobot v2.1 Converter")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #2c3e50; margin: 10px;")
        main_layout.addWidget(title)
        
        # Source Directory Selection
        source_group = QGroupBox("Source Data")
        source_layout = QHBoxLayout(source_group)
        
        source_layout.addWidget(QLabel("Data Directory:"))
        self.source_label = QLabel("No directory selected")
        self.source_label.setStyleSheet("color: #888; font-style: italic;")
        source_layout.addWidget(self.source_label, stretch=1)
        
        self.select_source_btn = QPushButton("Browse...")
        self.select_source_btn.clicked.connect(self._select_source_directory)
        self.select_source_btn.setStyleSheet("background-color: #3498db; color: white;")
        source_layout.addWidget(self.select_source_btn)
        
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
        self.dataset_name_edit = QLineEdit("dvrk_wound_closure")
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
        
        self.task_edit = QLineEdit("Wound Closure")
        self.task_edit.setPlaceholderText("Task description")
        meta_layout.addRow("Task Text:", self.task_edit)
        
        self.psm1_tool_edit = QLineEdit("Mega SutureCut Needle Driver")
        meta_layout.addRow("PSM1 Tool:", self.psm1_tool_edit)
        
        self.psm2_tool_edit = QLineEdit("Large Needle Driver")
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
        self.select_source_btn.setEnabled(False)
        self.select_output_btn.setEnabled(False)
        self.dataset_name_edit.setEnabled(False)
        self.task_edit.setEnabled(False)
        self.psm1_tool_edit.setEnabled(False)
        self.psm2_tool_edit.setEnabled(False)
        self.fps_spinbox.setEnabled(False)
        self.codec_combo.setEnabled(False)
        
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
            task_text=self.task_edit.text().strip(),
            psm1_tool=self.psm1_tool_edit.text().strip(),
            psm2_tool=self.psm2_tool_edit.text().strip(),
            fps=self.fps_spinbox.value(),
            resume_mode=resume_mode,
            skip_count=skip_count,
        )
        
        self.worker.progress.connect(self._on_progress)
        self.worker.log_message.connect(self._log)
        self.worker.episode_started.connect(self._on_episode_started)
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
    
    def _on_finished(self, success: bool, message: str):
        """Handle conversion completion"""
        # Re-enable UI
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.select_source_btn.setEnabled(True)
        self.select_output_btn.setEnabled(True)
        self.dataset_name_edit.setEnabled(True)
        self.task_edit.setEnabled(True)
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
