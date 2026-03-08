#!/usr/bin/env python
"""
SURPASS Dataset Validation Script

Validates LeRobot datasets produced by the SURPASS pipeline
(dvrk_lerobot_converter_v2.1.py) for structural correctness, data quality,
and training readiness.

Checks 8 categories:
    1. Directory structure (videos/, meta/, data/chunk-*)
    2. Metadata files (info.json, episodes.jsonl, tasks.jsonl, episodes_stats.jsonl)
    3. Dataset info (FPS, robot_type, splits)
    4. Feature schema (cameras, state, action, tool metadata)
    5. Video files (count, openability, resolution, frame count)
    6. Episode consistency (parquet ↔ metadata alignment, task labels)
    7. Data quality (NaN/Inf, quaternion norms, action std, episode lengths)
    8. LeRobot load test (full dataset load + sample frame retrieval)

Usage:
    python validate_surpass.py <dataset_path>
    python validate_surpass.py <dataset_path> --verbose
"""

import argparse
import io
import json
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Force UTF-8 stdout/stderr so emoji prints don't crash on Windows.
# ---------------------------------------------------------------------------
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    else:
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="replace"
        )
except Exception:
    pass

# Suppress FFmpeg log noise
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["FFMPEG_HIDE_BANNER"] = "1"

# ---------------------------------------------------------------------------
# Hardcoded repo-id for LeRobot load test
# ---------------------------------------------------------------------------
SURPASS_REPO_ID: str = "jacobdelgado1002/SURPASS_Cholecystectomy"


# ============================================================================
# Data classes
# ============================================================================
class ValidationLevel(Enum):
    """Severity levels for validation results."""

    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    SUCCESS = "SUCCESS"


@dataclass
class ValidationResult:
    """Single validation finding."""

    level: ValidationLevel
    category: str
    message: str
    details: Optional[str] = None


@dataclass
class ValidationReport:
    """Aggregated validation report."""

    results: List[ValidationResult] = field(default_factory=list)
    dataset_path: Optional[Path] = None

    @property
    def error_count(self) -> int:
        return sum(1 for r in self.results if r.level == ValidationLevel.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for r in self.results if r.level == ValidationLevel.WARNING)

    @property
    def info_count(self) -> int:
        return sum(1 for r in self.results if r.level == ValidationLevel.INFO)

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.results if r.level == ValidationLevel.SUCCESS)

    @property
    def is_compliant(self) -> bool:
        return self.error_count == 0


# ============================================================================
# Expected schema produced by dvrk_lerobot_converter_v2.1.py
# ============================================================================

# 16-dimensional: 7 pose + 1 jaw per arm × 2 arms
STATE_DIM: int = 16
ACTION_DIM: int = 16

# Indices of quaternion (x, y, z, w) components within 16-dim vectors.
# PSM1: indices 3-6, PSM2: indices 11-14
QUAT_SLICES: List[slice] = [slice(3, 7), slice(11, 15)]

EXPECTED_FEATURES: Dict[str, Dict[str, Any]] = {
    "observation.images.endoscope.left": {"dtype": "video"},
    "observation.images.endoscope.right": {"dtype": "video"},
    "observation.images.wrist.left": {"dtype": "video"},
    "observation.images.wrist.right": {"dtype": "video"},
    "observation.state": {"dtype": "float32", "shape_len": STATE_DIM},
    "action": {"dtype": "float32", "shape_len": ACTION_DIM},
    "action_hybrid_relative": {"dtype": "float32", "shape_len": ACTION_DIM},
    "observation.meta.tool.psm1": {"dtype": "string"},
    "observation.meta.tool.psm2": {"dtype": "string"},
    "instruction.text": {"dtype": "string"},
}

MIN_FPS: int = 20
MIN_RESOLUTION: tuple = (480, 480)
MIN_EPISODE_LENGTH: int = 10
QUAT_NORM_TOL: float = 0.05  # tolerance for ||q|| ≈ 1.0


# ============================================================================
# Validator
# ============================================================================
class SurpassDatasetValidator:
    """Validates a LeRobot dataset produced by the SURPASS dVRK pipeline.

    Args:
        dataset_path: Absolute path to the dataset root directory.
        verbose: If True, print SUCCESS and INFO results in addition
                 to ERROR and WARNING.
    """

    def __init__(self, dataset_path: Path, verbose: bool = False) -> None:
        self.dataset_path: Path = Path(dataset_path)
        self.verbose: bool = verbose
        self.report: ValidationReport = ValidationReport(
            dataset_path=self.dataset_path
        )

        if not self.dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {self.dataset_path}")

    # ------------------------------------------------------------------
    # Result helpers
    # ------------------------------------------------------------------

    def _add(
        self,
        level: ValidationLevel,
        category: str,
        message: str,
        details: Optional[str] = None,
    ) -> None:
        """Append a result and optionally print it.

        Args:
            level: Severity of the finding.
            category: Grouping label (e.g. "Directory Structure").
            message: One-line description.
            details: Optional extra context shown in verbose mode.
        """
        result = ValidationResult(level, category, message, details)
        self.report.results.append(result)
        if self.verbose or level in (ValidationLevel.ERROR, ValidationLevel.WARNING):
            self._print_result(result)

    def _print_result(self, result: ValidationResult) -> None:
        """Pretty-print a single result with colour and emoji."""
        symbols = {
            ValidationLevel.ERROR: "❌",
            ValidationLevel.WARNING: "⚠️",
            ValidationLevel.INFO: "ℹ️",
            ValidationLevel.SUCCESS: "✅",
        }
        colours = {
            ValidationLevel.ERROR: "\033[91m",
            ValidationLevel.WARNING: "\033[93m",
            ValidationLevel.INFO: "\033[94m",
            ValidationLevel.SUCCESS: "\033[92m",
        }
        reset = "\033[0m"
        sym = symbols[result.level]
        col = colours[result.level]
        print(f"{col}{sym} [{result.level.value}] {result.category}: {result.message}{reset}")
        if result.details and self.verbose:
            print(f"    Details: {result.details}")

    # ------------------------------------------------------------------
    # 1. Directory Structure
    # ------------------------------------------------------------------

    def validate_directory_structure(self) -> None:
        """Check that top-level LeRobot v2.1 directories and data chunks exist."""
        cat = "Directory Structure"

        for dir_name in ("videos", "meta", "data"):
            path = self.dataset_path / dir_name
            if path.is_dir():
                self._add(ValidationLevel.SUCCESS, cat, f"'{dir_name}/' exists")
            else:
                self._add(ValidationLevel.ERROR, cat, f"Required directory '{dir_name}/' not found")

        # Check chunk + parquet files inside data/
        data_dir = self.dataset_path / "data"
        if data_dir.is_dir():
            chunks = sorted(data_dir.glob("chunk-*"))
            if not chunks:
                self._add(ValidationLevel.ERROR, cat, "No chunk-* directories in data/")
            else:
                parquets = []
                for c in chunks:
                    parquets.extend(c.glob("episode_*.parquet"))
                if parquets:
                    self._add(
                        ValidationLevel.SUCCESS, cat,
                        f"Found {len(parquets)} parquet file(s) in {len(chunks)} chunk(s)",
                    )
                else:
                    self._add(ValidationLevel.ERROR, cat, "No episode_*.parquet files found in chunks")

    # ------------------------------------------------------------------
    # 2. Metadata Files
    # ------------------------------------------------------------------

    def validate_metadata_files(self) -> None:
        """Verify required metadata files exist and are parseable JSON/JSONL."""
        cat = "Metadata Files"
        meta_dir = self.dataset_path / "meta"
        if not meta_dir.is_dir():
            self._add(ValidationLevel.ERROR, cat, "meta/ directory missing — skipping metadata checks")
            return

        required = ["info.json", "episodes.jsonl", "tasks.jsonl", "episodes_stats.jsonl"]
        for fname in required:
            fpath = meta_dir / fname
            if not fpath.exists():
                self._add(ValidationLevel.ERROR, cat, f"Required file '{fname}' not found")
                continue

            # Quick parse test
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    if fname.endswith(".jsonl"):
                        lines = [json.loads(line) for line in f if line.strip()]
                        self._add(
                            ValidationLevel.SUCCESS, cat,
                            f"'{fname}' parseable ({len(lines)} record(s))",
                        )
                    else:
                        json.load(f)
                        self._add(ValidationLevel.SUCCESS, cat, f"'{fname}' parseable")
            except (json.JSONDecodeError, ValueError) as exc:
                self._add(ValidationLevel.ERROR, cat, f"'{fname}' is invalid JSON: {exc}")

    # ------------------------------------------------------------------
    # 3. Dataset Info
    # ------------------------------------------------------------------

    def validate_info_json(self) -> None:
        """Validate info.json fields: FPS, robot_type, and splits."""
        cat = "Dataset Info"
        info_path = self.dataset_path / "meta" / "info.json"
        if not info_path.exists():
            return

        try:
            with open(info_path, "r", encoding="utf-8") as f:
                info: Dict[str, Any] = json.load(f)
        except json.JSONDecodeError:
            return  # already reported in metadata check

        # FPS
        fps = info.get("fps")
        if fps is not None:
            if fps < MIN_FPS:
                self._add(
                    ValidationLevel.WARNING, cat,
                    f"FPS ({fps}) below recommended minimum ({MIN_FPS})",
                )
            else:
                self._add(ValidationLevel.SUCCESS, cat, f"FPS = {fps}")

        # Robot type
        rtype = info.get("robot_type")
        if rtype is None:
            self._add(ValidationLevel.WARNING, cat, "robot_type not specified")
        elif rtype != "dvrk":
            self._add(
                ValidationLevel.WARNING, cat,
                f"robot_type is '{rtype}', expected 'dvrk'",
            )
        else:
            self._add(ValidationLevel.SUCCESS, cat, "robot_type = 'dvrk'")

        # Splits
        splits = info.get("splits", {})
        if not splits:
            self._add(ValidationLevel.WARNING, cat, "No splits defined in info.json")
        else:
            for split_name in ("train", "val", "test"):
                if split_name in splits:
                    self._add(ValidationLevel.SUCCESS, cat, f"Split '{split_name}' defined: {splits[split_name]}")
                else:
                    self._add(ValidationLevel.WARNING, cat, f"Standard split '{split_name}' not defined")

            # SURPASS-specific: perfect / recovery
            if "perfect" in splits:
                self._add(ValidationLevel.SUCCESS, cat, f"Split 'perfect' defined: {splits['perfect']}")
            else:
                self._add(ValidationLevel.INFO, cat, "No 'perfect' split defined")

            if "recovery" in splits:
                self._add(ValidationLevel.SUCCESS, cat, f"Split 'recovery' defined: {splits['recovery']}")
            else:
                self._add(ValidationLevel.INFO, cat, "No 'recovery' split — consider adding recovery episodes")

        # Features (delegate detailed check to validate_feature_schema)
        if "features" in info:
            self._validate_feature_schema(info["features"])

    # ------------------------------------------------------------------
    # 4. Feature Schema
    # ------------------------------------------------------------------

    def _validate_feature_schema(self, features: Dict[str, Any]) -> None:
        """Validate the feature schema matches expected SURPASS converter output.

        Args:
            features: The 'features' dict from info.json.
        """
        cat = "Feature Schema"

        for feat_name, expected in EXPECTED_FEATURES.items():
            if feat_name not in features:
                self._add(ValidationLevel.ERROR, cat, f"Missing expected feature '{feat_name}'")
                continue

            feat_info = features[feat_name]

            # Dtype check
            actual_dtype = feat_info.get("dtype")
            if actual_dtype != expected["dtype"]:
                self._add(
                    ValidationLevel.ERROR, cat,
                    f"'{feat_name}' dtype is '{actual_dtype}', expected '{expected['dtype']}'",
                )
            else:
                self._add(ValidationLevel.SUCCESS, cat, f"'{feat_name}' dtype correct ({actual_dtype})")

            # Shape check for numeric features
            if "shape_len" in expected and "shape" in feat_info:
                shape = feat_info["shape"]
                if isinstance(shape, (list, tuple)) and len(shape) >= 1:
                    actual_dim = shape[0]
                    if actual_dim != expected["shape_len"]:
                        self._add(
                            ValidationLevel.ERROR, cat,
                            f"'{feat_name}' dim is {actual_dim}, expected {expected['shape_len']}",
                        )

        # Check for unexpected features (informational)
        extra = set(features.keys()) - set(EXPECTED_FEATURES.keys())
        if extra:
            self._add(
                ValidationLevel.INFO, cat,
                f"Extra features not in expected schema: {', '.join(sorted(extra))}",
            )

    # ------------------------------------------------------------------
    # 5. Video Files
    # ------------------------------------------------------------------

    def validate_video_files(self) -> None:
        """Check video file count, openability, FPS, and resolution."""
        cat = "Video Files"
        videos_dir = self.dataset_path / "videos"
        if not videos_dir.is_dir():
            return

        mp4s = sorted(videos_dir.glob("**/*.mp4"))
        if not mp4s:
            self._add(ValidationLevel.ERROR, cat, "No .mp4 files found in videos/")
            return

        self._add(ValidationLevel.SUCCESS, cat, f"Found {len(mp4s)} video file(s)")

        # Expected count = episodes × 4 cameras
        episodes_path = self.dataset_path / "meta" / "episodes.jsonl"
        if episodes_path.exists():
            with open(episodes_path, "r", encoding="utf-8") as f:
                n_episodes = sum(1 for line in f if line.strip())
            expected_count = n_episodes * 4
            if len(mp4s) != expected_count:
                self._add(
                    ValidationLevel.WARNING, cat,
                    f"Expected {expected_count} videos ({n_episodes} episodes × 4 cameras), found {len(mp4s)}",
                )
            else:
                self._add(ValidationLevel.SUCCESS, cat, f"Video count matches ({n_episodes} episodes × 4 cameras)")

        # Sample up to 4 videos for quality checks
        sample = mp4s[: min(4, len(mp4s))]
        for vpath in sample:
            self._check_single_video(vpath)

    def _check_single_video(self, video_path: Path) -> None:
        """Open a single video and check FPS, resolution, and frame count.

        Args:
            video_path: Path to an .mp4 file.
        """
        cat = "Video Quality"
        rel = video_path.relative_to(self.dataset_path)
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                self._add(ValidationLevel.ERROR, cat, f"Cannot open {rel}")
                return
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if n_frames == 0:
                self._add(ValidationLevel.ERROR, cat, f"{rel} has 0 frames")
            elif fps < MIN_FPS:
                self._add(ValidationLevel.WARNING, cat, f"{rel} FPS={fps:.1f} < {MIN_FPS}")
            elif h < MIN_RESOLUTION[0] or w < MIN_RESOLUTION[1]:
                self._add(ValidationLevel.WARNING, cat, f"{rel} resolution {w}×{h} below {MIN_RESOLUTION}")
            else:
                self._add(ValidationLevel.SUCCESS, cat, f"{rel}: {w}×{h} @ {fps:.0f}fps, {n_frames} frames")
        except Exception as exc:
            self._add(ValidationLevel.ERROR, cat, f"Error reading {rel}: {exc}")

    # ------------------------------------------------------------------
    # 6. Episode Consistency
    # ------------------------------------------------------------------

    def validate_episode_consistency(self) -> None:
        """Cross-check episode counts between metadata and parquet files, and validate task labels."""
        cat = "Episode Consistency"

        # Count parquet files
        data_dir = self.dataset_path / "data"
        parquet_count = 0
        if data_dir.is_dir():
            for chunk in data_dir.glob("chunk-*"):
                parquet_count += sum(1 for _ in chunk.glob("episode_*.parquet"))

        # Count episodes in metadata
        ep_path = self.dataset_path / "meta" / "episodes.jsonl"
        meta_count = 0
        if ep_path.exists():
            with open(ep_path, "r", encoding="utf-8") as f:
                meta_count = sum(1 for line in f if line.strip())

        if meta_count == 0 and parquet_count == 0:
            self._add(ValidationLevel.ERROR, cat, "No episodes found in metadata or data/")
            return

        if meta_count != parquet_count:
            self._add(
                ValidationLevel.ERROR, cat,
                f"Episode count mismatch: {meta_count} in episodes.jsonl vs {parquet_count} parquet files",
            )
        else:
            self._add(ValidationLevel.SUCCESS, cat, f"Episode count consistent: {meta_count}")

        # Validate task labels from tasks.jsonl
        tasks_path = self.dataset_path / "meta" / "tasks.jsonl"
        if tasks_path.exists():
            try:
                tasks: List[str] = []
                with open(tasks_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            record = json.loads(line)
                            task_str = record.get("task", "")
                            if task_str:
                                tasks.append(task_str)

                if tasks:
                    self._add(
                        ValidationLevel.SUCCESS, cat,
                        f"Found {len(tasks)} task label(s): {', '.join(sorted(set(tasks)))}",
                    )
                else:
                    self._add(ValidationLevel.WARNING, cat, "tasks.jsonl has no task labels")
            except Exception as exc:
                self._add(ValidationLevel.ERROR, cat, f"Error parsing tasks.jsonl: {exc}")

    # ------------------------------------------------------------------
    # 7. Data Quality
    # ------------------------------------------------------------------

    def validate_data_quality(self) -> None:
        """Sample parquet files and check for NaN/Inf, quaternion norms, action std, and episode lengths."""
        cat = "Data Quality"

        data_dir = self.dataset_path / "data"
        if not data_dir.is_dir():
            return

        parquets = sorted(data_dir.glob("chunk-*/episode_*.parquet"))
        if not parquets:
            return

        # Sample up to 10 parquets distributed across the dataset
        step = max(1, len(parquets) // 10)
        sample_paths = parquets[::step]

        all_states: List[np.ndarray] = []
        all_actions: List[np.ndarray] = []
        episode_lengths: List[int] = []
        nan_inf_episodes: List[str] = []
        quat_bad_episodes: List[str] = []

        for ppath in sample_paths:
            try:
                df = pd.read_parquet(ppath, engine="pyarrow")
            except Exception as exc:
                self._add(ValidationLevel.ERROR, cat, f"Cannot read {ppath.name}: {exc}")
                continue

            episode_lengths.append(len(df))
            ep_label = ppath.stem

            # --- Extract state and action columns ---
            state_cols = [c for c in df.columns if c.startswith("observation.state")]
            action_cols = [c for c in df.columns if c == "action" or c.startswith("action.")]

            # Try to get the state/action arrays
            state_arr = self._extract_tensor_column(df, "observation.state", STATE_DIM)
            action_arr = self._extract_tensor_column(df, "action", ACTION_DIM)

            # NaN / Inf check
            has_bad = False
            if state_arr is not None:
                all_states.append(state_arr)
                if np.any(~np.isfinite(state_arr)):
                    nan_inf_episodes.append(ep_label)
                    has_bad = True
            if action_arr is not None:
                all_actions.append(action_arr)
                if np.any(~np.isfinite(action_arr)):
                    if not has_bad:
                        nan_inf_episodes.append(ep_label)

            # Quaternion norm check on state
            if state_arr is not None:
                for qs in QUAT_SLICES:
                    quats = state_arr[:, qs]
                    norms = np.linalg.norm(quats, axis=1)
                    bad_mask = np.abs(norms - 1.0) > QUAT_NORM_TOL
                    if np.any(bad_mask):
                        pct = 100.0 * bad_mask.sum() / len(norms)
                        quat_bad_episodes.append(f"{ep_label}({pct:.1f}%)")

        # --- Report NaN/Inf ---
        if nan_inf_episodes:
            self._add(
                ValidationLevel.ERROR, cat,
                f"NaN/Inf found in {len(nan_inf_episodes)} sampled episode(s)",
                f"Episodes: {', '.join(nan_inf_episodes[:5])}",
            )
        else:
            self._add(ValidationLevel.SUCCESS, cat, f"No NaN/Inf in {len(sample_paths)} sampled episode(s)")

        # --- Report quaternion norms ---
        if quat_bad_episodes:
            self._add(
                ValidationLevel.WARNING, cat,
                f"Quaternion norm ≠ 1.0 (±{QUAT_NORM_TOL}) in: {', '.join(quat_bad_episodes[:5])}",
            )
        else:
            self._add(ValidationLevel.SUCCESS, cat, "Quaternion norms ≈ 1.0 in sampled episodes")

        # --- Report action std ---
        if all_actions:
            stacked = np.vstack(all_actions)
            per_dim_std = np.std(stacked, axis=0)
            zero_dims = int(np.sum(per_dim_std < 1e-8))
            overall_std = float(np.mean(per_dim_std))
            if overall_std < 1e-6:
                self._add(
                    ValidationLevel.WARNING, cat,
                    f"Action std is near zero ({overall_std:.2e}) — stationary trimming may not have worked",
                )
            elif zero_dims > 0:
                self._add(
                    ValidationLevel.WARNING, cat,
                    f"{zero_dims}/{ACTION_DIM} action dimensions have near-zero std",
                )
            else:
                self._add(
                    ValidationLevel.SUCCESS, cat,
                    f"Action std healthy (mean={overall_std:.4f} across {len(all_actions)} sampled episodes)",
                )

        # --- Report episode lengths ---
        if episode_lengths:
            min_len = min(episode_lengths)
            max_len = max(episode_lengths)
            mean_len = np.mean(episode_lengths)
            if min_len < MIN_EPISODE_LENGTH:
                self._add(
                    ValidationLevel.WARNING, cat,
                    f"Shortest sampled episode has {min_len} frames (< {MIN_EPISODE_LENGTH})",
                )
            else:
                self._add(
                    ValidationLevel.SUCCESS, cat,
                    f"Episode lengths: min={min_len}, max={max_len}, mean={mean_len:.0f}",
                )

    def _extract_tensor_column(
        self, df: pd.DataFrame, col_name: str, expected_dim: int
    ) -> Optional[np.ndarray]:
        """Extract a tensor-valued column from a parquet DataFrame.

        Handles both flat column layout (``col_name.0``, ``col_name.1``, …)
        and nested list layout (single ``col_name`` column of lists).

        Args:
            df: DataFrame from a single episode parquet file.
            col_name: Base name of the feature (e.g. "observation.state").
            expected_dim: Expected width of the tensor.

        Returns:
            2-D numpy array of shape (n_frames, expected_dim), or None
            if the column is not present.
        """
        # Case 1: column exists directly and contains list/array values
        if col_name in df.columns:
            sample = df[col_name].iloc[0]
            if isinstance(sample, (list, np.ndarray)):
                return np.stack(df[col_name].values)

        # Case 2: flattened columns like observation.state.0 … .15
        flat_cols = [f"{col_name}.{i}" for i in range(expected_dim)]
        if all(c in df.columns for c in flat_cols):
            return df[flat_cols].values.astype(np.float32)

        return None

    # ------------------------------------------------------------------
    # 8. LeRobot Load Test
    # ------------------------------------------------------------------

    def validate_lerobot_load(self) -> None:
        """Attempt to load the dataset with LeRobotDataset and retrieve a sample frame."""
        cat = "LeRobot Load Test"
        try:
            # Delay import so the rest of the script works even without lerobot
            os.environ["HF_LEROBOT_HOME"] = str(self.dataset_path.parent)
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            dataset = LeRobotDataset(
                repo_id=SURPASS_REPO_ID,
                root=str(self.dataset_path),
                video_backend="pyav",
            )
            self._add(ValidationLevel.SUCCESS, cat, "LeRobotDataset loaded successfully")

            # Frame count
            n_frames = len(dataset)
            self._add(ValidationLevel.SUCCESS, cat, f"Dataset contains {n_frames} frames")

            # Sample frame retrieval
            if n_frames > 0:
                sample = dataset[0]
                keys_present = list(sample.keys())
                self._add(
                    ValidationLevel.SUCCESS, cat,
                    f"Sample frame[0] has {len(keys_present)} keys",
                    f"Keys: {', '.join(sorted(keys_present))}",
                )
        except ImportError:
            self._add(
                ValidationLevel.WARNING, cat,
                "lerobot package not installed — skipping load test",
            )
        except Exception as exc:
            self._add(
                ValidationLevel.ERROR, cat,
                f"Failed to load dataset: {exc}",
            )

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run_validation(self) -> ValidationReport:
        """Execute all validation checks in sequence.

        Returns:
            The completed ValidationReport.
        """
        print(f"\n{'=' * 60}")
        print("SURPASS Dataset Validation")
        print(f"Dataset: {self.dataset_path}")
        print(f"{'=' * 60}\n")

        checks = [
            ("🔍 Directory structure", self.validate_directory_structure),
            ("📁 Metadata files", self.validate_metadata_files),
            ("📊 Dataset info & features", self.validate_info_json),
            ("🎬 Video files", self.validate_video_files),
            ("📝 Episode consistency", self.validate_episode_consistency),
            ("🧪 Data quality", self.validate_data_quality),
            ("🤖 LeRobot load test", self.validate_lerobot_load),
        ]
        for label, fn in checks:
            print(f"{label}...")
            fn()
            print()

        return self.report

    def print_summary(self) -> None:
        """Print a colour-coded summary grouped by category."""
        r = self.report
        print(f"{'=' * 60}")
        print("VALIDATION SUMMARY")
        print(f"{'=' * 60}")
        print(f"\n  ✅ Success : {r.success_count}")
        print(f"  ℹ️  Info    : {r.info_count}")
        print(f"  ⚠️  Warning : {r.warning_count}")
        print(f"  ❌ Error   : {r.error_count}")

        # Group by category for errors/warnings
        categories: Dict[str, List[ValidationResult]] = {}
        for res in r.results:
            categories.setdefault(res.category, []).append(res)

        if r.error_count:
            print("\n🚨 Critical Issues (must fix):")
            for cat, items in categories.items():
                errs = [i for i in items if i.level == ValidationLevel.ERROR]
                if errs:
                    print(f"\n  {cat}:")
                    for e in errs:
                        print(f"    • {e.message}")
                        if e.details:
                            print(f"      → {e.details}")

        if r.warning_count:
            print("\n⚠️  Warnings (should fix):")
            for cat, items in categories.items():
                warns = [i for i in items if i.level == ValidationLevel.WARNING]
                if warns:
                    print(f"\n  {cat}:")
                    for w in warns:
                        print(f"    • {w.message}")
                        if w.details:
                            print(f"      → {w.details}")

        # Verdict
        print(f"\n{'=' * 60}")
        if r.is_compliant:
            print("✅ Dataset PASSED validation.")
            if r.warning_count:
                print(f"   Consider addressing {r.warning_count} warning(s).")
        else:
            print(f"❌ Dataset FAILED validation with {r.error_count} error(s).")
            print("   Fix the issues above and re-run.")
        print(f"{'=' * 60}\n")


# ============================================================================
# CLI
# ============================================================================
def main() -> int:
    """Entry point for the SURPASS dataset validation CLI.

    Returns:
        0 if the dataset passes, 1 otherwise.
    """
    parser = argparse.ArgumentParser(
        description="Validate a SURPASS dVRK LeRobot dataset for correctness and training readiness.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "dataset_path", type=Path,
        help="Path to the LeRobot dataset root directory.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show all results including SUCCESS and INFO.",
    )
    args = parser.parse_args()

    try:
        validator = SurpassDatasetValidator(
            dataset_path=args.dataset_path, verbose=args.verbose
        )
        validator.run_validation()
        validator.print_summary()
        return 0 if validator.report.is_compliant else 1
    except Exception as exc:
        print(f"\n❌ Validation failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
