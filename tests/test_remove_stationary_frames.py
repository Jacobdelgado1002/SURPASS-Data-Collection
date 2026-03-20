import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from surpass_data_collection.scripts.post_processing.remove_stationary_frames import (
    discover_episodes,
    compute_deltas,
    find_trim_range,
    _sorted_images,
    trim_episode,
    run_remove_stationary_frames,
    KINEMATIC_CSV_NAME,
    MOTION_COLUMNS,
    CAMERA_DIRS
)

class TestRemoveStationaryFrames:

    # -------------------------------------------------------------------------
    # discover_episodes Tests
    # -------------------------------------------------------------------------
    def test_discover_episodes_success(self, tmp_path):
        """Test hierarchical traversal for datasets containing CSV targets."""
        ep1 = tmp_path / "tissue_1" / "1_grasp" / "episode_001"
        ep1.mkdir(parents=True)
        (ep1 / KINEMATIC_CSV_NAME).touch()

        ep2 = tmp_path / "tissue_1" / "2_dissect" / "episode_001"
        ep2.mkdir(parents=True)
        (ep2 / KINEMATIC_CSV_NAME).touch()

        # Missing CSV should be completely ignored
        empty_dir = tmp_path / "tissue_1" / "3_cut" / "episode_001"
        empty_dir.mkdir(parents=True)

        episodes = discover_episodes(tmp_path)
        
        assert len(episodes) == 2
        # Deterministic sorting guarantee
        assert episodes[0] == ep1
        assert episodes[1] == ep2

    # -------------------------------------------------------------------------
    # compute_deltas Tests
    # -------------------------------------------------------------------------
    def test_compute_deltas_success(self, tmp_path):
        """Test the C-engine pandas reader and np.einsum L2 norm logic produces rapid, exact arrays."""
        csv_path = tmp_path / KINEMATIC_CSV_NAME
        
        # 3 Rows of data. 2 of the 14 motion columns.
        # Row 1 -> Row 2: delta is (2-1)^2 + (4-2)^2 = 1 + 4 = 5 -> sqrt(5) = 2.236
        # Row 2 -> Row 3: delta is (5-2)^2 + (8-4)^2 = 9 + 16 = 25 -> sqrt(25) = 5
        df = pd.DataFrame({
            "timestamp": [0, 1, 2],
            "psm1_js[0]": [1, 2, 5],
            "psm1_js[1]": [2, 4, 8]
        })
        df.to_csv(csv_path, index=False)
        
        deltas, n_rows = compute_deltas(csv_path)
        
        assert n_rows == 3
        assert len(deltas) == 2
        
        np.testing.assert_almost_equal(deltas[0], np.sqrt(5))
        np.testing.assert_almost_equal(deltas[1], 5)

    def test_compute_deltas_no_motion_columns(self, tmp_path):
        """Test missing motion columns yields graceful (None, 0)."""
        csv_path = tmp_path / KINEMATIC_CSV_NAME
        df = pd.DataFrame({"unrelated_x": [1, 2], "unrelated_y": [3, 4]})
        df.to_csv(csv_path, index=False)
        
        deltas, n_rows = compute_deltas(csv_path)
        
        assert deltas is None
        assert n_rows == 0

    def test_compute_deltas_too_short(self, tmp_path):
        """Test dataframes with only 1 row automatically skip diff generation."""
        csv_path = tmp_path / KINEMATIC_CSV_NAME
        df = pd.DataFrame({"psm1_js[0]": [1]})
        df.to_csv(csv_path, index=False)
        
        deltas, n_rows = compute_deltas(csv_path)
        
        assert deltas is None
        assert n_rows == 1

    # -------------------------------------------------------------------------
    # find_trim_range Tests
    # -------------------------------------------------------------------------
    def test_find_trim_range_standard(self):
        """Test identification of moving bounds given clear leading/trailing zeros."""
        deltas = np.array([0.0, 0.0, 0.5, 0.8, 0.0, 0.0]) # Length 6 -> n_rows = 7
        n_rows = 7
        
        # Threshold: 0.1
        # Trims should be:
        # Start at index 2 (val 0.5)
        # End at index 3 + 1 = 4 (val 0.8 is the last movement)
        
        start, end = find_trim_range(deltas, n_rows, threshold=0.1)
        
        assert start == 2
        assert end == 4

    def test_find_trim_range_no_motion(self):
        """Test range captures entire sequence when zero motion surpasses threshold."""
        deltas = np.array([0.0, 0.01, 0.0, 0.01])
        n_rows = 5
        
        # Threshold 0.1 > all deltas
        start, end = find_trim_range(deltas, n_rows, threshold=0.1)
        
        assert start == 0
        assert end == 4

    # -------------------------------------------------------------------------
    # _sorted_images Tests
    # -------------------------------------------------------------------------
    def test_sorted_images(self, tmp_path):
        """Test images are efficiently captured while discarding misc files."""
        src = tmp_path / "images"
        src.mkdir()
        
        (src / "img2.jpg").touch()
        (src / "img1.png").touch()
        (src / "bad.txt").touch()
        (src / ".hidden.jpeg").touch()
        
        images = _sorted_images(str(src))
        
        assert len(images) == 3
        # Should be lexigraphically sorted
        assert images[0] == ".hidden.jpeg"
        assert images[1] == "img1.png"
        assert images[2] == "img2.jpg"

    def test_sorted_images_missing(self, tmp_path):
        """Test missing directories return empty arrays safely."""
        assert _sorted_images(str(tmp_path / "missing")) == []

    # -------------------------------------------------------------------------
    # trim_episode / Integration Tests
    # -------------------------------------------------------------------------
    def test_trim_episode_e2e_success(self, tmp_path):
        """Test end to end binary CSV rewriting and physical directory unlinking bounds."""
        ep_dir = tmp_path / "episode"
        ep_dir.mkdir()
        
        # Populate CSV: 5 total rows
        # Timestamps: 0, 1, 2, 3, 4
        # Movement happens entering row 2 (diff=5) and entering row 3 (diff=5)
        # Row 0 -> Row 1: delta 0
        # Row 1 -> Row 2: delta 5
        # Row 2 -> Row 3: delta 5
        # Row 3 -> Row 4: delta 0
        df = pd.DataFrame({
            "timestamp": [10, 20, 30, 40, 50],
            "psm1_js[0]": [0, 0, 5, 10, 10]
        })
        
        csv_path = ep_dir / KINEMATIC_CSV_NAME
        df.to_csv(csv_path, index=False)
        
        # Create standard camera directory with 5 images
        cam_dir = ep_dir / CAMERA_DIRS[0]
        cam_dir.mkdir()
        for i in range(5):
             (cam_dir / f"frame_{i:04d}.jpg").touch()
             
        # Expected Bounds:
        # deltas = [0, 5, 5, 0]
        # over threshold -> [False, True, True, False]
        # trim_start = 1
        # trim_end = 2 + 1 = 3
        # Kept rows: [1, 2, 3] -> 3 rows! (Indices 1 to 3 inclusive)
        
        stats = trim_episode(ep_dir, threshold=1.0, min_episode_length=2)
        
        assert stats["skipped"] is False
        assert stats["trimmed_length"] == 3
        assert stats["start_removed"] == 1
        assert stats["end_removed"] == 1  # 5 rows total. length 3. (1, 3) -> 1 removed start, 1 removed end.
        
        # Verify CSV was cleanly rebuilt in-place (binary string replacement)
        new_df = pd.read_csv(csv_path)
        assert len(new_df) == 3
        assert list(new_df["timestamp"]) == [20, 30, 40]
        
        # Verify physical unlinks resolved
        images = _sorted_images(str(cam_dir))
        assert len(images) == 3
        assert images[0] == "frame_0001.jpg"
        assert images[1] == "frame_0002.jpg"
        assert images[2] == "frame_0003.jpg"

    def test_trim_episode_too_short(self, tmp_path):
        """Test safeguards that intercept tiny bounds preventing data decimation."""
        ep_dir = tmp_path / "episode"
        ep_dir.mkdir()
        
        df = pd.DataFrame({
            "psm1_js[0]": [0, 0, 5, 0, 0]
        })
        df.to_csv(ep_dir / KINEMATIC_CSV_NAME, index=False)
        
        # Will result in a trim length of exactly 2.
        # Demand 10 minimum lengths
        stats = trim_episode(ep_dir, threshold=1.0, min_episode_length=10)
        
        assert stats["skipped"] is True
        assert "min 10" in stats["reason"]

    def test_run_remove_stationary_frames(self, tmp_path):
        """Test the master multiprocessor interface successfully spawns instances."""
        ep_dir = tmp_path / "ep"
        ep_dir.mkdir()
        df = pd.DataFrame({"psm1_js[0]": [0, 0, 5, 5, 0, 0]})
        df.to_csv(ep_dir / KINEMATIC_CSV_NAME, index=False)
        
        # Run sequential
        all_stats = run_remove_stationary_frames(
            tmp_path, threshold=1.0, min_episode_length=2, workers=1
        )
        
        assert len(all_stats) == 1
        assert all_stats[0]["skipped"] is False
