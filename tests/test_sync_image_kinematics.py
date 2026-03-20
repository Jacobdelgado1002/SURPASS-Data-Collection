import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from surpass_data_collection.scripts.sync_image_kinematics.sync_image_kinematics import (
    extract_timestamp_from_filename,
    load_image_timestamps,
    load_kinematics_data,
    find_nearest_kinematics,
    remove_outliers,
    process_episode_sync,
    CAMERA_CONFIGS,
    DEFAULT_CSV_FILENAME
)

class TestSyncImageKinematics:

    # -------------------------------------------------------------------------
    # extract_timestamp_from_filename Tests
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize("filename, expected", [
        ("frame1756826516968031906_left.jpg", 1756826516968031906),      # Old Format (left)
        ("frame1234_right.jpg", 1234),                                   # Old Format (snug bounds)
        ("frame000000_psm1_10_500.jpg", 10 * 1_000_000_000 + 500),       # New Format seconds / nanoseconds
        ("frame999_left_0_100.jpg", 100)                                 # New Format only ns
    ])
    def test_extract_timestamp_success(self, filename, expected):
        """Test timestamp parsing on both old and new regex patterns yields correct ns scale."""
        assert extract_timestamp_from_filename(filename) == expected

    @pytest.mark.parametrize("filename", [
        "frame_missing.jpg",            # Completely malformed
        "left_12345.jpg",               # Missing 'frame' prefix
        "frame1234center.jpg",          # Missing the underscore separator
        "frame1234_left.png",           # Not a jpg
        "1234_left.jpg"                 # Missing 'frame'
    ])
    def test_extract_timestamp_failure(self, filename):
        """Test extraction forcefully raises ValueError on unrecognized strings."""
        with pytest.raises(ValueError):
            extract_timestamp_from_filename(filename)

    # -------------------------------------------------------------------------
    # load_image_timestamps Tests
    # -------------------------------------------------------------------------
    def test_load_image_timestamps_success(self, tmp_path):
        """Test scanning image directories extracts & sorts identically to file count."""
        img_dir = tmp_path / "left_img_dir"
        img_dir.mkdir()

        # Unsorted creation chronological mix
        (img_dir / "frame3000_left.jpg").touch()
        (img_dir / "frame1000_left.jpg").touch()
        (img_dir / "frame2000_left.jpg").touch()
        (img_dir / "frame2000_right.jpg").touch()  # Should be skipped (wrong suffix)
        
        timestamps = load_image_timestamps(img_dir, "_left")
        
        assert len(timestamps) == 3
        
        # Verify strict temporal sorting
        assert timestamps[0] == ("frame1000_left.jpg", 1000)
        assert timestamps[1] == ("frame2000_left.jpg", 2000)
        assert timestamps[2] == ("frame3000_left.jpg", 3000)

    def test_load_image_timestamps_empty_or_missing(self, tmp_path):
        """Test os.scandir logic gracefully exits empty/missing directories."""
        assert load_image_timestamps(tmp_path / "fake", "_left") == []

        img_dir = tmp_path / "left_img_dir"
        img_dir.mkdir()
        assert load_image_timestamps(img_dir, "_left") == []

    # -------------------------------------------------------------------------
    # load_kinematics_data Tests
    # -------------------------------------------------------------------------
    def test_load_kinematics_existing_timestamp(self, tmp_path):
        """Test that dataframes with built-in 'time' or 'stamp' columns map correctly."""
        csv_file = tmp_path / DEFAULT_CSV_FILENAME
        df_orig = pd.DataFrame({"time_ns": [10, 20, 30], "x": [1, 2, 3]})
        df_orig.to_csv(csv_file, index=False)

        df = load_kinematics_data(csv_file)
        assert "timestamp_ns" in df.columns
        assert list(df["timestamp_ns"]) == [10, 20, 30]

    def test_load_kinematics_synthetic_generation(self, tmp_path):
        """Test that missing time columns enforce 30Hz synthetic ns distributions."""
        csv_file = tmp_path / DEFAULT_CSV_FILENAME
        df_orig = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}) # No time column
        df_orig.to_csv(csv_file, index=False)

        df = load_kinematics_data(csv_file)
        assert "timestamp_ns" in df.columns
        
        freq_ns = int(1e9 / 30)
        assert list(df["timestamp_ns"]) == [0, freq_ns, freq_ns * 2]

    def test_load_kinematics_failures(self, tmp_path):
        """Test expected loading failures raise the correct IO errors."""
        with pytest.raises(FileNotFoundError):
            load_kinematics_data(tmp_path / "missing.csv")

    # -------------------------------------------------------------------------
    # find_nearest_kinematics Tests
    # -------------------------------------------------------------------------
    def test_find_nearest_kinematics_success(self):
        """Test the np.searchsorted bidirectional gap measurement resolves appropriately."""
        image_timestamps = [
            ("frame10.jpg", 10),
            ("frame20.jpg", 20),
            ("frame30.jpg", 30)
        ]

        kinematics_df = pd.DataFrame({
            "timestamp_ns": [8, 18, 35] # Kinematics slightly offset
        })

        sync_df = find_nearest_kinematics(image_timestamps, kinematics_df)

        assert len(sync_df) == 3
        # Image 10 searches [8, 18, 35] -> closest is 8 (diff=2) -> index 0
        # Image 20 searches [8, 18, 35] -> closest is 18 (diff=2) -> index 1
        # Image 30 searches [8, 18, 35] -> closest is 35 (diff=5) -> index 2

        assert sync_df.loc[0, "kinematics_idx"] == 0
        assert sync_df.loc[1, "kinematics_idx"] == 1
        assert sync_df.loc[2, "kinematics_idx"] == 2

        assert sync_df.loc[0, "time_diff_ns"] == 2   # 10 - 8
        assert sync_df.loc[1, "time_diff_ns"] == 2   # 20 - 18
        assert sync_df.loc[2, "time_diff_ns"] == -5  # 30 - 35
        
    def test_find_nearest_kinematics_empty(self):
        """Test empty arrays cleanly return an empty sync dataframe."""
        df_empty = find_nearest_kinematics([], pd.DataFrame({"timestamp_ns": [1, 2, 3]}))
        assert df_empty.empty
        
        df_empty2 = find_nearest_kinematics([("img.jpg", 1)], pd.DataFrame({"timestamp_ns": pd.Series(dtype='int64')}))
        assert df_empty2.empty

    # -------------------------------------------------------------------------
    # remove_outliers Tests
    # -------------------------------------------------------------------------
    def test_remove_outliers_filtering(self):
        """Test Pandas absolute diff threshold bounds."""
        sync_df = pd.DataFrame({
            "time_diff_ms": [10.0, -15.0, 35.0, -45.0, 0.0]
        })

        # Threshold 30ms -> should keep 10.0, -15.0, 0.0
        filtered, removed = remove_outliers(sync_df, max_time_diff_ms=30.0)

        assert removed == 2
        assert len(filtered) == 3
        assert list(filtered["time_diff_ms"]) == [10.0, -15.0, 0.0]

    # -------------------------------------------------------------------------
    # process_episode_sync Tests
    # -------------------------------------------------------------------------
    def test_process_episode_sync_missing_data(self, tmp_path):
        """Test the master processing function safely aborts on uninitialized directories."""
        result = process_episode_sync(tmp_path)
        assert result["success"] is False
        assert "Image directory not found" in result["error"]

    def test_process_episode_sync_valid_chain(self, tmp_path):
        """Test the entirety of the pipeline chaining from start to finish via mocked disk."""
        ep_dir = tmp_path / "valid_episode"
        ep_dir.mkdir()
        
        img_dir = ep_dir / "left_img_dir"
        img_dir.mkdir()
        (img_dir / "frame1000_left.jpg").touch()
        (img_dir / "frame2000_left.jpg").touch()

        csv_file = ep_dir / DEFAULT_CSV_FILENAME
        pd.DataFrame({"time": [1000, 2005]}).to_csv(csv_file, index=False)

        # 30ms threshold covers 5ns (0.000005 ms) difference easily
        result = process_episode_sync(ep_dir, camera="left", max_time_diff_ms=30.0)

        assert result["success"] is True
        assert result["num_valid_images"] == 2
        assert result["outliers_removed"] == 0
        assert "frame1000_left.jpg" in result["valid_filenames"]
        assert len(result["sync_df"]) == 2
