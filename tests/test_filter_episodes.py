import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Adjusting import path dynamically if run from another location
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from surpass_data_collection.scripts.sync_image_kinematics.filter_episodes import (
    find_episodes,
    find_all_camera_matches_vectorized,
    load_camera_timestamps,
    run_sync_analysis_direct,
    validate_episode_structure,
    write_filtered_kinematics,
    KINEMATIC_CSV_NAME,
)

class TestFilterEpisodes:

    # -------------------------------------------------------------------------
    # find_episodes Tests
    # -------------------------------------------------------------------------
    def test_find_episodes_empty_dir(self, tmp_path):
        """Test finding episodes in an empty directory returns an empty list."""
        episodes = find_episodes(tmp_path)
        assert episodes == []

    def test_find_episodes_missing_source(self):
        """Test finding episodes with a non-existent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            find_episodes("non_existent_folder_xyz")

    def test_find_episodes_valid_directory(self, tmp_path):
        """Test finding a valid episode folder (has left_img_dir and CSV)."""
        ep1 = tmp_path / "session_1"
        ep1.mkdir()
        (ep1 / "left_img_dir").mkdir()
        (ep1 / KINEMATIC_CSV_NAME).touch()

        # Invalid: missing CSV
        ep2 = tmp_path / "session_2"
        ep2.mkdir()
        (ep2 / "left_img_dir").mkdir()

        # Invalid: missing left_img_dir
        ep3 = tmp_path / "session_3"
        ep3.mkdir()
        (ep3 / KINEMATIC_CSV_NAME).touch()

        episodes = find_episodes(tmp_path)
        
        assert len(episodes) == 1
        assert episodes[0] == ep1

    # -------------------------------------------------------------------------
    # validate_episode_structure Tests
    # -------------------------------------------------------------------------
    def test_validate_episode_structure_all_valid(self, tmp_path):
        """Test validation when both images and kinematics are perfectly present."""
        ep_dir = tmp_path / "valid_ep"
        ep_dir.mkdir()
        img_dir = ep_dir / "left_img_dir"
        img_dir.mkdir()
        
        # Add a jpg
        (img_dir / "frame1234_left.jpg").touch()
        
        csv_file = ep_dir / KINEMATIC_CSV_NAME
        csv_file.touch()

        validation = validate_episode_structure(ep_dir)
        
        assert validation["has_left_images"] is True
        assert validation["has_kinematics"] is True
        assert validation["left_img_dir"] == img_dir
        assert validation["kinematics_file"] == csv_file

    @pytest.mark.parametrize("missing_element", ["jpgs", "csv", "img_dir"])
    def test_validate_episode_structure_invalid(self, tmp_path, missing_element):
        """Test validation handles missing components properly."""
        ep_dir = tmp_path / f"invalid_ep_{missing_element}"
        ep_dir.mkdir()
        
        img_dir = ep_dir / "left_img_dir"
        
        if missing_element != "img_dir":
            img_dir.mkdir()
            
        if missing_element != "jpgs" and missing_element != "img_dir":
            (img_dir / "frame1234_left.jpg").touch()
            
        if missing_element != "csv":
            (ep_dir / KINEMATIC_CSV_NAME).touch()

        validation = validate_episode_structure(ep_dir)

        if missing_element in ["img_dir", "jpgs"]:
            assert validation["has_left_images"] is False
            assert validation["left_img_dir"] is None
        
        if missing_element == "csv":
            assert validation["has_kinematics"] is False
            assert validation["kinematics_file"] is None

    # -------------------------------------------------------------------------
    # run_sync_analysis_direct Tests
    # -------------------------------------------------------------------------
    def test_run_sync_analysis_direct_mocked(self, tmp_path, monkeypatch):
        """Test the sync analysis wrapper correctly calls the internal process_episode_sync."""
        
        mock_result = {
            "success": True,
            "valid_filenames": ["a.jpg", "b.jpg"],
            "sync_df": pd.DataFrame(),
            "num_valid_images": 2,
            "outliers_removed": 0
        }

        # Mock the imported function directly in the filter_episodes module
        def mock_process(*args, **kwargs):
            return mock_result
            
        monkeypatch.setattr(
            "surpass_data_collection.scripts.sync_image_kinematics.filter_episodes.process_episode_sync", 
            mock_process
        )
        
        result = run_sync_analysis_direct(tmp_path, max_time_diff=30.0)
        assert result == mock_result

    # -------------------------------------------------------------------------
    # write_filtered_kinematics Tests
    # -------------------------------------------------------------------------
    def test_write_filtered_kinematics_success(self, tmp_path):
        """Test extracting precise rows from the original CSV into a filtered CSV."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()
        
        # 1. Create original kinematics CSV with 5 rows
        csv_path = src_dir / KINEMATIC_CSV_NAME
        df_orig = pd.DataFrame({
            "timestamp": [1000, 2000, 3000, 4000, 5000],
            "x": [1, 2, 3, 4, 5]
        })
        df_orig.to_csv(csv_path, index=False)
        
        validation = {"kinematics_file": csv_path}
        
        # 2. Create sync_df which dictates what to keep
        sync_df = pd.DataFrame({
            "image_filename": ["keep1.jpg", "keep2.jpg"],
            "kinematics_idx": [1, 3]  # Map to rows with x=2 and x=4
        })
        
        kept_filenames = ["keep1.jpg", "keep2.jpg"]
        
        write_filtered_kinematics(dest_dir, sync_df, validation, kept_filenames)
        
        dest_csv = dest_dir / KINEMATIC_CSV_NAME
        assert dest_csv.exists()
        
        # Verify output contents
        df_filtered = pd.read_csv(dest_csv)
        assert len(df_filtered) == 2
        assert list(df_filtered["x"].values) == [2, 4]
        assert list(df_filtered["timestamp"].values) == [2000, 4000]

    def test_write_filtered_kinematics_empty_inputs(self, tmp_path):
        """Test filtering gracefully exits when dataframes or lists are empty."""
        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()
        
        write_filtered_kinematics(dest_dir, pd.DataFrame(), {"kinematics_file": "fake"}, ["a.jpg"])
        assert not (dest_dir / KINEMATIC_CSV_NAME).exists()
        
        sync_df = pd.DataFrame({"image_filename": ["a.jpg"], "kinematics_idx": [0]})
        write_filtered_kinematics(dest_dir, sync_df, {"kinematics_file": "fake"}, [])
        assert not (dest_dir / KINEMATIC_CSV_NAME).exists()

    # -------------------------------------------------------------------------
    # load_camera_timestamps Tests
    # -------------------------------------------------------------------------
    def test_load_camera_timestamps_success(self, tmp_path):
        """Test timestamp extraction from secondary camera filename structures."""
        ep_dir = tmp_path / "ep"
        cam_dir = ep_dir / "right_img_dir"
        cam_dir.mkdir(parents=True)
        
        # Create fake images varying chronological orders
        # Using typical naming convention from generic datasets: frame{timestamp}_{suffix}
        (cam_dir / "frame100_right.jpg").touch()
        (cam_dir / "frame50_right.jpg").touch() 
        (cam_dir / "invalid_name.jpg").touch()
        (cam_dir / "frame200_wrong.jpg").touch()  # Missing expected '_right' suffix
        
        # NOTE: extraction regex logic falls back to extract_timestamp_from_filename.
        timestamps = load_camera_timestamps(ep_dir, "right_img_dir", "_right.jpg")
        
        assert len(timestamps) == 2
        # Automatically sorted validation!
        assert timestamps[0][1] == "frame50_right.jpg"
        assert timestamps[1][1] == "frame100_right.jpg"

    def test_load_camera_timestamps_missing_dir(self, tmp_path):
        """Test loading timestamps handles nonexistent directories safely."""
        timestamps = load_camera_timestamps(tmp_path, "fake_dir", "_left.jpg")
        assert timestamps == []

    # -------------------------------------------------------------------------
    # find_all_camera_matches_vectorized Tests
    # -------------------------------------------------------------------------
    def test_find_all_camera_matches_vectorized_perfect(self):
        """Test vectorized binary search for closest timestamp matches."""
        left_ts = np.array([1000, 2000, 3000], dtype=np.int64)
        cand_ts = np.array([900, 2005, 3050, 4000], dtype=np.int64)
        max_diff = 50.0  # Allow +/- 50 ns 
        
        indices, mask = find_all_camera_matches_vectorized(left_ts, cand_ts, max_diff)
        
        # 1000 -> closest is 900 (diff=100) -> Invalid
        # 2000 -> closest is 2005 (diff=5) -> Valid, index 1
        # 3000 -> closest is 3050 (diff=50) -> Valid, index 2
        
        assert len(indices) == 3
        # Valid mask should be [False, True, True]
        np.testing.assert_array_equal(mask, [False, True, True])
        
        # Indices array maps left_ts requests to cand_ts answers
        assert indices[1] == 1
        assert indices[2] == 2

    def test_find_all_camera_matches_vectorized_empty(self):
        """Test vectorized matching gracefully ignores empty candidate queries."""
        left_ts = np.array([1000, 2000], dtype=np.int64)
        cand_ts = np.array([], dtype=np.int64)
        
        indices, mask = find_all_camera_matches_vectorized(left_ts, cand_ts, max_diff_ns=50.0)
        
        # Should return arrays of length len(left_ts) filled with zeros
        np.testing.assert_array_equal(indices, np.array([0, 0], dtype=np.int64))
        np.testing.assert_array_equal(mask, np.array([False, False], dtype=bool))
