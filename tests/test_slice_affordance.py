import json
import os
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from surpass_data_collection.scripts.post_processing.slice_affordance import (
    extract_timestamp,
    _frame_key,
    list_sorted_frames,
    find_sessions,
    read_annotation_jsons,
    plan_episodes,
)

class TestSliceAffordance:

    # -------------------------------------------------------------------------
    # extract_timestamp Tests
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize("filename, expected", [
        ("frame1756826516968031906_left.jpg", 1756826516968031906),      # Old Format (left)
        ("frame1234_right.jpg", 1234),                                   # Old Format (snug bounds)
        ("frame000000_psm1_10_500.jpg", 10 * 1_000_000_000 + 500),       # New Format seconds / nanoseconds
        ("frame999_left_0_100.jpg", 100)                                 # New Format only ns
    ])
    def test_extract_timestamp_success(self, filename, expected):
        """Test timestamp parsing on both old and new regex patterns yields correct ns scale."""
        assert extract_timestamp(filename) == expected

    @pytest.mark.parametrize("filename", [
        "frame_missing.jpg",            # Completely malformed
        "left_12345.jpg",               # Missing 'frame' prefix
        "image1234_center.jpg",         # Missing 'frame' keyword entirely
        "1234_left.jpg"                 # Missing 'frame'
    ])
    def test_extract_timestamp_failure(self, filename):
        """Test extraction forcefully raises ValueError on unrecognized strings."""
        with pytest.raises(ValueError):
            extract_timestamp(filename)

    # -------------------------------------------------------------------------
    # _frame_key Tests
    # -------------------------------------------------------------------------
    def test_frame_key_numeric_extraction(self):
        """Test the sorting key extraction accurately finds trailing integers."""
        assert _frame_key("frame10.jpg") == 10
        assert _frame_key("frame02_left.jpg") == 2
        assert _frame_key("frame1756.jpg") == 1756
        assert _frame_key("not_a_frame.jpg") == "not_a_frame.jpg"

    # -------------------------------------------------------------------------
    # list_sorted_frames Tests
    # -------------------------------------------------------------------------
    def test_list_sorted_frames_success(self, tmp_path):
        """Test that list_sorted_frames scans directories and sorts using _frame_key."""
        src_dir = tmp_path / "left_img_dir"
        src_dir.mkdir()
        
        # Chronologically randomized touch
        (src_dir / "frame100_left_10_0.jpg").touch()
        (src_dir / "frame20_left.jpg").touch()
        (src_dir / "frame150_right.jpg").touch()  # Should be ignored (wrong suffix)
        (src_dir / "frame5_left_0_0.jpg").touch()
        
        frames = list_sorted_frames(src_dir, "_left.jpg")
        
        assert len(frames) == 3
        # Strict temporal order validation
        assert frames[0] == "frame5_left_0_0.jpg"
        assert frames[1] == "frame20_left.jpg"
        assert frames[2] == "frame100_left_10_0.jpg"

    def test_list_sorted_frames_missing_dir(self, tmp_path):
        """Test missing directory gracefully returns empty array."""
        assert list_sorted_frames(tmp_path / "missing", "_left.jpg") == []

    # -------------------------------------------------------------------------
    # find_sessions Tests
    # -------------------------------------------------------------------------
    def test_find_sessions_patterns(self, tmp_path):
        """Test regex generator matching yields correct parsed identifiers."""
        post_dir = tmp_path / "post_process"
        post_dir.mkdir()
        
        # 1. Valid Old Format
        old_dir = post_dir / "cautery_tissue1_recordingABC_left_video"
        old_dir.mkdir()
        
        # 2. Valid New Format
        new_dir = post_dir / "john_tissue5_20261111"
        new_dir.mkdir()
        
        # 3. Invalid Format
        invalid_dir = post_dir / "some_random_bad_format_session"
        invalid_dir.mkdir()
        
        sessions = list(find_sessions(post_dir))
        
        assert len(sessions) == 2
        
        # Since os.iterdir isn't guaranteed deterministic order, cross-check them:
        session_strings = [str(s[0].name) for s in sessions]
        
        for s in sessions:
            path, tissue, session_name, collector = s
            if path.name == "cautery_tissue1_recordingABC_left_video":
                assert tissue == 1
                assert session_name == "recordingABC"
                assert collector is None
            elif path.name == "john_tissue5_20261111":
                assert tissue == 5
                assert session_name == "20261111"
                assert collector == "john"
            else:
                pytest.fail(f"Unexpected session mapped: {path.name}")

    def test_find_sessions_missing_dir(self, tmp_path):
        """Test generator exits safely on missing dir."""
        sessions = list(find_sessions(tmp_path / "fake"))
        assert len(sessions) == 0

    # -------------------------------------------------------------------------
    # read_annotation_jsons Tests
    # -------------------------------------------------------------------------
    def test_read_annotation_jsons_filtering(self, tmp_path):
        """Test recursive JSON discovery while ignoring _prompt.json exclusion clauses."""
        ann_dir = tmp_path / "annotation"
        ann_dir.mkdir()
        
        (ann_dir / "action_1.json").write_text("{}")
        (ann_dir / "bad_prompt.json").write_text("{}")        # Should be excluded
        (ann_dir / "action_2.JSON").write_text("{}")          # Case insensitive
        (ann_dir / "not_json.txt").write_text("{}")
        
        sub_dir = ann_dir / "nested"
        sub_dir.mkdir()
        (sub_dir / "action_3.json").write_text("{}")
        (sub_dir / "nested_prompt.json").write_text("{}")     # Should be excluded
        
        jsons = list(read_annotation_jsons(ann_dir))
        
        assert len(jsons) == 3
        names = [j.name.lower() for j in jsons]
        assert "action_1.json" in names
        assert "action_2.json" in names
        assert "action_3.json" in names

    # -------------------------------------------------------------------------
    # plan_episodes Tests
    # -------------------------------------------------------------------------
    def test_plan_episodes_success(self, tmp_path):
        """Full end-to-end extraction planner test building mock filesystems."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        
        annotations_dir = tmp_path / "annotations"
        annotations_dir.mkdir()
        
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        
        # 1. Create a "New format" valid annotation session
        # {collector}_tissue{N}_{timestamp}
        sess_name = "john_tissue2_sessionABC"
        ann_sess_dir = annotations_dir / sess_name
        ann_sess_dir.mkdir()
        
        ann_json_dir = ann_sess_dir / "annotation"
        ann_json_dir.mkdir()
        
        # Write valid JSON mapping affordance range and action name
        json_data = {
            "action": "grasp",
            "affordance_range": {"start": 50, "end": 100}
        }
        (ann_json_dir / "annot1.json").write_text(json.dumps(json_data))
        
        # Write valid JSON mapping another action (dissect)
        json_data2 = {
            "action": "dissect",
            "affordance_range": {"start": 200, "end": 450}
        }
        (ann_json_dir / "annot2.json").write_text(json.dumps(json_data2))
        
        # 2. Setup Source Dataset Directory mapping to simulate successful resolution
        # New format relies on reference session: source / collector / Tissue#N / session
        ref_sess_dir = source_dir / "john" / "Tissue#2" / "sessionABC"
        ref_sess_dir.mkdir(parents=True)
        
        # plan_episodes checks source_dataset_dir fallback chains
        # We tell plan_episodes that source_dataset_dir == source_dir
        
        episodes = plan_episodes(source_dir, annotations_dir, out_dir, source_dataset_dir=source_dir)
        
        assert len(episodes) == 2
        
        # Format: (ann_path, ref_sess_dir, src_sess_dir, dst_base, start, end)
        
        # Grasp assertion
        ep1 = [e for e in episodes if e[3].parent.name == "1_grasp"][0]
        assert ep1[1] == ref_sess_dir  # ref_sess_dir
        assert ep1[2] == ref_sess_dir  # src_sess_dir
        assert ep1[3].name == "episode_001"
        assert ep1[4] == 50   # Start
        assert ep1[5] == 100  # End
        
        # Dissect assertion
        ep2 = [e for e in episodes if e[3].parent.name == "2_dissect"][0]
        assert ep2[3].name == "episode_001"
        assert ep2[4] == 200
        assert ep2[5] == 450
