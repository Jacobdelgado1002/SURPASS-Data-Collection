import os
from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import importlib.util

MODULE_PATH = Path(__file__).resolve().parent.parent / "src" / "surpass_data_collection" / "scripts" / "lerobot_conversion" / "accelerated-dvrk-lerobot-converter" / "dvrk_lerobot_converter_v2.1.py"

spec = importlib.util.spec_from_file_location("dvrk_lerobot_converter", MODULE_PATH)
converter_mod = importlib.util.module_from_spec(spec)
# We don't want it running full main execution on import, 
# although Python import inherently executes top-level code.
# The script has if __name__ == "__main__": so it's safe.
spec.loader.exec_module(converter_mod)

safe_normalize = converter_mod.safe_normalize
quat_inverse = converter_mod.quat_inverse
quat_multiply = converter_mod.quat_multiply
ensure_quat_continuity = converter_mod.ensure_quat_continuity
compute_action_hybrid_rel = converter_mod.compute_action_hybrid_rel
extract_timestamp = converter_mod.extract_timestamp
load_frames_from_dir = converter_mod.load_frames_from_dir
validate_episode = converter_mod.validate_episode
LEFT_IMG_DIR = converter_mod.LEFT_IMG_DIR
CSV_FILE = converter_mod.CSV_FILE

class TestDvrkLerobotConverter:

    # -------------------------------------------------------------------------
    # Quaternion / Math Operations Tests
    # -------------------------------------------------------------------------
    def test_safe_normalize_standard(self):
        """Test array normalization preserves directions and standardizes magnitude."""
        q = np.array([2.0, 0.0, 0.0, 0.0])
        q_norm = safe_normalize(q)
        np.testing.assert_array_almost_equal(q_norm, [1.0, 0.0, 0.0, 0.0])
        
        q2 = np.array([1.0, 1.0, 1.0, 1.0])
        q2_norm = safe_normalize(q2)
        np.testing.assert_array_almost_equal(q2_norm, [0.5, 0.5, 0.5, 0.5])

    def test_safe_normalize_zero(self):
        """Test normalization intercepts tiny floats/zeros turning them into Identity Quaternions."""
        q_zero = np.array([0.0, 0.0, 1e-10, 0.0])
        q_norm = safe_normalize(q_zero)
        # Fallback identity format
        np.testing.assert_array_almost_equal(q_norm, [0.0, 0.0, 0.0, 1.0])

    def test_quat_inverse(self):
        """Test quaternion inversion exclusively flips xyz geometries."""
        q = np.array([0.5, 0.5, 0.5, 0.5])
        q_inv = quat_inverse(q)
        np.testing.assert_array_almost_equal(q_inv, [-0.5, -0.5, -0.5, 0.5])

    def test_quat_multiply(self):
        """Test standard Hamiltonian multiplication matrix calculations."""
        # Two identity quaternions multiplied should yield identity
        q1 = np.array([0.0, 0.0, 0.0, 1.0])
        q2 = np.array([0.0, 0.0, 0.0, 1.0])
        q_out = quat_multiply(q1, q2)
        np.testing.assert_array_almost_equal(q_out, [0.0, 0.0, 0.0, 1.0])
        
        # Test 90 degree orthogonal rotation
        qa = np.array([1.0, 0.0, 0.0, 0.0])
        qb = np.array([0.0, 1.0, 0.0, 0.0])
        q_out2 = quat_multiply(qa, qb)
        np.testing.assert_array_almost_equal(q_out2, [0.0, 0.0, 1.0, 0.0])

    def test_ensure_quat_continuity(self):
        """Test dot product bounds check sign flipping for interpolation safety."""
        q_prev = np.array([0.0, 0.0, 0.0, 1.0])
        q_curr_same = np.array([0.0, 0.0, 0.0, 1.0])
        q_curr_opp = np.array([0.0, 0.0, 0.0, -1.0])
        
        res1 = ensure_quat_continuity(q_prev, q_curr_same)
        np.testing.assert_array_equal(res1, q_curr_same)
        
        res2 = ensure_quat_continuity(q_prev, q_curr_opp)
        np.testing.assert_array_equal(res2, [0.0, 0.0, 0.0, 1.0])  # Sign flipped

    def test_compute_action_hybrid_rel(self):
        """Test the relative positioning and quaternion rotation extraction loop (Action 16D array)."""
        # Baseline action Array representing [x,y,z, qx,qy,qz,qw, jaw] x 2 PSM arms
        action_t = np.array([
            0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 1.0,  1.0,  # PSM1
            1.0, 1.0, 1.0,  0.0, 0.0, 0.0, 1.0,  0.0   # PSM2
        ], dtype=np.float32)
        
        action_next = np.array([
            1.0, -1.0, 0.5,  0.0, 0.0, 0.0, 1.0,  1.0,  # PSM1 moves to 1, -1, 0.5 
            1.0,  1.5, 1.0,  0.0, 1.0, 0.0, 0.0,  0.0   # PSM2 moves +0.5 on Y, rotates on Y globally
        ], dtype=np.float32)
        
        rel_action = compute_action_hybrid_rel(action_t, action_next)
        
        # Validate PSM1 bounds (dx=1, dy=-1, dz=0.5, q_rel=identity, jaw=1)
        np.testing.assert_array_almost_equal(rel_action[0:3], [1.0, -1.0, 0.5])
        np.testing.assert_array_almost_equal(rel_action[3:7], [0.0, 0.0, 0.0, 1.0])
        assert rel_action[7] == 1.0
        
        # Validate PSM2 bounds (dx=0, dy=0.5, dz=0, q_rel=[0,1,0,0], jaw=0)
        np.testing.assert_array_almost_equal(rel_action[8:11], [0.0, 0.5, 0.0])
        np.testing.assert_array_almost_equal(rel_action[11:15], [0.0, 1.0, 0.0, 0.0])
        assert rel_action[15] == 0.0

    # -------------------------------------------------------------------------
    # Utility / File IO Tests
    # -------------------------------------------------------------------------
    def test_extract_timestamp(self):
        """Test timestamp extraction bounds."""
        assert extract_timestamp("frame100_left.jpg") == 100
        assert extract_timestamp("frame100_right_0_555.jpg") == 555
        
        with pytest.raises(ValueError):
            extract_timestamp("unrelated_file.jpg")

    def test_load_frames_from_dir(self, tmp_path):
        """Test FrameInfo namedtuple assembly via sorting and regex scanning."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        
        (img_dir / "frame100_left.jpg").touch()
        (img_dir / "frame50_left.jpg").touch()
        (img_dir / "badfile.jpg").touch()
        (img_dir / "unrelated.txt").touch()
        
        frames = load_frames_from_dir(img_dir)
        
        assert len(frames) == 2
        # Automatically sorted validation!
        assert frames[0].timestamp == 50
        assert frames[1].timestamp == 100
        assert "frame50" in str(frames[0].path)

    def test_trim_stationary_episodes_alignment(self, tmp_path):
        """Test the asynchronous alignment in trimmer maps 1:1 synced frames instead of raw frames."""
        # Setup mock directories
        ref_dir = tmp_path / "cautery_tissue1" / "session1"
        src_dir = tmp_path / "john_tissue1" / "session1"
        ref_left = ref_dir / LEFT_IMG_DIR
        src_left = src_dir / LEFT_IMG_DIR
        ref_left.mkdir(parents=True)
        src_left.mkdir(parents=True)
        
        # Reference (Raw) Video has 30 frames (0 to 29)
        for i in range(30):
            (ref_left / f"frame{i * 1000}_left.jpg").touch()
            
        # Source (Filtered) Video dropped the first 2 frames. It has 28 frames (2 to 29)
        for i in range(2, 30):
            (src_left / f"frame{i * 1000}_left.jpg").touch()
            
        # Timestamp array corresponding to 28 frames
        timestamps = [i * 1000 for i in range(2, 30)]
        # Movement happens entering index 10 and entering index 20 (meaning rows 10, 20 are where movement ends/starts)
        # That means start motion = 10, end motion = 20
        # If we pad perfectly:
        js = [0]*10 + [10]*10 + [20]*8
        # deltas over 27 transitions:
        # idx 0-8: 0
        # idx 9: 10
        # idx 10-18: 0
        # idx 19: 10
        # idx 20-26: 0
        
        # So trim_start = 9, trim_end = 20
        
        import pandas as pd
        df = pd.DataFrame({
            "timestamp": timestamps,
            "psm1_js[0]": js
        })
        df.to_csv(src_dir / CSV_FILE, index=False)
        
        class MockWorker:
            trim_threshold = 1.0
            def __init__(self):
                # Bind the isolated method to test it without instantiating the full worker
                self._trim_stationary_episodes = converter_mod.ConversionWorker._trim_stationary_episodes.__get__(self)
        
        worker = MockWorker()
        
        # Planned coordinates refer to Raw Video parameters.
        # Say the annotation mapped from raw frame 1 to raw frame 28.
        planned = [("ann.json", ref_dir, src_dir, "dst", 1, 28)]
        
        trimmed_planned, stats = worker._trim_stationary_episodes(planned)
        
        assert stats["trimmed"] == 1
        
        new_plan = trimmed_planned[0]
        # Exact reference bounds recovered! 11000ns = ref index 11. 22000ns = ref index 22.
        assert new_plan[4] == 11
        assert new_plan[5] == 22
