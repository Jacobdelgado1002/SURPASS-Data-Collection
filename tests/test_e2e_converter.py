import os
import sys
import json
import shutil
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from PIL import Image

# Import the module dynamically to handle the naming conventions
import importlib.util
MODULE_PATH = Path(__file__).resolve().parent.parent / "src" / "surpass_data_collection" / "scripts" / "lerobot_conversion" / "accelerated-dvrk-lerobot-converter" / "dvrk_lerobot_converter_v2.1.py"
spec = importlib.util.spec_from_file_location("dvrk_lerobot_converter", MODULE_PATH)
converter_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter_mod)

STATES_NAME = getattr(converter_mod, "STATES_NAME", [])
ACTIONS_NAME = getattr(converter_mod, "ACTIONS_NAME", [])
LEFT_IMG_DIR = converter_mod.LEFT_IMG_DIR
RIGHT_IMG_DIR = "right_img_dir" if not hasattr(converter_mod, 'RIGHT_IMG_DIR') else converter_mod.RIGHT_IMG_DIR
ENDO_PSM1_DIR = converter_mod.ENDO_PSM1_DIR
ENDO_PSM2_DIR = converter_mod.ENDO_PSM2_DIR
CSV_FILE = converter_mod.CSV_FILE


@pytest.fixture
def real_subset_dataset(tmp_path):
    """
    Generates a highly realistic, structurally identical dataset subset
    representing a 30-frame sequence of dual-arm robotic movement.
    """
    source_dir = tmp_path / "Data" / "tissues"
    annotations_dir = tmp_path / "Data" / "annotations"
    
    # 1. Setup paths resembling: Data/tissues/Jacob/Tissue#1/20260304-175604-460599
    session_name = "20260304-175604-460599"
    tissue_dir = source_dir / "Jacob" / "Tissue#1"
    session_dir = tissue_dir / session_name
    
    # Actually, the pipeline uses 'cautery' logic or source logic in slice_affordance.
    # We will just ensure the path matches what slice_affordance expects. 
    # v2.1 uses: {dataset_name}/{collector}/Tissue#{N}/{session} or cautery_tissue#{N}/{session}.
    # To accurately trigger the pipeline's regex layout detection (which extracts collector names)
    # we use the modern non-legacy format so slice_affordance detects them properly
    legacy_session_dir = source_dir / "Jacob" / "Tissue#1" / session_name
    for sub in [LEFT_IMG_DIR, RIGHT_IMG_DIR, ENDO_PSM1_DIR, ENDO_PSM2_DIR]:
        (legacy_session_dir / sub).mkdir(parents=True, exist_ok=True)
        
    num_frames = 30
    timestamps = [int(1e9 + i * 33333333) for i in range(num_frames)] # ~30fps in nanoseconds
    
    # 2. Generate Dummy JPEG RGB images
    # We must generate valid jpegs so ffmpeg can actually encode them later in the E2E pipeline
    endo_shape = (540, 960, 3)
    wrist_shape = (480, 640, 3)
    
    endo_img_arr = np.zeros(endo_shape, dtype=np.uint8)
    wrist_img_arr = np.zeros(wrist_shape, dtype=np.uint8)
    
    # Paint some simple color to prevent extreme compression algorithms from breaking
    endo_img_arr[:, :, 1] = 120
    wrist_img_arr[:, :, 2] = 200
    
    endo_img = Image.fromarray(endo_img_arr)
    wrist_img = Image.fromarray(wrist_img_arr)
    
    for i, ts in enumerate(timestamps):
        endo_img.save(legacy_session_dir / LEFT_IMG_DIR / f"frame{ts}_left.jpg", format="JPEG")
        endo_img.save(legacy_session_dir / RIGHT_IMG_DIR / f"frame{ts}_right.jpg", format="JPEG")
        wrist_img.save(legacy_session_dir / ENDO_PSM1_DIR / f"frame{ts}_psm1.jpg", format="JPEG")
        wrist_img.save(legacy_session_dir / ENDO_PSM2_DIR / f"frame{ts}_psm2.jpg", format="JPEG")
        
    # 3. Generate Valid Kinematic CSV
    # We must provide all columns defined in STATES_NAME and ACTIONS_NAME
    csv_data = {"timestamp": timestamps}
    for col in STATES_NAME + ACTIONS_NAME:
        # Create a moving trajectory (0 to 1) and some stationary tail
        # Frames 0-5: stationary at zero
        # Frames 5-25: linear movement
        # Frames 25-30: stationary
        trajectory = np.zeros(num_frames, dtype=np.float32)
        trajectory[5:25] = np.linspace(0.0, 1.0, 20)
        trajectory[25:] = 1.0
        
        # If it's a quaternion column, normalize it so pipeline logic doesn't panic
        if "q" in col.lower() or "rot" in col.lower() or "quaternion" in col.lower():
            trajectory = np.ones(num_frames, dtype=np.float32) * 0.5  # Vector length 1 across 4 axes approx
            
        csv_data[col] = trajectory
        
    df = pd.DataFrame(csv_data)
    df.to_csv(legacy_session_dir / CSV_FILE, index=False)
    
    # 4. Generate Annotation JSON
    ann_session_dir = annotations_dir / f"Jacob_tissue1_{session_name}"
    ann_json_dir = ann_session_dir / "annotation"
    ann_json_dir.mkdir(parents=True, exist_ok=True)
    
    annot_file = ann_json_dir / f"annot1.json"
    
    # Slicing from raw frame 2 to raw frame 28 (keeping some stationary padding to test trimming)
    annotation_payload = {
        "session": session_name,
        "filename": f"{session_name}.mp4",
        "action": "testing_action",
        "affordance_range": {"start": 2, "end": 28}
    }
    with open(annot_file, "w") as f:
        json.dump(annotation_payload, f)
        
    return source_dir, annotations_dir, legacy_session_dir


@pytest.fixture
def dummy_lerobot_home(tmp_path):
    """Temporary storage for huggingface datasets during conversion"""
    hf_home = tmp_path / "hf_home"
    hf_home.mkdir()
    # We mock the environment variable before importing lerobot heavily
    os.environ["HF_LEROBOT_HOME"] = str(hf_home)
    return hf_home


class TestEndToEndConverterPipeline:
    
    def test_e2e_full_pipeline_success(self, real_subset_dataset, dummy_lerobot_home):
        """
        Executes the full pipeline end-to-end, validating intermediate states
        and output dataset architectural integrity.
        """
        source_dir, annotations_dir, session_dir = real_subset_dataset
        
        output_dir = dummy_lerobot_home / "output"
        output_dir.mkdir()
        dataset_name = "test_e2e_dataset"
        
        cmd_args = [
            "dvrk_lerobot_converter_v2.1.py",
            "--source-dir", str(source_dir),
            "--annotations-dir", str(annotations_dir),
            "--output-dir", str(output_dir),
            "--dataset-name", dataset_name,
            "--fps", "30",
            # We enforce trim_threshold > 0 to natively execute Stage 2.5
            "--trim-threshold", "1e-5",
            # Ensure psm variables are definitively populated
            "--psm1-tool", "Permanent Cautery Hook",
            "--psm2-tool", "Prograsp Forceps",
            # Use a universally available fast codec for CI testing
            "--codec", "h264"
        ]
        
        with patch.object(sys, "argv", cmd_args):
            try:
                converter_mod.main()
            except SystemExit as e:
                if e.code != 0:
                    pytest.fail(f"Pipeline exited with error code {e.code}")

        # ==========================================================
        # VALIDATION PHASE
        # ==========================================================
        
        # 1. Validate Stage 1 (Filtered Cache)
        filtered_cache = output_dir / "_filtered_cache"
        assert filtered_cache.exists(), "Stage 1 failed to construct _filtered_cache"
        filtered_session = filtered_cache / "Jacob" / "Tissue#1" / session_dir.name
        assert (filtered_session / CSV_FILE).exists(), "_filtered_cache missing ee_csv.csv synchronization"
        
        # 2. Validate Final HuggingFace Architectural Output
        dataset_root = output_dir / dataset_name
        assert dataset_root.exists(), "Dataset root directory missing"
        assert (dataset_root / "meta" / "info.json").exists(), "LeRobot dataset missing info.json"
        
        videos_dir = dataset_root / "videos"
        assert videos_dir.exists(), "LeRobot dataset missing videos subpath"
        
        # 3. Validate Semantics Using LeRobot API
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        
        # Disable huggingface hub validation locally
        dataset = LeRobotDataset(dataset_name, root=dataset_root)
        
        # Verify length constraints (Original: 26 sliced frames. Trimmed stationary tails: ~20 frames)
        assert len(dataset) > 10, "Dataset violently over-trimmed frames."
        assert len(dataset) < 30, "Dataset failed to slice or trim."
        
        # Verify frame integrities and data shapes
        first_frame = dataset[0]
        
        # Verify essential tensor keys exist
        expected_keys = [
            "observation.state", 
            "action", 
            "action_hybrid_relative",
            "observation.images.endoscope.left",
            "observation.images.endoscope.right",
            "observation.images.wrist.left",
            "observation.images.wrist.right"
        ]
        for key in expected_keys:
            assert key in first_frame, f"Missing key '{key}' in dataset frame"
            
        # Verify no NaNs leaked into the tensors
        assert not np.isnan(first_frame["observation.state"].numpy()).any(), "NaN found in observation.state"
        assert not np.isnan(first_frame["action"].numpy()).any(), "NaN found in action"
        assert not np.isnan(first_frame["action_hybrid_relative"].numpy()).any(), "NaN found in action_hybrid_relative"
        
        # Assert the dummy image was actually read properly by `av` without corruption
        assert first_frame["observation.images.endoscope.left"].shape == (3, 540, 960), "Endoscope tensor shape mismatch"
        assert first_frame["observation.images.wrist.left"].shape == (3, 480, 640), "Wrist tensor shape mismatch"
        
        # Ensure timestamps are strictly monotonic across the episodes
        timestamps = dataset.hf_dataset["timestamp"]
        assert np.all(np.diff(timestamps) > 0), "Timestamps are not strictly monotonic!"

    def test_e2e_missing_camera_aborts_securely(self, real_subset_dataset, dummy_lerobot_home, capsys):
        """
        Validates the newly patched structural integrity check ensuring the
        dataset rejects compilation if a camera channel goes offline.
        """
        source_dir, annotations_dir, session_dir = real_subset_dataset
        
        # Introduce catastrophic missing camera defect (Drop the right endoscope)
        import shutil
        shutil.rmtree(session_dir / RIGHT_IMG_DIR)
        
        output_dir = dummy_lerobot_home / "output2"
        output_dir.mkdir()
        dataset_name = "test_broken_dataset"
        
        cmd_args = [
            "dvrk_lerobot_converter_v2.1.py",
            "--source-dir", str(source_dir),
            "--annotations-dir", str(annotations_dir),
            "--output-dir", str(output_dir),
            "--dataset-name", dataset_name,
            "--fps", "30",
        ]
        
        with patch.object(sys, "argv", cmd_args):
            with pytest.raises(SystemExit) as e:
                converter_mod.main()
            assert e.value.code == 1, "Pipeline failed to exit securely with code 1"
            
        # Verify the warning was successfully caught
        captured = capsys.readouterr()
        assert "No episodes planned" in captured.err, "Pipeline missed critical planning ValueError"
