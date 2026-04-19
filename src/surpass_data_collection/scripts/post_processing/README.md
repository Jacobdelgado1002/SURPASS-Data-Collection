# Post-Processing Scripts

This directory contains scripts for reformatting and organizing sliced surgical robotics datasets into training-ready formats.

## Overview

After synchronization and action-based slicing, datasets need to be reformatted to ensure consistency and compatibility with machine learning frameworks. These scripts handle:

- **Timestamp normalization**: Converting absolute nanosecond timestamps to relative seconds
- **Frame renaming**: Sequential renaming with consistent naming conventions
- **Folder organization**: Restructuring tissue directories for clean dataset organization
- **Action-based slicing**: Extracting action segments from full surgical sessions using JSON annotations

---

## Directory Contents

```
post_processing/
├── reformat_data.py               # Normalize timestamps and rename frames
├── slice_affordance.py            # Slice sessions into action-based episodes
├── analyze_trim_threshold.py      # Diagnostic tool to calculate and visualize trim thresholds
├── remove_stationary_frames.py    # Script to trim leading/trailing stationary frames from sliced episodes
└── README.md                      # This file
```

---

## Scripts

### reformat_data.py

**Purpose**: Normalize timestamps and standardize frame naming in sliced datasets.

**Key Features**:
- Timestamp normalization: Converts nanosecond timestamps to relative seconds (starting at 0.0)
- Frame renaming: Standardizes image names to `frame000000_{camera}.jpg` format
- Tissue folder renaming: Renames timestamp-style folders to `{index}_{action_name}`
- Parallel processing: Efficient batch processing of multiple episodes
- Idempotent: Safe to re-run without data corruption

**Use Cases**:
- After slicing data into episodes
- Before training machine learning models
- When preparing data for LeRobot or other frameworks

**Command-Line Interface**:

```bash
# Basic usage - normalize all episodes
python reformat_data.py --data-path dataset_sliced

# Rename tissue folders
python reformat_data.py --data-path dataset_sliced \
    --rename-folders --new-name cholecystectomy

# Parallel processing with 12 workers
python reformat_data.py --data-path dataset_sliced --workers 12

# Only normalize timestamps (skip frame renaming)
python reformat_data.py --data-path dataset_sliced --timestamps-only

# Only rename frames (skip timestamp normalization)
python reformat_data.py --data-path dataset_sliced --frames-only

# Dry run to preview changes
python reformat_data.py --data-path dataset_sliced --dry-run
```

**Programmatic Usage**:

```python
from reformat_data import run_reformat_data
from pathlib import Path

# Full reformatting
rows, images = run_reformat_data(
    base_dir=Path("dataset_sliced"),
    workers=8,
    normalize_timestamps=True,
    normalize_frames=True,
    rename_folders=True,
    new_name="cholecystectomy"
)

print(f"Normalized {rows} timestamp rows, renamed {images} images")
```

**Input Structure**:
```
dataset_sliced/
└── tissue_1/
    ├── 20251217-150034-281409/    # Timestamp folder
    │   ├── episode_001/
    │   │   ├── left_img_dir/
    │   │   │   └── frame17068...906_left.jpg  # Absolute timestamp
    │   │   ├── right_img_dir/
    │   │   ├── endo_psm1/
    │   │   ├── endo_psm2/
    │   │   └── ee_csv.csv          # Nanosecond timestamps
    │   └── episode_002/
    └── 20251217-160045-381510/
```

**Output Structure**:
```
dataset_sliced/
└── tissue_1/
    ├── 1_cholecystectomy/         # Renamed folder
    │   ├── episode_001/
    │   │   ├── left_img_dir/
    │   │   │   └── frame000000_left.jpg  # Sequential naming
    │   │   ├── right_img_dir/
    │   │   ├── endo_psm1/
    │   │   ├── endo_psm2/
    │   │   └── ee_csv.csv          # Relative seconds (0.0, 0.033, ...)
    │   └── episode_002/
    └── 2_cholecystectomy/
```

**Processing Details**:

*Timestamp Normalization*:
- First timestamp becomes 0.0
- Subsequent timestamps converted to seconds: `(current_ns - first_ns) / 1e9`
- Precision: 9 decimal places (nanosecond accuracy preserved)
- Headers preserved automatically

*Frame Renaming*:
- Two-pass strategy to avoid name collisions
- Natural ordering maintained (sorted by name or mtime)
- Format: `frame{index:06d}_{camera}{extension}`
- Example: `frame000042_left.jpg`, `frame000042_psm1.jpg`

*Folder Renaming*:
- Deterministic lexicographic sorting
- Sequential numbering starting at 1
- Format: `{index}_{name}` (e.g., `1_cholecystectomy`, `2_cholecystectomy`)

**Options**:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data-path` | Path | Required | Root directory containing tissue folders |
| `--workers` | int | CPU count | Number of parallel worker processes |
| `--rename-folders` | flag | False | Rename tissue timestamp folders |
| `--new-name` | str | cholecystectomy | Suffix for renamed folders |
| `--timestamps-only` | flag | False | Only normalize timestamps |
| `--frames-only` | flag | False | Only rename frames |
| `--sort-by` | str | name | Frame sorting method ('name' or 'mtime') |
| `--dry-run` | flag | False | Preview without modifying files |

**Performance Notes**:
- Uses ProcessPoolExecutor for parallel episode processing
- Recommended workers: 4-12 depending on CPU cores
- Timestamp normalization: ~1000 rows/second
- Frame renaming: ~100 images/second
- Memory efficient: Streams CSV reading/writing

---

### analyze_trim_threshold.py

**Purpose**: A diagnostic tool used to determine the optimal movement threshold for trimming stationary frames at the beginning and end of episodes.

**Key Features**:
- Evaluates kinematic data across multiple threshold values.
- Plots total movement and proportion of trimmed frames to help identify the optimal noise threshold.
- Aids in distinguishing between actual intentional robot actions and noise jitter.
- Can run over multiple sessions to provide dataset-wide recommendations.

**Use Cases**:
- Tuning the stationary trimming parameters for the dataset.
- Identifying anomalous episodes with continuous motion.

**Programmatic Usage**:

```python
from analyze_trim_threshold import analyze_directory

# Analyze a directory with multiple threshold values
results = analyze_directory("filtered_data", thresholds=[1e-5, 1e-4, 1e-3, 1e-2])
print(results)
```

**Command-Line Interface**:
```bash
python analyze_trim_threshold.py /path/to/data
```

---

### remove_stationary_frames.py

**Purpose**: Trims stationary (non-moving) frames from the beginning and end of each sliced episode to ensure the dataset strictly contains actionable movements.

**Key Features**:
- Calculates position and orientation deltas from the kinematics CSV.
- Automatically detects and trims leading and trailing stationary blocks based on a configured `theta` threshold.
- Directly modifies the input directory in-place (removes trimmed images and CSV rows).
- Designed for speed with minimal memory footprint.

**Use Cases**:
- Post-processing affordance slices before packaging them into LeRobot datasets.
- Cleaning up padding or delays caused by human operator reaction time.

**Programmatic Usage**:

```python
from remove_stationary_frames import process_all_episodes
from pathlib import Path

# Trims stationary frames from all episodes within a dataset directory
stats = process_all_episodes(Path("dataset_sliced"), threshold=1e-4)

print(f"Total Trimmed: {stats['trimmed_episodes']} / {stats['total_episodes']}")
```

---

### slice_affordance.py

**Purpose**: Provide annotation parsing and episode planning utilities for downstream conversion pipelines.

**Key Features**:

- Reads JSON annotations (e.g., `action_001.json`) containing `affordance_range`
- Maps frame indices to sequential episode IDs based on action types
- Validates source and reference directories
- Provides programmatic API (`plan_episodes`) to yield matched `(start, end)` frame bounds

**Use Cases**:

- Integrating action annotations into the `dvrk_lerobot_converter_v2.1` pipeline.
- Sorting frames and extracting timestamp data natively.

**Programmatic Usage**:

```python
from slice_affordance import plan_episodes
from pathlib import Path

# Plan episodes from the annotation directory mapped to raw/filtered data
planned = plan_episodes(
    post_dir=Path("post_process"),
    cautery_dir=Path("cautery"),
    out_dir=Path("dataset_sliced"),
    source_dataset_dir=Path("filtered_data")
)

# Returns a list of tuples:
# (annotation_path, ref_session_dir, src_session_dir, dst_base_dir, start_frame, end_frame)
for item in planned:
    ann, ref, src, dst, s, e = item
    print(f"Planned range: {s}-{e} to {dst.name}")
```

**Expected Data Structure Overview**:

*Annotation Data* (post_process):
```
post_process/
└── cautery_tissue1_session_name_left_video/
    └── annotation/
        ├── action_001.json   # Contains affordance_range
        ├── action_002.json
        └── ...
```

*JSON Annotation Format*:
```json
{
    "action": "grasp",
    "affordance_range": {
        "start": 150,
        "end": 350
    }
}
```

**Action Mapping**:
Actions are mapped to numbered subdirectories for consistent ordering:
```python
ACTION_SUBDIRS = {
    "grasp": "1_grasp",
    "dissect": "2_dissect",
}
```

*Note: This script was refactored to remove command-line orchestration and physical file-copying mechanics. It now operates strictly as an in-memory planning module.*

---

## Typical Workflow

The recommended processing sequence for post-processing:

```bash
# Step 1: Filter and synchronize data (prerequisite)
python ../sync_image_kinematics/filter_episodes.py /raw_data /filtered_data

# Step 2: Slice into action-based episodes
# Currently typically driven by integration in dvrk_zarr_to_lerobot.py instead of directly

# Step 3: Trim stationary frames
python remove_stationary_frames.py dataset_sliced --threshold 1e-4

# Step 4: Normalize timestamps and frame names
python reformat_data.py \
    --data-path dataset_sliced \
    --rename-folders \
    --new-name cholecystectomy \
    --workers 8
```

---

## Configuration

### Episode Discovery

Both scripts discover episodes automatically by searching for directories containing `ee_csv.csv`. This allows flexibility in directory structures while ensuring consistent processing.

### Parallel Processing

Parallelism is implemented at multiple levels:
- **Episode-level**: Multiple episodes processed simultaneously (ProcessPoolExecutor)
- **File-level**: Files within episode copied in parallel (ThreadPoolExecutor)
- **Safe defaults**: CPU count used if not specified
- **Recommended settings**: 4-8 workers for local processing

### Frame Sorting

Frames can be sorted by:
- **name** (default): Lexicographic/natural ordering
- **mtime**: File modification time (chronological)

Use modification time sorting if filenames don't reflect capture order.

---

## Troubleshooting

### Common Issues

**Issue**: Empty CSV files after slicing
- **Cause**: Frame index mapping misalignment between filtered and reference datasets
- **Solution**: Ensure reference dataset matches source dataset structure

**Issue**: Missing frames in sliced episodes
- **Cause**: Timestamp alignment tolerance too strict
- **Solution**: Check that source dataset has required timestamps; verify filtering didn't remove critical frames

**Issue**: Slow copying performance
- **Cause**: Using file copy across different filesystems
- **Solution**: Use `--hardlink` flag if source and destination on same filesystem

**Issue**: Timestamp normalization produces negative values
- **Cause**: Corrupted or out-of-order timestamp data
- **Solution**: Validate source CSV timestamps are monotonically increasing

### Best Practices

- **Always use hardlinks** when possible (10-100x faster)
- **Filter before slicing** to reduce dataset size early
- **Validate annotations** before running full pipeline
- **Use dry-run** to preview changes on large datasets
- **Monitor disk space** when copying large datasets

---

## Dependencies

```bash
# Required packages
pip install numpy pandas tqdm

# Optional for visualization (slice_affordance.py)
pip install matplotlib
```

---

## Additional Notes

### Data Integrity

Both scripts are designed with data safety in mind:
- **Idempotent operations**: Safe to re-run
- **Atomic writes**: Temporary files used, then renamed
- **Validation checks**: Input validation before processing
- **Error handling**: Graceful degradation on failure

### Memory Efficiency

- CSV processing uses streaming (low memory footprint)
- Image processing is file-based (no bulk loading)
- Parallel workers limited to prevent memory exhaustion

### Extensibility

To add new camera modalities or action types:

1. Update `CAMERA_MODALITIES` or `ACTION_SUBDIRS` constants
2. Ensure directory structure matches expectations
3. No other code changes needed (automatic discovery)

---

## See Also

- **Main src README**: `../README.md`
- **Synchronization scripts**: `../sync_image_kinematics/README.md`
- **Video processing**: `../video_processing/README.md`
