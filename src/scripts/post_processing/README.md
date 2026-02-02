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
├── reformat_data.py         # Normalize timestamps and rename frames
├── slice_affordance.py      # Slice sessions into action-based episodes
└── README.md                # This file
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

### slice_affordance.py

**Purpose**: Slice surgical robot session data into action-based episodes using JSON annotations.

**Key Features**:
- Timestamp-based alignment between reference and source datasets
- Multi-modal data slicing (4 camera views + kinematic CSV)
- Parallel processing for efficiency (episode-level and file-level)
- Optional frame normalization after slicing
- Hardlink support for fast copying
- Binary search for frame alignment

**Use Cases**:
- Extracting action segments from full surgical sessions
- Slicing filtered datasets based on annotations
- Creating action-specific training sets
- Organizing data by surgical task type (grasp, dissect, etc.)

**Command-Line Interface**:

```bash
# Basic slicing from raw cautery data
python slice_affordance.py

# Slice from filtered dataset
python slice_affordance.py --source_dataset_dir filtered_data \
    --out_dir sliced_episodes

# Dry run to preview planned episodes
python slice_affordance.py --dry_run

# Use hardlinks for fast copying (same filesystem required)
python slice_affordance.py --hardlink

# Parallel processing with 8 episode workers, 16 copy workers
python slice_affordance.py --episode-workers 8 --workers 16

# Automatically reformat frame names after slicing
python slice_affordance.py --reformat
```

**Expected Data Structure**:

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

*Reference Dataset* (for timestamps):
```
cautery/
└── cautery_tissue#1/
    └── session_name/
        ├── left_img_dir/
        │   └── frame1756826516968031906_left.jpg
        ├── right_img_dir/
        ├── endo_psm1/
        ├── endo_psm2/
        └── ee_csv.csv
```

*Source Dataset* (to slice - may be filtered):
```
source_dataset/
└── tissue_1/
    └── session_name/
        ├── left_img_dir/
        ├── right_img_dir/
        ├── endo_psm1/
        ├── endo_psm2/
        └── ee_csv.csv
```

**Output Structure**:
```
dataset_sliced/
└── tissue_1/
    ├── 1_grasp/              # Mapped from "grasp" action
    │   ├── episode_001/
    │   │   ├── left_img_dir/
    │   │   ├── right_img_dir/
    │   │   ├── endo_psm1/
    │   │   ├── endo_psm2/
    │   │   └── ee_csv.csv
    │   └── episode_002/
    └── 2_dissect/            # Mapped from "dissect" action
        └── episode_001/
```

**Processing Pipeline**:

1. **Discovery**: Find post-process annotation directories
2. **Parsing**: Extract affordance_range (start/end frame indices) from JSON
3. **Mapping**: Map frame indices to timestamps using reference dataset
4. **Alignment**: Find corresponding frames in source via binary search
5. **Copying**: Copy sliced data (images + CSV) to organized output
6. **Normalization** (optional): Run reformat_data.py on output

**Timestamp-Based Alignment**:

The script ensures semantic consistency when slicing from filtered datasets:
- Reference dataset provides ground truth timestamps
- Source dataset may have different frame counts (due to filtering)
- Binary search finds nearest matching timestamps
- Alignment handles gaps and removed frames correctly

**Action Mapping**:

Actions are mapped to numbered subdirectories for consistent ordering:
```python
ACTION_SUBDIRS = {
    "grasp": "1_grasp",
    "dissect": "2_dissect",
}
```

**Options**:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--post_process_dir` | Path | post_process | Annotation directory |
| `--cautery_dir` | Path | cautery | Reference dataset (timestamps) |
| `--source_dataset_dir` | Path | cautery_dir | Dataset to slice |
| `--out_dir` | Path | dataset_sliced | Output directory |
| `--episode-workers` | int | 4 | Parallel episodes to process |
| `--workers` | int | 8 | File copy workers per episode |
| `--hardlink` | flag | False | Use hardlinks instead of copying |
| `--reformat` | flag | False | Run reformat_data.py after slicing |
| `--dry_run` | flag | False | Preview planned episodes |

**CSV Slicing**:
- Preserves headers automatically
- Uses 0-indexed row ranges
- Handles missing timestamps gracefully
- Clamps indices to valid ranges

**Performance Optimization**:
- Episode-level parallelism: Process multiple episodes simultaneously
- File-level parallelism: Copy files within episode in parallel
- Hardlink support: Instant "copying" on same filesystem
- Binary search: O(log n) frame lookups
- Streaming CSV: Memory-efficient row filtering

**Common Use Cases**:

```bash
# Typical workflow: slice from filtered data
python sync_image_kinematics/filter_episodes.py /raw /filtered
python slice_affordance.py --source_dataset_dir filtered --hardlink

# Slice and immediately reformat
python slice_affordance.py --reformat

# High-performance slicing
python slice_affordance.py --episode-workers 12 --workers 24 --hardlink
```

---

## Typical Workflow

The recommended processing sequence for post-processing:

```bash
# Step 1: Filter and synchronize data (prerequisite)
python ../sync_image_kinematics/filter_episodes.py /raw_data /filtered_data

# Step 2: Slice into action-based episodes
python slice_affordance.py \
    --source_dataset_dir filtered_data \
    --out_dir dataset_sliced \
    --hardlink

# Step 3: Normalize timestamps and frame names
python reformat_data.py \
    --data-path dataset_sliced \
    --rename-folders \
    --new-name cholecystectomy \
    --workers 8

# Alternative: Combine steps 2 and 3
python slice_affordance.py --source_dataset_dir filtered_data --reformat
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
