# Surgical Robotics Data Collection - Source Code

This directory contains the complete data processing pipeline for the OpenH surgical robotics data collection project. The codebase is designed to process, synchronize, filter, and organize surgical robot teleoperation data from dVRK (da Vinci Research Kit) systems.

## Project Overview

This project processes multi-modal surgical robot datasets, including:
- **4 camera views**: Left stereo, right stereo, endoscope PSM1, endoscope PSM2
- **Kinematic data**: End-effector positions, velocities, and timestamps
- **Temporal synchronization**: Alignment of camera frames with kinematic data
- **Action-based slicing**: Segmentation into discrete surgical tasks

The pipeline transforms raw data collection sessions into training-ready datasets for surgical robotics machine learning applications.

---

## Directory Structure

```
src/
├── logger_config.py              # Centralized logging configuration
├── scripts/                      # Main processing scripts organized by function
│   ├── post_processing/          # Dataset reformatting and action-based slicing
│   │   ├── reformat_data.py
│   │   └── slice_affordance.py
│   ├── sync_image_kinematics/    # Temporal synchronization and filtering
│   │   ├── filter_episodes.py
│   │   └── sync_image_kinematics.py
│   └── video_processing/         # Frame-to-video conversion utilities
│       ├── frames_to_vids.py
│       └── merge_actions_to_vids.py
└── README.md                     # This file
```

---

## Module Descriptions

### logger_config.py

**Purpose**: Centralized logging configuration for the entire project.

**Features**:
- Consistent log formatting across all modules
- Project-wide root logger namespace (`data_collection`)
- Per-module log level customization
- Optional file logging support
- Safe repeated imports (no duplicate handlers)

**Usage**:
```python
from logger_config import get_logger

# Basic usage - inherits INFO level
logger = get_logger(__name__)
logger.info("Processing started")

# Debug level for specific module
logger = get_logger(__name__, level=logging.DEBUG)

# Log to file in addition to console
logger = get_logger(__name__, log_file="run.log")
```

**Key Functions**:
- `get_logger(name, level=None, log_file=None)`: Create or retrieve a project logger

---

### scripts/post_processing/

Contains scripts for reformatting sliced datasets and organizing them into training-ready structures.

**See [scripts/post_processing/README.md](scripts/post_processing/README.md) for detailed documentation.**

---

### scripts/sync_image_kinematics/

Handles temporal synchronization between camera frames and kinematic data, plus multi-camera synchronization and filtering.

**See [scripts/sync_image_kinematics/README.md](scripts/sync_image_kinematics/README.md) for detailed documentation.**

---

### scripts/video_processing/

Utilities for converting image frame sequences into video files for visualization and storage.

**See [scripts/video_processing/README.md](scripts/video_processing/README.md) for detailed documentation.**

---

## Quick Start

### Prerequisites

Install required dependencies:

```bash
pip install opencv-python numpy matplotlib pandas tqdm
```

Pip install this package. From the repository base directory, run:
```bash
pip install -e .
```

### Typical Workflow

**NOTE: The below assumed that your original data is in the `../Data` directory...**

#### 1. **Synchronize Images and Kinematics** (First Step)
```bash
# Analyze single episode
# python src/scripts/sync_image_kinematics/sync_image_kinematics.py /path/to/episode --camera left

# Filter entire dataset
python -m surpass_data_collection.scripts.sync_image_kinematics.filter_episodes.py ../Data ../Filtered_Data
```

#### 2. **Slice Into Actions** (After synchronization)
```bash
# Slice based on annotations
python -m surpass_data_collection.scripts.post_processing.slice_affordance --post_process_dir ../Data/Cholecystectomy/post_annotation/ --cautery_dir ../Filtered_Data/Cholecystectomy/tissues --out_dir ../Filtered_Data/Cholecystectomy/tissues_sliced
```

#### 3. **Reformat for Training** (After slicing)
```bash
# Normalize timestamps and rename frames
python -m surpass_data_collection.scripts.post_processing.reformat_data --data-path ../Filtered_Data/Cholecystectomy/tissues_sliced
```

#### 4. **Generate Videos** (Optional - for visualization)
```bash
# Convert frames to videos
python src/scripts/video_processing/frames_to_vids.py --root_dir cautery
```

---

## Common Command Examples

### Synchronization and Filtering
```bash
# Basic filtering with default 30ms threshold
python src/scripts/sync_image_kinematics/filter_episodes.py /source /output

# Custom threshold and parallel processing
python src/scripts/sync_image_kinematics/filter_episodes.py /source /output \
    --max-time-diff 50.0 --workers 8
```

### Action Slicing
```bash
# Slice from filtered dataset
python src/scripts/post_processing/slice_affordance.py \
    --source_dataset_dir filtered_data --out_dir sliced_episodes

# Use hardlinks for faster processing (same filesystem required)
python src/scripts/post_processing/slice_affordance.py --hardlink
```

### Data Reformatting
```bash
# Normalize timestamps and frames
python src/scripts/post_processing/reformat_data.py --data-path dataset_sliced

# Rename tissue folders
python src/scripts/post_processing/reformat_data.py --data-path dataset_sliced \
    --rename-folders --new-name cholecystectomy

# Only normalize timestamps
python src/scripts/post_processing/reformat_data.py --data-path dataset_sliced \
    --timestamps-only
```

### Video Generation
```bash
# Convert frames to videos
python src/scripts/video_processing/frames_to_vids.py --root_dir cautery --fps 30

# Merge multiple runs into single videos per tissue
python src/scripts/video_processing/merge_actions_to_vids.py /path/to/cautery --fps 30
```

---

## Configuration

### Logging

The logging system is automatically initialized on import. Default settings:

- **Log Level**: INFO
- **Format**: `%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(name)s - %(message)s`
- **Root Logger**: `data_collection`

To customize logging for a script:
```python
from logger_config import get_logger
import logging

# Set to DEBUG for verbose output
logger = get_logger(__name__, level=logging.DEBUG)

# Add file logging
logger = get_logger(__name__, log_file="processing.log")
```

### Synchronization Thresholds

Default time difference thresholds for synchronization:
- **Image-Kinematic sync**: 30ms
- **Multi-camera sync**: 30ms

These can be adjusted via command-line arguments. Higher thresholds accept more frames but may reduce temporal accuracy. Lower thresholds are more strict but may filter out valid frames.

### Parallel Processing

Most scripts support parallel processing via `--workers` argument:
- **Default**: CPU count
- **Recommended**: 4-8 workers for local processing
- **Maximum**: Limited by available CPU cores and memory

---

## Data Flow

The typical data processing pipeline flow:

```
Raw Data Collection
        ↓
[frames_to_vids.py]           ← Generate videos for annotation
        ↓
[filter_episodes.py]           ← Filter & synchronize all cameras
        ↓
Synchronized Dataset
        ↓
[slice_affordance.py]          ← Slice into action-based episodes
        ↓
Sliced Dataset
        ↓
[reformat_data.py]             ← Normalize timestamps & frame names
        ↓
Training-Ready Dataset
```

---

## Expected Data Structures

### Raw Data Collection
```
cautery/
└── cautery_tissue#1/
    └── session_timestamp/
        ├── left_img_dir/
        │   └── frame{timestamp}_left.jpg
        ├── right_img_dir/
        ├── endo_psm1/
        ├── endo_psm2/
        └── ee_csv.csv
```

### After Filtering
```
filtered_data/
└── tissue_1/
    └── session_name/
        ├── left_img_dir/      # Only synchronized frames
        ├── right_img_dir/
        ├── endo_psm1/
        ├── endo_psm2/
        └── ee_csv.csv         # Filtered kinematics
```

### After Slicing
```
dataset_sliced/
└── tissue_1/
    ├── 1_grasp/
    │   ├── episode_001/
    │   │   ├── left_img_dir/
    │   │   ├── right_img_dir/
    │   │   ├── endo_psm1/
    │   │   ├── endo_psm2/
    │   │   └── ee_csv.csv
    │   └── episode_002/
    └── 2_dissect/
```

---

## Troubleshooting

### Common Issues

**Issue**: `No valid episodes found`
- **Cause**: Missing `left_img_dir` or `ee_csv.csv` in episode directories
- **Solution**: Verify data structure matches expected format

**Issue**: `VideoWriter failed to initialize`
- **Cause**: Missing or incompatible video codecs
- **Solution**: Ensure OpenCV is installed with video codec support (try `pip install opencv-python-headless`)

**Issue**: `Empty CSV files after slicing`
- **Cause**: Frame index mapping mismatch between filtered and reference datasets
- **Solution**: Ensure reference and source datasets use the same timestamp structure

**Issue**: `Timestamp extraction failed`
- **Cause**: Image filenames don't match expected pattern `frame{timestamp}_{camera}.jpg`
- **Solution**: Check filename format or adjust regex in `extract_timestamp()` function

### Performance Optimization

- **Use hardlinks**: Add `--hardlink` flag when source and destination are on the same filesystem
- **Increase workers**: Adjust `--workers` parameter based on CPU cores (typically 4-8)
- **Filter first**: Run filtering before slicing to reduce dataset size early
- **Batch processing**: Process multiple episodes in parallel when possible

---

## Development Notes

### Code Style
- Type hints used throughout for clarity
- Comprehensive docstrings following Google style
- Logging at appropriate levels (DEBUG, INFO, WARNING, ERROR)
- Error handling with graceful degradation

### Testing Approach
- Dry-run modes available in most scripts (`--dry-run`)
- Progress logging with statistics
- Validation checks before processing
- Idempotent operations where possible

### Adding New Scripts

When adding new processing scripts:

1. Import logging: `from logger_config import get_logger`
2. Initialize logger: `logger = get_logger(__name__)`
3. Add comprehensive docstrings
4. Include CLI argument parsing with `--help`
5. Implement validation before processing
6. Add progress logging for long operations
7. Update relevant README files

---

## Additional Resources

- **Main Project README**: `../README.md`
- **Post-processing Documentation**: `scripts/post_processing/README.md`
- **Synchronization Documentation**: `scripts/sync_image_kinematics/README.md`
- **Video Processing Documentation**: `scripts/video_processing/README.md`

---

## License

See project root LICENSE file for details.
