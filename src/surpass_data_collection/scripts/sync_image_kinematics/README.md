# Synchronization and Filtering Scripts

This directory contains scripts for temporal synchronization of surgical robotics data and multi-camera filtering to ensure high-quality, synchronized datasets.

## Overview

Temporal synchronization is critical for surgical robotics datasets because:
- **Camera frames and kinematics** are captured by different systems with separate clocks
- **Multiple cameras** must be synchronized to ensure consistent multi-view observations
- **Timestamp drift** can occur over long recording sessions
- **Missing or dropped frames** need to be identified and handled

These scripts provide a robust pipeline for analyzing synchronization quality, filtering out poorly synchronized frames, and ensuring strict multi-camera alignment.

---

## Directory Contents

```
sync_image_kinematics/
├── sync_image_kinematics.py   # Analyze synchronization quality (single episode)
├── filter_episodes.py         # Batch filtering with multi-camera sync
└── README.md                  # This file
```

---

## Scripts

### sync_image_kinematics.py

**Purpose**: Synchronize image timestamps with kinematic data and analyze temporal alignment quality.

**Key Features**:
- Timestamp extraction from image filenames
- Nearest-neighbor matching with kinematic data (binary search)
- Outlier detection and removal
- Synchronization quality visualization
- Can be used as library (no subprocess overhead)

**Use Cases**:
- Analyzing synchronization quality of a single episode
- Debugging timestamp alignment issues
- Generating synchronization reports for QA
- Imported by filter_episodes.py for batch processing

**Command-Line Interface**:

```bash
# Basic analysis of left camera
python sync_image_kinematics.py /path/to/episode --camera left

# Custom synchronization threshold
python sync_image_kinematics.py /path/to/episode --camera left \
    --max-time-diff 50.0

# Specify output directory for results
python sync_image_kinematics.py /path/to/episode --camera left \
    --output-dir sync_results

# Analyze different camera view
python sync_image_kinematics.py /path/to/episode --camera psm1
```

**Library Usage** (Recommended for programmatic access):

```python
from sync_image_kinematics import process_episode_sync

result = process_episode_sync(
    episode_path="/data/episode_001",
    camera="left",
    max_time_diff_ms=30.0,
    plot=False,           # Skip plotting for batch processing
    save_results=False    # In-memory only (faster)
)

if result['success']:
    valid_files = result['valid_filenames']
    sync_df = result['sync_df']
    print(f"Found {result['num_valid_images']} valid images")
    print(f"Removed {result['outliers_removed']} outliers")
```

**Expected Data Structure**:
```
episode_dir/
├── left_img_dir/
│   ├── frame1756826516968031906_left.jpg
│   ├── frame1756826517035142017_left.jpg
│   └── ...
├── right_img_dir/
│   └── frame1756826516968031906_right.jpg
├── endo_psm1/
├── endo_psm2/
└── ee_csv.csv         # Kinematic data with timestamp column
```

**Output Files** (when save_results=True):
```
episode_dir/sync_analysis/
├── sync_results_original.csv    # All sync results
├── sync_results_filtered.csv    # After outlier removal
├── sync_results_outliers.csv    # Removed outliers
├── valid_image_filenames.txt    # List of valid filenames
└── sync_analysis.png            # Visualization (if plot=True)
```

**Processing Pipeline**:

1. **Timestamp Extraction**: Parse nanosecond timestamps from image filenames
2. **Kinematic Loading**: Read CSV and detect/generate timestamps
3. **Nearest-Neighbor Matching**: Binary search to find closest kinematic point for each image
4. **Time Difference Calculation**: Compute signed difference in milliseconds
5. **Outlier Removal**: Filter frames exceeding threshold
6. **Optional Visualization**: Generate plots showing sync quality

**Timestamp Detection**:

The script automatically handles different timestamp formats:
- **Existing timestamps**: Searches for columns containing "time" or "stamp"
- **Synthetic timestamps**: Generates 30Hz timestamps if none found
- **Format**: Unix epoch nanoseconds

**Synchronization Algorithm**:

Uses efficient binary search (`np.searchsorted`) to find nearest kinematic timestamp:
```python
# For each image timestamp:
idx = np.searchsorted(kinematic_timestamps, image_timestamp)
# Check both neighbors (idx-1 and idx)
# Select minimum absolute distance
```

**Outlier Removal**:

Frames are considered outliers if:
```
|time_diff_ms| > max_time_diff_ms
```

Default threshold: **30ms**

**Camera Configurations**:

| Camera | Directory | Suffix |
|--------|-----------|--------|
| left | left_img_dir | _left |
| right | right_img_dir | _right |
| psm1 | endo_psm1 | _psm1 |
| psm2 | endo_psm2 | _psm2 |

**Options**:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `episode_path` | Path | Required | Episode directory to analyze |
| `--camera` | str | left | Camera view (left/right/psm1/psm2) |
| `--output-dir` | Path | sync_analysis | Output directory for results |
| `--max-time-diff` | float | 30.0 | Max time diff threshold (ms) |
| `--csv-filename` | str | ee_csv.csv | Kinematic CSV filename |

**Return Dictionary** (programmatic usage):

```python
{
    'success': bool,                    # Processing succeeded
    'valid_filenames': List[str],       # Filenames within threshold
    'sync_df': pd.DataFrame,            # Filtered sync DataFrame
    'sync_output_dir': Path,            # Output directory (if saved)
    'num_valid_images': int,            # Count of valid images
    'outliers_removed': int,            # Count of outliers
    'error': str                        # Error message (if success=False)
}
```

**Visualization Output**:

The generated plot (when `--plot` enabled) shows:
1. **Time series**: Time differences over image sequence
2. **Histogram**: Distribution of time differences
3. **Statistics**: Mean, std, and max absolute difference

---

### filter_episodes.py

**Purpose**: Filter and copy synchronized episodes with strict multi-camera synchronization.

**Key Features**:
- Batch processing of entire datasets
- Strict multi-camera synchronization (all cameras must match)
- Direct module import (no subprocess overhead)
- Parallel processing with progress bars
- Hardlink support for fast copying
- Secondary camera frame renaming for consistency

**Use Cases**:
- Filtering raw datasets before training
- Ensuring strict temporal alignment across all cameras
- Creating synchronized subsets of large datasets
- Quality control for data collection

**Command-Line Interface**:

```bash
# Basic filtering with defaults (30ms threshold)
python filter_episodes.py /source /destination

# Custom synchronization threshold
python filter_episodes.py /source /output --max-time-diff 50.0

# Require minimum valid images per episode
python filter_episodes.py /source /output --min-images 100

# Dry run to preview processing
python filter_episodes.py /source /output --dry-run

# Parallel processing with 8 workers
python filter_episodes.py /source /output --workers 8

# Use hardlinks for faster processing (same filesystem required)
python filter_episodes.py /source /output --hardlink
```

**Input Structure**:
```
source_dir/
└── cautery_tissue_001/
    └── run_timestamp_1/
        ├── left_img_dir/
        │   └── frame123456789_left.jpg
        ├── right_img_dir/
        │   └── frame123456790_right.jpg   # Different timestamp OK
        ├── endo_psm1/
        │   └── frame123456788_psm1.jpg
        ├── endo_psm2/
        │   └── frame123456791_psm2.jpg
        └── ee_csv.csv
```

**Output Structure**:
```
out_dir/
└── cautery_tissue_001/
    └── run_timestamp_1/
        ├── left_img_dir/
        │   └── frame123456789_left.jpg      # Kept (all cameras matched)
        ├── right_img_dir/
        │   └── frame123456789_right.jpg     # Renamed to match left
        ├── endo_psm1/
        │   └── frame123456789_psm1.jpg      # Renamed to match left
        ├── endo_psm2/
        │   └── frame123456789_psm2.jpg      # Renamed to match left
        └── ee_csv.csv                       # Filtered to match kept frames
```

**Synchronization Strategy**:

The script implements **strict multi-camera synchronization**:

1. **Left camera is reference**: Determines which frames to keep
2. **For each valid left frame**:
   - Find nearest frame in each secondary camera (right, psm1, psm2)
   - Check if ALL cameras have matches within threshold
3. **Frame is kept only if**:
   - All cameras have matching frames
   - All time differences < `max_time_diff_ms`
4. **Secondary frames are renamed** to match left timestamp
5. **Ensures 1:1 correspondence** across all modalities

**Processing Pipeline**:

1. **Episode Discovery**: Find all directories with `left_img_dir` and `ee_csv.csv`
2. **Validation**: Check episode structure (images + kinematics)
3. **Sync Analysis**: Run `sync_image_kinematics.py` directly (no subprocess)
4. **Camera Matching**: Load timestamps for all secondary cameras
5. **Multi-Camera Sync**: Binary search to find matches for each camera
6. **Filtering**: Keep only frames where ALL cameras match
7. **Copying**: Copy left frame + matched secondary frames (renamed)
8. **Kinematic Filtering**: Filter CSV to match kept frames

**Binary Search Matching**:

For each left frame timestamp:
```python
# Binary search in secondary camera timestamps
idx = bisect.bisect_left(camera_timestamps, left_ts)

# Check neighbors (idx and idx-1)
# Select minimum distance
# Return match if distance <= max_time_diff_ns
```

**Frame Renaming Logic**:

Secondary camera frames are renamed to ensure consistency:
```python
# Left frame (unchanged)
frame123456789_left.jpg

# Right frame (original: frame123456790_right.jpg)
# Renamed to:
frame123456789_right.jpg  # Matches left timestamp

# Similar for psm1, psm2
```

**Kinematic Filtering**:

The CSV is filtered to maintain 1:1 correspondence with images:
1. Get list of kept left filenames
2. Filter sync DataFrame to those filenames
3. Extract corresponding kinematic indices
4. Select those rows from original CSV
5. Reset index for clean output

**Episode Validation**:

Each episode is validated for:
- `left_img_dir` exists and contains JPG files
- `ee_csv.csv` exists
- At least one valid image found

Failed validation results in episode being skipped (logged).

**Camera Configurations**:

```python
CAMERA_CONFIGS = [
    ("left_img_dir", "_left.jpg", "left_img_dir"),
    ("right_img_dir", "_right.jpg", "right_img_dir"),
    ("endo_psm1", "_psm1.jpg", "endo_psm1"),
    ("endo_psm2", "_psm2.jpg", "endo_psm2"),
]
```

**Options**:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `source_dir` | Path | Required | Source dataset directory |
| `out_dir` | Path | Required | Destination directory |
| `--max-time-diff` | float | 30.0 | Max sync threshold (ms) |
| `--min-images` | int | 10 | Min valid images to keep episode |
| `--workers` | int | CPU count | Parallel episode workers (max 8) |
| `--hardlink` | flag | False | Use hardlinks instead of copying |
| `--dry-run` | flag | False | Preview without processing |

**Performance Optimization**:

- **Direct import**: No subprocess overhead (10x faster than calling script)
- **Binary search**: O(log n) frame lookups
- **Parallel processing**: Multiple episodes processed simultaneously
- **Hardlinks**: Instant "copying" on same filesystem
- **Pre-calculated lists**: Camera timestamp lists built once per episode

**Statistics Reporting**:

Progress bars show:
- Episodes processed
- Frames kept/filtered ratio
- Processing rate (episodes/second)

Final summary shows:
- Total episodes processed
- Total episodes kept/skipped
- Total frames synchronized
- Average synchronization ratio

**Disk Space Considerations**:

| Copy Mode | Operation | Speed | Requirements |
|-----------|-----------|-------|--------------|
| **Regular Copy** | Duplicates data | ~100MB/s | 2x storage space |
| **Hardlink** | Links to original | Instant | Same filesystem |

Always use `--hardlink` when possible for development/testing.

**Common Patterns**:

```bash
# High-quality filtering (strict threshold)
python filter_episodes.py /raw /filtered --max-time-diff 20.0

# Fast filtering with parallel processing
python filter_episodes.py /raw /filtered --workers 8 --hardlink

# Conservative filtering (keep more frames)
python filter_episodes.py /raw /filtered --max-time-diff 50.0

# Preview before running
python filter_episodes.py /raw /filtered --dry-run
```

---

## Typical Workflow

The recommended synchronization workflow:

```bash
# Step 1: Analyze single episode to understand sync quality
python sync_image_kinematics.py /raw/tissue_1/session_1 --camera left

# Review sync_analysis.png to determine appropriate threshold

# Step 2: Filter entire dataset with chosen threshold
python filter_episodes.py /raw /filtered \
    --max-time-diff 30.0 \
    --workers 8 \
    --hardlink

# Step 3: Proceed to slicing and reformatting
python ../post_processing/slice_affordance.py --source_dataset_dir filtered
```

---

## Synchronization Quality

### Understanding Time Differences

**Good synchronization** (< 30ms):
- Typical camera-kinematic lag in dVRK systems
- Acceptable for most learning tasks
- Minimal impact on temporal relationships

**Moderate synchronization** (30-50ms):
- May include startup/shutdown frames
- Consider increasing threshold if losing many frames
- Review visualization to check for systematic issues

**Poor synchronization** (> 50ms):
- Clock drift or system issues
- Investigate root cause before training
- May indicate hardware problems

### Threshold Selection

Choose threshold based on:
- **Application requirements**: High-frequency tasks need stricter sync
- **Data quality**: Review sync_analysis.png histograms
- **Frame retention**: Balance quality vs. dataset size
- **Camera specs**: Different cameras may have different latencies

**Recommended thresholds**:
- **Default**: 30ms (good balance)
- **Strict**: 20ms (high-quality, may lose frames)
- **Relaxed**: 50ms (keep more data, lower temporal precision)

---

## Troubleshooting

### Common Issues

**Issue**: `No valid episodes found`
- **Cause**: Missing `left_img_dir` or `ee_csv.csv`
- **Solution**: Verify data structure matches expected format

**Issue**: `No frames passed multi-camera synchronization`
- **Cause**: Secondary cameras don't have matching timestamps
- **Solution**: Check camera recording start times; increase threshold

**Issue**: `Kinematic timestamps not found`
- **Cause**: CSV doesn't have timestamp column
- **Solution**: Script generates synthetic 30Hz timestamps automatically

**Issue**: Low frame retention (< 50%)
- **Cause**: Poor camera synchronization or too strict threshold
- **Solution**: Analyze with sync_image_kinematics.py; adjust threshold

**Issue**: `ValueError: Could not extract timestamp from filename`
- **Cause**: Filenames don't match pattern `frame{timestamp}_{camera}.jpg`
- **Solution**: Check filename format; ensure timestamp collection was correct

### Performance Issues

**Slow processing**:
- Use `--hardlink` flag (same filesystem)
- Increase `--workers` (4-8 recommended)
- Use SSD for source and destination

**High memory usage**:
- Reduce `--workers` count
- Process episodes sequentially (--workers 1)

### Data Quality Issues

**Missing camera views**:
- Script gracefully skips missing cameras
- Only processes modalities that exist in ALL runs

**Inconsistent timestamps**:
- Run sync_image_kinematics.py to diagnose
- Check for system clock issues during recording

---

## Configuration

### Default Settings

```python
# Synchronization threshold
DEFAULT_MAX_TIME_DIFF_MS = 30.0

# Minimum images to keep episode
DEFAULT_MIN_IMAGES = 10

# Maximum parallel workers
MAX_WORKERS = 8

# Synthetic timestamp frequency (if timestamps missing)
SYNTHETIC_TIMESTAMP_FREQ_HZ = 30
```

### Camera Modalities

To add new camera views, update `CAMERA_CONFIGS`:

```python
CAMERA_CONFIGS = [
    ("left_img_dir", "_left.jpg", "left_img_dir"),
    ("right_img_dir", "_right.jpg", "right_img_dir"),
    ("endo_psm1", "_psm1.jpg", "endo_psm1"),
    ("endo_psm2", "_psm2.jpg", "endo_psm2"),
    # Add new camera here:
    # ("new_camera_dir", "_new.jpg", "new_camera_dir"),
]
```

---

## Dependencies

```bash
# Required packages
pip install numpy pandas matplotlib

# Optional (for progress bars in filter_episodes.py)
pip install tqdm
```

---

## Advanced Usage

### Batch Synchronization Analysis

Analyze all episodes in a dataset:

```bash
for episode in /raw/tissue_*/session_*/; do
    python sync_image_kinematics.py "$episode" --camera left \
        --output-dir "${episode}/sync_analysis"
done
```

### Custom Filtering Logic

Both scripts can be imported as libraries:

```python
from sync_image_kinematics import process_episode_sync, find_nearest_kinematics
from filter_episodes import find_episodes, process_single_episode

# Custom processing pipeline
episodes = find_episodes("/source")
for ep in episodes:
    result = process_episode_sync(ep, camera="left", max_time_diff_ms=25.0)
    if result['num_valid_images'] >= 50:
        # Custom logic here
        pass
```

### Generating Sync Reports

Create comprehensive sync reports:

```python
import pandas as pd
from sync_image_kinematics import process_episode_sync

episodes = [...]  # List of episode paths
results = []

for ep in episodes:
    res = process_episode_sync(ep, save_results=False)
    if res['success']:
        results.append({
            'episode': ep.name,
            'valid_images': res['num_valid_images'],
            'outliers': res['outliers_removed'],
            'retention': res['num_valid_images'] / (res['num_valid_images'] + res['outliers_removed'])
        })

df = pd.DataFrame(results)
df.to_csv('sync_report.csv', index=False)
print(df.describe())
```

---

## See Also

- **Main src README**: `../README.md`
- **Post-processing scripts**: `../post_processing/README.md`
- **Video processing**: `../video_processing/README.md`
