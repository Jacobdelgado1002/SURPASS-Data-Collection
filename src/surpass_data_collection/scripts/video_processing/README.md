# Video Processing Scripts

This directory contains utilities for converting image frame sequences into video files for visualization, storage, and presentation purposes.

## Overview

While the primary data format for machine learning is individual frames + kinematic CSV, videos are valuable for:
- **Visualization**: Quickly reviewing surgical sessions
- **Presentation**: Demonstrating data collection to stakeholders
- **Debugging**: Identifying data quality issues
- **Storage**: Compressed video files for archival
- **Sharing**: Easier to share than thousands of individual frames

These scripts handle frame-to-video conversion with automatic resizing, frame sorting, and multi-run merging.

---

## Directory Contents

```
video_processing/
├── frames_to_vids.py         # Convert frame directories to videos
├── merge_actions_to_vids.py  # Merge multiple runs into continuous videos
└── README.md                 # This file
```

---

## Scripts

### frames_to_vids.py

**Purpose**: Traverse cautery folder structure and convert image sequences into MP4 videos.

**Key Features**:
- Automatic directory traversal (finds all tissue/run folders)
- Natural sorting of frames (frame1, frame2, ..., frame10, frame11)
- Automatic frame resizing to match first frame
- Codec fallback (MP4V → MJPG if needed)
- Progress logging every 500 frames
- Handles corrupted/unreadable frames gracefully

**Use Cases**:
- Creating videos from raw data collection sessions
- Generating visualizations for presentations
- Archiving data in compressed format
- Quick review of recorded sessions

**Command-Line Interface**:

```bash
# Basic usage with default settings
python frames_to_vids.py

# Custom root directory and frame rate
python frames_to_vids.py --root_dir mydata --fps 30

# Dry run to preview what will be processed
python frames_to_vids.py --dry_run

# Overwrite existing videos
python frames_to_vids.py --overwrite

# Custom output directory
python frames_to_vids.py --out_dir my_videos
```

**Expected Data Structure**:
```
cautery/
└── cautery_tissue_001/
    ├── run_001/
    │   ├── endo_psm1/
    │   │   ├── frame_0001.png
    │   │   ├── frame_0002.png
    │   │   └── ...
    │   ├── endo_psm2/
    │   ├── left_img_dir/
    │   └── right_img_dir/
    └── run_002/
        └── ...
```

**Output Structure**:
```
videos/
└── cautery_tissue_001/
    ├── run_001/
    │   ├── endo_psm1.mp4
    │   ├── endo_psm2.mp4
    │   ├── left_img_dir.mp4
    │   └── right_img_dir.mp4
    └── run_002/
        └── ...
```

**Processing Pipeline**:

1. **Directory Discovery**:
   - Find all `cautery_tissue*` directories in root
   - For each tissue, find all run directories
   - For each run, find all camera folders

2. **Frame Collection**:
   - Collect all image files (`.jpg`, `.png`)
   - Sort using natural ordering (handles frame10 correctly)
   - Skip non-image files (CSV, TXT, JSON, XML)

3. **Video Initialization**:
   - Read first frame to determine dimensions
   - Initialize VideoWriter with MP4V codec
   - Fallback to MJPG/AVI if MP4V fails

4. **Frame Processing**:
   - Read each frame in order
   - Resize to match reference dimensions if needed
   - Skip corrupted frames with warning
   - Log progress periodically

5. **Finalization**:
   - Release VideoWriter
   - Log summary (frames written, frames skipped)

**Frame Sorting Logic**:

Uses natural key sorting to ensure proper ordering:
```python
# Natural sorting handles:
frame_1.png     → 1
frame_2.png     → 2
frame_10.png    → 10  # Correct!
frame_100.png   → 100

# (Lexicographic sorting would give: frame_1, frame_10, frame_100, frame_2)
```

**Codec Fallback**:

Primary codec: **MP4V** (widely compatible, good compression)
- Modern, widely supported
- Good quality/size ratio
- `.mp4` extension

Fallback codec: **MJPG** (maximum compatibility)
- Works on all OpenCV builds
- Larger file sizes
- `.avi` extension

The script automatically detects codec availability and falls back if needed.

**Frame Resizing**:

All frames in a video must have the same dimensions. The script:
1. Uses first readable frame as reference size
2. Resizes subsequent frames to match
3. Uses bilinear interpolation (`cv2.INTER_LINEAR`)
4. Logs resizing operations for debugging

**Skipped Frame Handling**:

Corrupted or unreadable frames are:
- Logged at DEBUG level individually
- Counted and reported in summary
- Warned about every 50 skips
- Non-fatal (processing continues)

**Options**:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--root_dir` | Path | cautery | Root directory with tissue folders |
| `--fps` | int | 30 | Frames per second for output videos |
| `--out_dir` | Path | videos | Output directory for videos |
| `--dry_run` | flag | False | Preview processing without creating videos |
| `--overwrite` | flag | False | Overwrite existing videos |

**Performance Characteristics**:

- **Frame Processing**: ~100-200 frames/second (CPU-dependent)
- **Codec**: MP4V offers ~10x compression vs. raw frames
- **Memory**: Processes one frame at a time (memory efficient)
- **Disk I/O**: Limited by read speed of source images

**Common Patterns**:

```bash
# Preview what will be processed
python frames_to_vids.py --dry_run

# Process all default structure with custom FPS
python frames_to_vids.py --fps 60

# Process custom directory structure
python frames_to_vids.py --root_dir /data/surgical_videos --out_dir /output/vids

# Regenerate all videos (overwrite existing)
python frames_to_vids.py --overwrite
```

---

### merge_actions_to_vids.py

**Purpose**: Consolidate multiple action runs within each tissue folder into unified video streams and merged kinematic data.

**Key Features**:
- Concatenates frames from multiple runs chronologically
- Merges kinematic CSV files vertically
- Maintains temporal ordering
- Automatic frame resizing per modality
- Handles missing runs gracefully
- Preserves original directory structure

**Use Cases**:
- Creating continuous videos from multiple recording sessions
- Merging data from interrupted recordings
- Consolidating daily collection runs
- Generating overview videos of entire tissue samples

**Command-Line Interface**:

```bash
# Process all tissue folders with defaults
python merge_actions_to_vids.py /path/to/cautery

# Specify custom FPS
python merge_actions_to_vids.py /path/to/cautery --fps 60

# Overwrite existing outputs
python merge_actions_to_vids.py /path/to/cautery --overwrite

# Combine options
python merge_actions_to_vids.py /path/to/cautery --fps 30 --overwrite
```

**Expected Data Structure**:
```
root_dir/
└── cautery_tissue_001/
    ├── 2024_01_15_10_30_45/     # Run 1 (timestamped)
    │   ├── endo_psm1/
    │   │   ├── frame_0001.png
    │   │   └── ...
    │   ├── endo_psm2/
    │   ├── left_img_dir/
    │   ├── right_img_dir/
    │   └── ee_csv.csv
    ├── 2024_01_15_11_45_30/     # Run 2
    │   └── ...
    └── 2024_01_15_14_20_10/     # Run 3
        └── ...
```

**Output Structure**:
```
cautery_tissue_001/
└── videos/
    ├── endo_psm1.mp4           # Concatenated from all runs
    ├── endo_psm2.mp4
    ├── left_img_dir.mp4
    ├── right_img_dir.mp4
    └── ee_csv.csv              # Merged kinematic data
```

**Processing Pipeline**:

1. **Tissue Discovery**:
   - Find all `cautery_tissue*` directories in root
   - Sort alphabetically for consistent processing

2. **Run Discovery**:
   - Find all subdirectories within each tissue folder
   - Validate that each run has all 4 modalities
   - Sort chronologically (timestamp-based sorting)
   - Skip incomplete runs with warning

3. **Frame Collection**:
   - Collect frames from all runs for each modality
   - Concatenate in run order
   - Handle missing modalities gracefully

4. **Video Generation**:
   - Determine reference dimensions from first frame
   - Initialize VideoWriter for modality
   - Process all frames with automatic resizing
   - Skip corrupted frames (logged)

5. **CSV Merging**:
   - Read all `ee_csv.csv` files from runs
   - Preserve first run's header
   - Strip headers from subsequent runs
   - Append data rows vertically
   - Write merged output

**Run Validation**:

Runs must contain all modalities to be included:
```python
MODALITIES = ["endo_psm1", "endo_psm2", "left_img_dir", "right_img_dir"]

# Run is valid only if ALL modalities present
for modality in MODALITIES:
    if not (run_dir / modality).exists():
        # Skip this run
```

This ensures temporal consistency across all camera views.

**CSV Merging Strategy**:

Vertical concatenation with header handling:

```
Run 1 CSV:                Run 2 CSV:
timestamp,x,y,z          timestamp,x,y,z    (header)
0.0,1.0,2.0,3.0         10.0,1.1,2.1,3.1
0.033,1.1,2.1,3.1       10.033,1.2,2.2,3.2
...                      ...

Merged CSV:
timestamp,x,y,z          (header from run 1)
0.0,1.0,2.0,3.0         (run 1 data)
0.033,1.1,2.1,3.1
...
10.0,1.1,2.1,3.1        (run 2 data, header stripped)
10.033,1.2,2.2,3.2
...
```

**Frame Resizing Logic**:

Each modality uses its own reference dimensions:
- Reference frame = first readable frame for that modality
- All subsequent frames resized to match reference
- Allows different resolutions across modalities
- Ensures consistent dimensions within each video

**Missing Data Handling**:

The script gracefully handles:
- **Missing modalities**: Warned and skipped
- **Missing CSV files**: Warned, merge continues with other runs
- **Empty directories**: Skipped with warning
- **Corrupted frames**: Skipped, processing continues

**Temporal Ordering**:

Runs are processed in sorted order:
- Timestamp-based directory names sort chronologically
- Ensures temporal coherence in merged videos
- Maintains original recording sequence

**Options**:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `root_dir` | Path | Required | Root directory with tissue folders |
| `--fps` | int | 30 | Frames per second for output videos |
| `--overwrite` | flag | False | Overwrite existing output files |

**Constants**:

```python
# Modalities to process
MODALITIES = ["endo_psm1", "endo_psm2", "left_img_dir", "right_img_dir"]

# Image file extensions
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# CSV filename
KINEMATIC_CSV_NAME = "ee_csv.csv"

# Output directory name
OUTPUT_DIR_NAME = "videos"

# Video codec
VIDEO_CODEC = "mp4v"
```

**Performance Considerations**:

- **Memory**: One frame at a time (efficient)
- **Disk I/O**: Limited by source image read speed
- **Processing**: Linear in total frame count
- **Codec**: MP4V compression (~10x)

**Common Patterns**:

```bash
# Standard merging
python merge_actions_to_vids.py /data/cautery

# High frame rate videos
python merge_actions_to_vids.py /data/cautery --fps 60

# Regenerate merged videos
python merge_actions_to_vids.py /data/cautery --overwrite

# Process different root structure
python merge_actions_to_vids.py /custom/data/path --fps 30
```

**Output Summary**:

After completion, the script logs:
- Number of tissue directories processed
- Number of runs per tissue
- Frames written per modality
- Frames skipped per modality
- CSV rows merged
- Output file locations

---

## Typical Workflows

### Single Run Video Generation

Convert individual runs to videos for review:

```bash
# Generate videos for all runs
python frames_to_vids.py --root_dir cautery --fps 30

# Review specific tissue
ls videos/cautery_tissue_001/
```

### Multi-Run Consolidation

Merge multiple collection sessions into continuous videos:

```bash
# Merge all runs within tissues
python merge_actions_to_vids.py cautery --fps 30

# Result: One video per modality per tissue
ls cautery_tissue_001/videos/
# endo_psm1.mp4  endo_psm2.mp4  left_img_dir.mp4  right_img_dir.mp4  ee_csv.csv
```

### Regenerating Videos

Update videos after data changes:

```bash
# Regenerate all videos
python frames_to_vids.py --overwrite
# or
python merge_actions_to_vids.py cautery --overwrite
```

---

## Video Specifications

### Output Format

| Property | Value | Notes |
|----------|-------|-------|
| **Container** | MP4 | Or AVI if fallback used |
| **Codec** | MP4V | Or MJPG if fallback |
| **FPS** | 30 (default) | Configurable via `--fps` |
| **Resolution** | Variable | Matches first frame |
| **Color Space** | BGR | OpenCV standard |
| **Compression** | ~10x | Codec-dependent |

### Typical File Sizes

For reference (720p, 30 FPS, 1000 frames):
- **Raw frames**: ~500MB (500KB per JPEG)
- **MP4V video**: ~50MB (~10x compression)
- **MJPG video**: ~200MB (less compression)

Actual sizes vary based on:
- Resolution
- Frame content complexity
- Codec settings
- Frame count

---

## Troubleshooting

### Common Issues

**Issue**: `VideoWriter failed to initialize`
- **Cause**: Missing or incompatible video codec
- **Solution**: 
  - Install OpenCV with codec support: `pip install opencv-python`
  - Script automatically falls back to MJPG
  - Check OpenCV build: `python -c "import cv2; print(cv2.getBuildInformation())"`

**Issue**: Videos have wrong dimensions
- **Cause**: First frame is corrupted or has unusual size
- **Solution**: 
  - Check first frame in each directory
  - Manually delete corrupted first frame
  - Re-run script to use next valid frame

**Issue**: Frames out of order in video
- **Cause**: Filenames don't sort naturally
- **Solution**: 
  - Verify frame naming follows `frame_NNNN.ext` pattern
  - Check for leading zeros in frame numbers
  - Natural sorting should handle most cases

**Issue**: `No image files found` warning
- **Cause**: Directory is empty or contains only non-image files
- **Solution**: 
  - Verify camera folders contain JPG/PNG files
  - Check file extensions (case-sensitive)
  - Review data collection process

**Issue**: Video playback is too fast/slow
- **Cause**: FPS mismatch with recording rate
- **Solution**: 
  - Adjust `--fps` parameter
  - Typical dVRK recording: 30 FPS
  - High-speed cameras: 60+ FPS

**Issue**: Merge creates empty videos
- **Cause**: No valid runs found in tissue directory
- **Solution**: 
  - Verify runs contain all 4 modalities
  - Check directory naming (may not match pattern)
  - Review validation log messages

### Performance Issues

**Slow processing**:
- Use SSD for source images
- Reduce resolution at source if possible
- Consider using fewer FPS for preview videos

**Large file sizes**:
- MP4V should give ~10x compression
- If files too large, consider:
  - Lower FPS
  - External compression tools (ffmpeg)
  - Different codec (requires OpenCV build)

**Memory issues**:
- Should not occur (one frame at a time)
- If crashes, check OpenCV installation
- Verify sufficient disk space for output

---

## Configuration

### Codec Selection

The scripts use a two-tier codec fallback:

1. **Primary: MP4V**
   - Modern, widely supported
   - Good compression ratio
   - `.mp4` output

2. **Fallback: MJPG**
   - Universal compatibility
   - Larger file sizes
   - `.avi` output

Codec is automatically selected based on availability.

### Frame Rate

Default: **30 FPS** (matches typical dVRK recording rate)

Adjust based on:
- **Recording rate**: Match source frame rate
- **Playback speed**: Lower FPS for slow-motion effect
- **File size**: Lower FPS reduces output size

### Image Extensions

Supported extensions:
```python
VALID_EXTENSIONS = ('.jpg', '.png')  # frames_to_vids.py
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')  # merge_actions_to_vids.py
```

Case-insensitive matching (`.JPG` and `.jpg` both work).

---

## Advanced Usage

### Custom Directory Structures

Process non-standard directory layouts:

```bash
# Custom root directory
python frames_to_vids.py --root_dir /custom/path --out_dir /output

# Process single tissue folder
cd cautery_tissue_001
python ../frames_to_vids.py --root_dir . --out_dir ./videos
```

### Batch Processing

Process multiple datasets:

```bash
#!/bin/bash
for dataset in /data/datasets/*; do
    echo "Processing $dataset"
    python frames_to_vids.py --root_dir "$dataset" --out_dir "${dataset}_videos"
done
```

### Quality Control

Generate videos for QA review:

```bash
# Low-res, high-FPS preview
python frames_to_vids.py --fps 60 --out_dir preview_videos

# Review output
vlc videos/cautery_tissue_001/run_001/endo_psm1.mp4
```

### Storage Archival

Create compressed videos for long-term storage:

```bash
# Generate videos
python merge_actions_to_vids.py cautery

# Archive video folder
tar -czf cautery_videos_$(date +%Y%m%d).tar.gz cautery_tissue_*/videos/

# Delete original frames (if backup exists)
# rm -rf cautery_tissue_*/*/endo_psm*/frame*.png
```

---

## Dependencies

```bash
# Required packages
pip install opencv-python  # or opencv-python-headless

# Verify installation
python -c "import cv2; print(cv2.__version__)"
```

**Minimum versions**:
- OpenCV: >= 4.0

**Optional**:
- FFmpeg (for codec troubleshooting)
- VLC or similar (for playback verification)

---

## See Also

- **Main src README**: `../README.md`
- **Post-processing scripts**: `../post_processing/README.md`
- **Synchronization scripts**: `../sync_image_kinematics/README.md`
- **OpenCV VideoWriter documentation**: https://docs.opencv.org/4.x/dd/d9e/classcv_1_1VideoWriter.html
