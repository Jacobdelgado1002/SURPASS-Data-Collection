# DVRK to LeRobot Converter GUIs

This module provides two GUI tools to convert DVRK surgical robot datasets into HuggingFace LeRobot formats with nanosecond-precision timestamp alignment across multiple camera views.

Two versions are provided to support different LeRobot versions and pipeline requirements:

1. `dvrk_lerobot_converter_gui.py`: Built for **LeRobot v3.0 (v0.4.3)**. Supports direct ingestion of filtered affordance slices via `slice_affordance.py` annotations, skipping physical intermediate file copies.
2. `dvrk_lerobot_converter_gui_v2.1.py`: Built for **LeRobot v2.1**. A legacy flattener that expects a pre-copied `episode_xxx/` structure.

## Features

- **PyQt5 GUI**: No command-line arguments needed
- **Multi-camera Sync**: Timestamp-based alignment across 4 streams (left/right endoscope, left/right wrist)
- **Extreme Encoding Optimization**: Pre-places uncompressed camera JPEGs via hardlinks into fast NVMe caching (`TEMP_IMAGE_DIR`) to bypass Python decode/encode bottlenecks—letting FFmpeg crunch raw bytes natively.
- **NVIDIA GPU Acceleration**: Native bindings for GPU video decoding/encoding inside LeRobot processes
- **Pipeline Integration (v3.0)**: Reads slice boundaries dynamically from annotation CSVs, slicing NumPy RAM arrays without creating physical copies of the frames.
- **Resume Support**: Safely detects completed chunks and resumes partial encodes.

## Requirements

- **Python 3.10+**
- **NVIDIA GPU** with up-to-date drivers (recommended for best performance)
- A fast local drive (SSD/NVMe) for intermediate files

## Expected Input Data Structure

Both converters require the DVRK robot images to contain nanosecond timestamps, e.g., `frame1767971796430639266_psm1.jpg`. The kinematics CSV must have a `timestamp` column along with PSM state and action columns.

### Pipeline Data (Annotations based)

The v3.0 Converter expects raw data and annotation data directly from `slice_affordance.py`:

```
source_data/
  session_dir/
    left_img_dir/, right_img_dir/, endo_psm1/, endo_psm2/, ee_csv.csv
annotations_dir/
  task_data.csv          # Defines start/end points mapping into 'source_data'
```

Image filenames must contain a nanosecond timestamp, e.g. `frame1767971796430639266_psm1.jpg`.

The CSV must have a `timestamp` column (nanoseconds) plus the state and action columns defined in the script (PSM1/PSM2 pose, orientation, jaw, and setpoints).

## Setup

### 1. Create a conda environment (recommended)

```bash
conda create -n lerobot python=3.10 -y
conda activate lerobot
```

### 2. Install PyTorch (with CUDA)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 3. Install LeRobot

**For the v3.0 GUI:**

```bash
pip install lerobot==0.4.3
```

**For the v2.1 Legacy GUI:**

```bash
pip install lerobot==0.2.1
```

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

## Usage

**For the v3.0 Annotated Pipeline:**

```bash
python dvrk_lerobot_converter_gui.py
```

1. Set the **Raw Data Directory** to the source recordings.
2. Set the **Annotations Dir** to the folder holding the post-processed CSVs defining the slices.
3. Select an **Output Directory** and define a **Dataset Name**.
4. The **Video Codec** defaults to `h264 (CPU)` to comply with HuggingFace Hub web playback, but you can select `hevc` if desired.
5. Click **Start Conversion**.

**For the v2.1 Legacy Flattened Pipeline:**

```bash
python dvrk_lerobot_converter_gui_v2.1.py
```

1. Select the **Source Directory** holding subdirectories like `episode_000/`.
2. Select an **Output Directory** and define a **Dataset Name**.
3. Fill in the **Task Text**, as this script does not auto-derive affordance tasks.
4. Set the **Video Codec** to `h264_nvenc` for NVIDIA hardware acceleration.
5. Click **Start Conversion**.

### Resuming

If the conversion is interrupted (crash, cancel, power loss), re-run the tool with the same output directory and dataset name. It will detect the existing partial dataset and offer to **Resume** from where it left off.

## Output

The v3.0 converter produces the standard HuggingFace structure via Parquet meta:

```
output_dir/
  dataset_name/
    meta/
      info.json          # Dataset metadata (FPS, splits)
      episodes/
        chunk-000/
          episode_data.parquet # LeRobot v3 metadata chunk
    videos/
      chunk-000/
        observation.images.endoscope.left/
          episode_000000.mp4
```

*(The v2.1 format is slightly different, outputting direct episode parquet chunks into a `data/` block rather than `meta/episodes/`)*

## Running Without an NVIDIA GPU (CPU-only)

If you don't have an NVIDIA GPU, change two things:

1. **Install CPU-only PyTorch** in step 2 instead:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

1. **Change the Video Codec** dropdown in the GUI from `h264_nvenc` to one of:
   - `h264 (CPU)` -- good compression, reasonable speed, works on any machine
   - `h264_amf` -- if you have an AMD GPU
   - `h264_qsv` -- if you have an Intel CPU with Quick Sync
   - `libsvtav1` -- best compression but very slow (not recommended unless file size is critical)

Everything else works the same.

## Troubleshooting

- **"No module named lerobot"**: Make sure you installed LeRobot (step 3 above).
- **NVENC codec errors**: Your machine doesn't have an NVIDIA GPU or the drivers are missing. Switch the codec dropdown to `h264 (CPU)`.
- **Slow conversion**: Make sure you're using `h264_nvenc` and your source data is on a fast drive (SSD/NVMe).
- **Out of memory**: The tool buffers frames in memory for pipelining. If you run out of RAM, try closing other applications.
