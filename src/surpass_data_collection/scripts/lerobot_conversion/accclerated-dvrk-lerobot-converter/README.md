# DVRK to LeRobot v2.1 Converter

A GUI tool to convert DVRK surgical robot datasets into [LeRobot v2.1](https://github.com/huggingface/lerobot) format with timestamp-based alignment across multiple camera views.

## Features

- PyQt5 GUI -- no command-line arguments needed
- Timestamp-based alignment across 4 camera streams (left/right endoscope, left/right wrist)
- NVIDIA GPU-accelerated video encoding (10-50x faster than CPU)
- Resume support -- safely resume after crashes or interruptions
- Parallel video encoding and pipelined frame processing for speed

## Requirements

- **Python 3.10+**
- **NVIDIA GPU** with up-to-date drivers (recommended for best performance)
- A fast local drive (SSD/NVMe) for intermediate files

## Expected Input Data Structure

Each episode should be a subdirectory containing:

```
source_data/
  episode_001/
    left_img_dir/        # Left endoscope images (frame{timestamp}_left.jpg)
    right_img_dir/       # Right endoscope images
    endo_psm1/           # PSM1 wrist camera images
    endo_psm2/           # PSM2 wrist camera images
    ee_csv.csv           # End-effector CSV with timestamp, state, and action columns
  episode_002/
    ...
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

```bash
pip install git+https://github.com/huggingface/lerobot.git
```

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

## Usage

```bash
python dvrk_lerobot_converter_gui.py
```

The GUI will open. Follow these steps:

1. **Browse** to select your source data directory (the folder containing episode subdirectories).
2. **Set the output directory** where the LeRobot dataset will be created.
3. **Enter a dataset name** (e.g. `dvrk_wound_closure`).
4. **Fill in metadata**: task description, PSM1/PSM2 tool names, FPS.
5. The **Video Codec** defaults to `h264_nvenc (NVIDIA GPU)` for best performance. Leave this as-is unless you need to change it (see below).
6. Click **Start Conversion**.

### Resuming

If the conversion is interrupted (crash, cancel, power loss), re-run the tool with the same output directory and dataset name. It will detect the existing partial dataset and offer to **Resume** from where it left off.

## Output

The tool creates a standard LeRobot v2.1 dataset:

```
output_dir/
  dataset_name/
    meta/
      info.json          # Dataset metadata (FPS, features, splits, etc.)
    data/
      chunk-000/
        episode_000000.parquet
        episode_000001.parquet
        ...
    videos/
      chunk-000/
        observation.images.endoscope.left/
          episode_000000.mp4
          ...
        observation.images.endoscope.right/
          ...
        observation.images.wrist.left/
          ...
        observation.images.wrist.right/
          ...
```

## Running Without an NVIDIA GPU (CPU-only)

If you don't have an NVIDIA GPU, change two things:

1. **Install CPU-only PyTorch** in step 2 instead:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

2. **Change the Video Codec** dropdown in the GUI from `h264_nvenc` to one of:
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
