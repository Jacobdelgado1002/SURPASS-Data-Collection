# LeRobot Conversion Scripts

This directory contains scripts responsible for converting the filtered and post-processed surgical data into the standardized Hugging Face LeRobot format. These pipelines ensure the data is properly schema-ed for deep learning models that ingest robot kinematics and multi-view video streams.

## Overview

Conversion scripts take the output of `post_processing` (which is sliced, synchronized, and optionally flattened) and write out the formal LeRobot datasets containing chunked `.parquet` metadata and `.mp4` video streams. They also define the specific feature schema expected by the models (e.g. state size, action sizing, and coordinate definitions).

## Scripts

### `dvrk_zarr_to_lerobot.py`

**Purpose**: The central processing script for transforming standard DVRK/Zarr structures into LeRobot datasets. 

**Key Features**:
- Defines the LeRobot metadata schema (states, actions, observations, multi-camera shapes).
- Converts frame-based sequences into chunked dataset files alongside standard video compression.
- Maps internal CSV action schemas to standard reinforcement learning action spaces.

## High-Performance Accelerated Conversion

For large-scale processing, we heavily depend on a high-performance variant located in the `accelerated-dvrk-lerobot-converter` subdirectory. 

This submodule provides highly optimized CPU/GPU hybrid encoder binaries and GUI frontends to avoid typical Python overhead, making dataset packaging substantially faster.

Please read the [Accelerated Converter README](accelerated-dvrk-lerobot-converter/README.md) for details on:
- **`dvrk_lerobot_converter_v2.1.py`** (The primary CLI entry for accelerated builds)
- **`dvrk_lerobot_converter_gui.py`** (GUI for LeRobot v3.0 / annotated datasets)
- **`dvrk_lerobot_converter_gui_v2.1.py`** (GUI for Legacy Flattened Pipeline)
