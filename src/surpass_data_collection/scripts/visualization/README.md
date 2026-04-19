# Dataset Visualization Utilities

This directory contains scripts and GUI tools used for visually inspecting constructed datasets, specifically Hugging Face LeRobot datasets. 

## Overview

Visualization is critical for confirming that camera alignment, kinematics parsing, and affordance slicing have all been performed cleanly. These tools provide interactive ways to scrub through data.

## Scripts

### visualize_lerobot.py

**Purpose**: An interactive GUI tool to visualize datasets formatted for LeRobot. 

**Key Features**:
- Connects directly to Hugging Face LeRobot cached or local dataset structures.
- Provides a fast, synchronous playback slider rendering all camera streams (e.g., Left Endoscope, Right Endoscope, and Wrist views) simultaneously.
- Overlays or outputs kinematics data aligned precisely with the current frame context.
- Extremely useful for debugging temporal sync issues or testing the bounding and start/end trimming points of affordance slices.

**Usage**:

```bash
python visualize_lerobot.py --dataset_dir /path/to/lerobot/dataset
```
