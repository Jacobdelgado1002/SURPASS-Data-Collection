# Dataset Validation Utilities

This directory contains utility scripts to validate the structural integrity, data completeness, and formatting of constructed datasets before they are used for downstream machine learning tasks.

## Overview

The validation scripts check various conditions that frequently arise when building and restructuring multimodal surgical robotics datasets. They ensure correctness at the frame level (e.g., matching image timestamps) and dataset level (e.g., proper metadata).

## Scripts

### validate_open_h.py

**Purpose**: Verifies that a dataset conforms specifically to the Open-H dataset standard organization and timestamp semantics. 

**Key Validations**:
- Directory structure checking (expecting left, right, endoscope directories and csv files).
- Frame validation to ensure no missing frames between image views and the kinematics CSV.
- Ensuring valid temporal ranges and ordering.

### validate_surpass.py

**Purpose**: A validation suite tailored directly for the SURPASS surgical robotics dataset format.

**Key Validations**:
- Confirms the inclusion of correct tissue metadata and task structures.
- Parses deeper affordance annotations or extra modalities specific to SURPASS datasets.
- Summarizes any missing or irregular sub-episodes.

## Usage

```bash
# Validate an Open-H format dataset
python validate_open_h.py /path/to/dataset

# Validate a SURPASS format dataset
python validate_surpass.py /path/to/surpass_data
```
