# OpenH Surgical Robotics Data Collection

A comprehensive multimodal processing pipeline designed to transform raw dVRK (da Vinci Research Kit) teleoperation data into training-ready machine learning datasets (like Hugging Face LeRobot format).

## Overview

The `OpenH-Data-Collection` suite processes synchronization, filtering, slicing, reformatting, and LeRobot conversion for high-frequency kinematics and multiple surgical camera streams (Stereo Endoscope & Wrist Cams).

## Documentation Map

- **Pipeline Documentation & Source**: [src/surpass_data_collection](src/surpass_data_collection)
  - Find all sub-modules for video processing, filtering, array processing, and dataset conversion here.
- **Hardware Setup & dVRK Docs**: [docs/](docs/)

## Getting Started

To get started with the full Python pipeline, see the central guide at **[src/surpass_data_collection/README.md](src/surpass_data_collection/README.md)**.
