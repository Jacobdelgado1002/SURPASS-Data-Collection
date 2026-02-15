# da Vinci Research Kit (dVRK) - Hardware Setup and Calibration Guide

A comprehensive guide for setting up, configuring, and calibrating the da Vinci Research Kit (dVRK) hardware components and camera systems.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
  - [Software Requirements](#software-requirements)
  - [Hardware Requirements](#hardware-requirements)
- [dVRK System Setup](#dvrk-system-setup)
  - [Hardware Connection](#hardware-connection)
  - [Camera Calibration](#camera-calibration)
  - [Hand-Eye Calibration](#hand-eye-calibration)

---

## Overview

This guide provides step-by-step instructions for the initial setup and calibration of the da Vinci Research Kit (dVRK) robotic surgical system

## Prerequisites

### Software Requirements

- ROS installed and configured
- dVRK hardware properly connected
- Camera devices configured and accessible
- Required ROS packages:
  - `dvrk_robot`
  - `dvrk_video`
  - `rqt_mypkg`

### Hardware Requirements

- ✅ dVRK System (Si or Classic variant)
- ✅ Stereo endoscope camera (with proper calibration rig)
- ✅ PSM wrist cameras (typically 2-3 units)
- ✅ ArUco calibration markers (for hand-eye calibration)
- ✅ Checkerboard calibration pattern (for intrinsic calibration)
- ✅ Two networked computers:
  - **Computer #1:** ROS Master (roscore)
  - **Computer #2:** dVRK control system

## dVRK System Setup

### Hardware Connection

- Connect all dVRK hardware components to the robot system
- Verify power connections and communication cables (cameras, tools, etc.)
- Make sure wrist cameras have a 1 inch distance from tip and are aligned correctly

### Camera Calibration

**Frequency:** Once per camera installation or replacement

**Duration:** ~30 minutes per stereo pair

#### Intrinsic Calibration

This determines internal camera parameters (focal length, distortion, etc.).

##### Step 1: Capture Calibration Images

```bash
cd ~/catkin_ws_dvrk/src/dvrk_camera_registration
rosrun dvrk_camera_registration save_sync_images_gstream.py
```

**Instructions:**

1. Hold the checkerboard pattern in front of the cameras
2. Press **ENTER** to capture an image
3. Move the checkerboard to different:
   - Positions (center, corners, edges)
   - Angles (tilted, rotated)
   - Depths (near and far)
4. Capture **33 images total** with good variety

**✅ Success Criteria:**

- Images saved to: `~/catkin_ws_dvrk/src/dvrk_camera_registration/calibration_images/`
- All images show the complete checkerboard
- Variety of poses captured

#### Step 2: Compute Calibration Parameters

```bash
rosrun dvrk_camera_registration stereo_calib_GStream.py
```

**Expected Output:**

```text
Calibration successful!
RMS Error: 0.3-0.5 pixels (acceptable range)
Calibration files saved:
  - camera_matrix_left.yaml
  - camera_matrix_right.yaml
  - stereo_params.yaml
```

**❌ Troubleshooting:**

- If RMS error > 1.0: Recapture images with better checkerboard visibility
- If calibration fails: Verify checkerboard dimensions in config file

### Hand-Eye Calibration

**Frequency:** **Every system startup** (calibration is lost on restart)

**Duration:** ~5 minutes

**⚠️ CRITICAL:** If you move any PSM arm after calibration, you **MUST** recalibrate.

This establishes the transformation between the camera frame and robot base frame.

#### Step 1: Prepare ArUco Markers

- Attach ArUco markers to each PSM wrist
- Ensure markers are clearly visible to the stereo camera
- Markers should be planar and unobstructed

#### Step 2: Run Calibration

```bash
rosrun dvrk_camera_registration camera_registration.py
```

**Instructions:**

1. The script will start capturing data automatically
2. Move each PSM arm smoothly through various positions:
   - Different X, Y, Z locations
   - Different orientations (roll, pitch, yaw)
   - Keep markers in camera view
3. Continue for **60-90 seconds per arm**
4. Press **Ctrl+C** when done

**✅ Success Criteria:**

```text
Registration completed for PSM1
  - Transform saved: PSM1-registration-open-cv.json
  - Reprojection error: < 5 pixels
Registration completed for PSM2
  - Transform saved: PSM2-registration-open-cv.json
  - Reprojection error: < 5 pixels
```

#### Step 3: Verify Calibration

⚠️ Important: Replace <PSM_ID> and <CAMERA_SIDE> below with your actual setup values.

- <PSM_ID> must be one of: PSM1, PSM2, or PSM3

- <CAMERA_SIDE> must be either: left or right

```bash
rosrun dvrk_camera_registration vis_gripper_pose.py \
  -p <PSM_ID> \
  -c /dvrk_csr/<CAMERA_SIDE> \
  -H <PSM_ID>-registration-open-cv.json
```

**Parameters Explained:**

- `-p`: PSM identifier (PSM1, PSM2, or PSM3)
- `-c`: Camera topic name (/dvrk_csr/left or /dvrk_csr/right)
- `-H`: Path to hand-eye calibration JSON file

**✅ Expected Result:**

- A visualization window shows the camera feed
- A 3D gripper overlay appears aligned with the actual gripper
- Overlay follows gripper movement accurately (< 5mm offset)

**❌ If visualization is misaligned:**

- Repeat hand-eye calibration with more varied arm movements
- Verify ArUco markers are clean and well-lit
- Check that correct calibration file is specified

---

**Maintainer:** [Jacob M. Delgado López]  
**Last Updated:** [02/15/2026]
