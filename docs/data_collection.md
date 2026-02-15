# SURPASS Data Collection Pipeline

A ROS-based data collection system for the da Vinci Research Kit (dVRK).

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Daily Startup Procedure](#daily-startup-procedure)
  - [Hardware Connection](#hardware-connection)
  - [Camera Calibration](#camera-calibration)
  - [Hand-Eye Calibration](#hand-eye-calibration)

## Overview

This pipeline enables synchronized data collection from multiple sensors on the dVRK system, including:

- PSM (Patient Side Manipulator) wrist cameras
- Stereo endoscope camera
- Robot kinematics and joint states
- Pedal inputs and operational data

## System Architecture

```text
┌─────────────────┐
│  Computer #1    │
│   (ROS Master)  │
│    roscore      │
└────────┬────────┘
         │
         │ Network
         │
┌────────┴────────┐
│  Computer #2    │
│  dVRK Control   │
│  ├─ PSM1/2/3    │
│  ├─ MTM L/R     │
│  ├─ ECM         │
│  └─ SUJ         │
└─────────────────┘
         │
         ├─ Stereo Camera
         ├─ PSM Wrist Cameras
         └─ Pedal Inputs
```

## Daily Startup Procedure

Follow this sequence every time you start the system for data collection.

### Terminal 1: ROS Master (Computer #1)

```bash
roscore
```

**✅ Expected Output:**

```text
... logging to /home/user/.ros/log/...
...
started core service [/rosout]
```

**Keep this terminal running throughout the session.**

---

### Terminal 2: dVRK System (Computer #2)

```bash
cd ~/catkin_ws_dvrk/src/dvrk/dvrk_config_jhu/jhu-daVinci-Si
rosrun dvrk_robot dvrk_system \
  -j system-SUJ-ECM-MTML-PSM2-MTMR-PSM1-PSM3-Teleop-custom.json \
  -p 0.001 \
  -K
```

**Parameters:**

- `-j`: Configuration file (defines which arms are active)
- `-p 0.001`: Control loop period (1 kHz = 1000 Hz)
- `-K`: Enable keyboard control mode

**✅ Expected Output:**

```text
Loading configuration: system-SUJ-ECM-MTML-PSM2-MTMR-PSM1-PSM3-Teleop-custom.json
Connecting to PSM1... [OK]
Connecting to PSM2... [OK]
Connecting to PSM3... [OK]
Connecting to MTML... [OK]
Connecting to MTMR... [OK]
System ready. Press 'h' for help.
```

---

### Terminal 3: Stereo Endoscope Camera

```bash
roslaunch dvrk_video v4l_stereo_goovis.launch stereo_rig_name:=dvrk_csr
```

**✅ Expected Output:**

```text
process[dvrk_csr/left-1]: started with pid [######]
process[dvrk_csr/right-2]: started with pid [######]
Publishing stream...
Started stream
Publishing stream...
Started stream
```

---

### Terminal 4 and 5: PSM Wrist Cameras

⚠️ Important: Replace <PSM_ID> and <CAMERA_SIDE> below with your actual setup values.

- <VIDEO_NUMBER> must check rostopics but usually one of: video0, video2

- <PSM_NAME> must be one of: endopsm1, endopsm2, or endopsm3

```bash
roslaunch dvrk_video gscam_v4l.launch \
  device:=/dev/video0 \
  camera_name:=<PSM_NAME> \
  images_per_second:=30
```

**✅ Expected Output:**

```text
process[<PSM_NAME>-1]: started with pid [######]
Publishing stream...
Started stream
```

---

### Terminal 6: Camera Viewer (Verification)

```bash
rosrun rqt_image_view rqt_image_view
```

**Instructions:**

1. Select each camera topic from the dropdown menu
2. Verify image quality:
   - ✅ Correct exposure and focus
   - ✅ No motion blur
   - ✅ Proper lighting
   - ✅ Frames updating smoothly

**Available Topics:**

- `/dvrk_csr/left/image_raw` - Left endoscope
- `/dvrk_csr/right/image_raw` - Right endoscope
- `/endopsm1/image_raw` - PSM1 wrist
- `/endopsm2/image_raw` - PSM2 wrist
- `/endopsm3/image_raw` - PSM3 wrist (if applicable)

---

### Terminal 7: Hand-Eye Calibration

**⚠️ REQUIRED at each startup**

```bash
# Perform calibration for all PSMs
rosrun dvrk_camera_registration camera_registration.py

# Verify each calibration
rosrun dvrk_camera_registration vis_gripper_pose.py -p PSM1 -c /dvrk_csr/left -H PSM1-registration-open-cv.json
rosrun dvrk_camera_registration vis_gripper_pose.py -p PSM2 -c /dvrk_csr/right -H PSM2-registration-open-cv.json
```

See [Hand-Eye Calibration](#hand-eye-calibration) section for detailed instructions.

---

## Data Collection Workflow

Once all components are running and verified:

### Terminal 8: Start Recording

```bash
cd ~/catkin_ws_dvrk/src
roslaunch rqt_mypkg record_op_pedals.launch
```

**✅ Expected Output:**

```text
process[record-1]: started with pid [######]
```

### Recording Controls

- **Press COAG pedal continuously to record**: Toggle recording on/off
- **'CTRL+C'**: Quit and save all data

### During Recording

**Best Practices:**

- Record in 5-10 minute segments
- Leave 2-3 seconds before/after each task
- Perform smooth, deliberate movements
- Avoid occluding wrist cameras
- Keep ArUco markers visible (if needed for tracking)

---

**Maintainer:** [Jacob M. Delgado López]  
**Last Updated:** [02/15/2026]
