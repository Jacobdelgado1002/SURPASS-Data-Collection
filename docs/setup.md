# Open-H Data Collection Pipeline

A ROS-based data collection system for the da Vinci Research Kit (dVRK).

## Table of Contents
- [Prerequisites](#prerequisites)
- [dVRK System Setup](#dvrk-system-setup)
- [Data Collection Setup](#data-collection-setup)
- [Running Data Collection](#running-data-collection)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- ROS installed and configured
- dVRK hardware properly connected
- Camera devices configured and accessible
- Required ROS packages:
  - `dvrk_robot`
  - `dvrk_video`
  - `rqt_mypkg`

## dVRK System Setup

### IRL (In Real Life) Hardware Setup

Complete these steps before starting the dVRK system:

#### 1. Hardware Connection
- Connect all dVRK hardware components to the robot system
- Verify power connections and communication cables (cameras, tools, etc.)

#### 2. Camera Calibration (One-time setup)
Perform intrinsic camera calibration once per camera installation:
```bash
rosrun dvrk_camera_registration camera_calibration.py
```

**Note:** This only needs to be done once unless cameras are physically moved or replaced.

#### 3. Hand-Eye Calibration (Required at each startup)
Perform hand-eye calibration every time the system restarts to establish the spatial relationship between the camera and robot manipulator.

**Run the registration script:**
```bash
rosrun dvrk_camera_registration camera_registration.py
```

#### 4. Verify Calibration

After calibration, visualize the gripper pose to verify accuracy:
```bash
rosrun dvrk_camera_registration vis_gripper_pose.py \
  -p PSM2 \
  -c /dvrk_csr/left \
  -H PSM2-registration-open-cv.json
```

**Parameters:**
- `-p`: PSM arm identifier (PSM1, PSM2, or PSM3)
- `-c`: Camera topic name
- `-H`: Hand-eye calibration file (JSON format)

**Expected Result:** The visualization should show the gripper pose accurately overlaid on the camera feed.

### Starting the dVRK System

After completing the IRL setup, launch the main dVRK system:
```bash
cd catkin_ws_dvrk/src/dvrk/dvrk_config_jhu/jhu-daVinci-Si
rosrun dvrk_robot dvrk_system \
  -j system-SUJ-ECM-MTML-PSM2-MTMR-PSM1-PSM3-Teleop-custom.json \
  -p 0.001 \
  -K
```

**Parameters:**
- `-j`: JSON configuration file for the dVRK system
- `-p`: Period/rate setting (0.001 = 1000 Hz)
- `-K`: Keyboard control mode

## Data Collection Setup

### 1. Start ROS Core

In a new terminal:
```bash
cd ~/Desktop
roscore
```

### 2. Launch Stereo Camera

In a new terminal:
```bash
roslaunch dvrk_video v4l_stereo_goovis.launch stereo_rig_name:=dvrk_csr
```

### 3. Launch PSM1 Endoscope Camera

In a new terminal:
```bash
roslaunch dvrk_video gscam_v4l.launch \
  device:=/dev/video[X] \
  camera_name:=endopsm1
```

**Note:** Replace `[X]` with the actual video device number (e.g., `/dev/video0`)

### 4. Launch PSM2 Endoscope Camera

In a new terminal:
```bash
roslaunch dvrk_video gscam_v4l.launch \
  device:=/dev/video[Y] \
  camera_name:=endopsm2
```

**Note:** Replace `[Y]` with the actual video device number (e.g., `/dev/video2`)

## Running Data Collection

Once all components are launched, start the data collection:
```bash
cd ~/catkin_ws_dvrk/src
roslaunch rqt_mypkg record_op_pedals.launch
```

This will begin recording operational data and pedal inputs.

## Troubleshooting

### Finding Video Device Numbers

To identify available video devices:
```bash
ls -l /dev/video*
v4l2-ctl --list-devices
```

### See Video Device view

```bash
rqt_image_view
```

### Common Issues

- **ROS core not responding:** Ensure only one `roscore` instance is running
- **Camera not found:** Verify device permissions with `ls -l /dev/video*` and ensure your user is in the `video` group
- **dVRK system fails to start:** Check that all hardware connections are secure and the JSON configuration path is correct

## Additional Resources

- [dVRK Documentation](https://github.com/jhu-dvrk/sawIntuitiveResearchKit/wiki)
- [ROS Tutorials](http://wiki.ros.org/ROS/Tutorials)

---

**Maintainer:** [Jacob M. Delgado López]  
**Last Updated:** [11/5/2025]