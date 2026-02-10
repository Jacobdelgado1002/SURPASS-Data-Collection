# SURPASS Data Collection Pipeline

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
- Make sure wrist cameras have a 1 inch distance from tip and are aligned correctly

#### 2. Camera Calibration (One-time setup)
Perform intrinsic camera calibration once per camera installation:

This will start a script that will capture images every time you press ENTER. You must take the checkerboard and move it around to capture images of the board at different positions and angles. (33 captured images)
```bash
rosrun dvrk_camera_registration save_sync_images_gstream.py
```

This will take the saved images and perform stereo calibration.
```bash
rosrun dvrk_camera_registration stereo_calib_GStream.py
```

**Note:** This calibration only needs to be done once, unless the endoscope camera is replaced.

#### 3. Hand-Eye Calibration (Required at each startup)
Perform hand-eye calibration every time the system restarts to establish the spatial relationship between the camera and robot manipulator.

**Run the registration script:**
Put the Aruco tag on the wrists and move it around for a bit in order to collect data.
```bash
rosrun dvrk_camera_registration camera_registration.py
```

#### 4. Verify Calibration

After calibration, visualize the gripper pose to verify accuracy:
```bash
rosrun dvrk_camera_registration vis_gripper_pose.py \
  -p [X] \
  -c /dvrk_csr/[X]t \
  -H [X]-registration-open-cv.json
```

**Parameters:**
- `-p`: PSM arm identifier (PSM1, PSM2, or PSM3)
- `-c`: Camera topic name
- `-H`: Hand-eye calibration file (JSON format)

**Expected Result:** The visualization should show the gripper pose accurately overlaid on the camera feed.

### Starting the dVRK System

### 1. Start ROS Core
Run in a terminal in computer #1
```bash
roscore
```

### 2. Launch the main dVRK system
After completing the IRL setup, launch the main dVRK system in computer #2:
```bash
cd catkin_ws_dvrk/src/dvrk/dvrk_config_jhu/jhu-daVinci-Si
rosrun dvrk_robot dvrk_console_json \
  -j console-SUJ-ECM-MTML-PSM2-MTMR-PSM1-PSM3-Teleop.json
```

**Parameters:**
- `-j`: JSON configuration file for the dVRK system
- `-p`: Period/rate setting (0.001 = 1000 Hz)
- `-K`: Keyboard control mode

## Data Collection Setup

### 1. Start ROS Core

In a new terminal:
```bash
roscore
```

### 2. Launch Stereo Camera

In a new terminal:
```bash
roslaunch dvrk_video stereo_decklink_goovis.launch stereo_rig_name:=jhu_daVinci images_per_seconds:=30
```

### 3. View PSM wrist cameras
```bash
rosrun rqt_image_view rqt_image_view
```

<!-- ### 3. Launch PSM1 Endoscope Camera

In a new terminal:
```bash
roslaunch dvrk_video gscam_v4l.launch \
  device:=/dev/video[X] \
  camera_name:=endopsm1 \
  images_per_second:=30 
```

**Note:** Replace `[X]` with the actual video device number (e.g., `/dev/video0`)

### 4. Launch PSM2 Endoscope Camera

In a new terminal:
```bash
roslaunch dvrk_video gscam_v4l.launch \
  device:=/dev/video[Y] \
  camera_name:=endopsm2 \
  images_per_second:=30
```

**Note:** Replace `[Y]` with the actual video device number (e.g., `/dev/video2`) -->

## Running Data Collection

Once all components are launched, start the data collection:
```bash
cd ~/catkin_ws_dvrk/src
roslaunch rqt_mypkg record_op_pedals_toggle.launch
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
**Last Updated:** [02/9/2026]
