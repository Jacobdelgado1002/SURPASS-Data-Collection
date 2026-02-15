# dVRK Data Collection - Troubleshooting Guide

A comprehensive troubleshooting reference for resolving common issues during dVRK data collection operations.

## Table of Contents

- [Overview](#overview)
- [Quick Diagnostic Commands](#quick-diagnostic-commands)
- [Camera Issues](#camera-issues)
  - [Artifacts in Camera View](#artifacts-in-camera-view)
  - [Camera Not Found](#camera-not-found)
  - [Empty Camera Topics](#empty-camera-topics)
- [Robot System Issues](#robot-system-issues)
  - [Robot Not Responding](#robot-not-responding)
  - [Robot Teleop Drifting](#robot-teleop-drifting)
  - [Empty ROS Topics](#empty-ros-topics)
  - [dVRK System Fails to Start](#dvrk-system-fails-to-start)
- [ROS Core Issues](#ros-core-issues)
  - [ROS Core Not Responding](#ros-core-not-responding)
  - [Multiple ROS Core Instances](#multiple-ros-core-instances)
- [Network and Communication Issues](#network-and-communication-issues)
- [Additional Resources](#additional-resources)

---

## Overview

This guide provides solutions to common issues encountered during dVRK data collection. Each issue includes:

- **Symptom:** What you observe
- **Diagnosis:** How to verify the problem
- **Solution:** Step-by-step fix
- **Verification:** How to confirm the issue is resolved

**General Troubleshooting Approach:**

1. Check the symptoms match the description
2. Run diagnostic commands to confirm the issue
3. Follow solution steps in order
4. Verify the fix worked before resuming data collection

---

## Quick Diagnostic Commands

Use these commands to quickly identify common issues:

```bash
# Check ROS core status
rostopic list

# Check if cameras are publishing
rostopic hz /endopsm1/image_raw
rostopic hz /dvrk_csr/left/image_raw

# Check robot topics
rostopic echo /dvrk/PSM1/position_cartesian_current

# List video devices
ls -l /dev/video*
v4l2-ctl --list-devices

# Check running ROS nodes
rosnode list

# Verify user permissions
groups $USER | grep video

# Check network configuration
echo $ROS_MASTER_URI
echo $ROS_IP
```

---

## Camera Issues

### Artifacts in Camera View

**Symptom:**

- Visual artifacts, noise, or glitches appear in camera feed
- Image quality degraded with static, lines, or color distortions
- Intermittent flickering or corruption

**Diagnosis:**

```bash
# View the affected camera feed
rosrun rqt_image_view rqt_image_view

# Check camera topic rate (should be ~30 Hz)
rostopic hz /endopsm1/image_raw
```

**Solution:**

**Step 1: Restart Camera Node (Soft Reset)**

```bash
# Identify the camera node name
rosnode list | grep camera

# Kill the specific camera node
rosnode kill /endopsm1_camera

# Wait 5 seconds, then relaunch
roslaunch dvrk_video gscam_v4l.launch \
  device:=/dev/video0 \
  camera_name:=endopsm1 \
  images_per_second:=30
```

**Step 2: Power Cycle Camera (Hard Reset)**

If soft reset doesn't work:

```bash
# Stop the camera node
rosnode kill /endopsm1_camera

# Physically unplug the USB cable from the camera
# Wait 10 seconds

# Plug the camera back in
# Wait 5 seconds for device enumeration

# Verify device is detected
v4l2-ctl --list-devices

# Relaunch camera node
roslaunch dvrk_video gscam_v4l.launch \
  device:=/dev/video0 \
  camera_name:=endopsm1 \
  images_per_second:=30
```

**Verification:**

```bash
# Check image feed is clean
rosrun rqt_image_view rqt_image_view

# Verify stable frame rate
rostopic hz /endopsm1/image_raw
```

**✅ Expected Result:**

- Clear image with no artifacts
- Stable ~30 Hz frame rate
- No flickering or corruption

---

### Camera Not Found

**Symptom:**

- Camera launch fails with "device not found" error
- `/dev/video*` device missing
- Camera node crashes immediately after launch

**Diagnosis:**

```bash
# Check if video devices exist
ls -l /dev/video*

# Get detailed device information
v4l2-ctl --list-devices

# Check USB device enumeration
lsusb | grep -i camera
```

**Solution:**

**Step 1: Verify Physical Connection**

1. Check USB cable is securely connected
2. Try a different USB port (preferably USB 3.0)
3. Verify camera has power (LED indicator if present)

**Step 2: Check Device Permissions**

```bash
# Check current permissions
ls -l /dev/video*

# Expected output: crw-rw----+ 1 root video

# Add your user to video group (if not already)
sudo usermod -a -G video $USER

# Apply group changes (logout/login or use)
newgrp video

# Set device permissions (temporary fix)
sudo chmod 666 /dev/video*
```

**Step 3: Reload USB Video Driver**

```bash
# Unload the driver
sudo rmmod uvcvideo

# Wait 3 seconds

# Reload the driver
sudo modprobe uvcvideo

# Verify devices are back
ls -l /dev/video*
```

**Verification:**

```bash
# Device should be visible
v4l2-ctl --list-devices

# Launch camera and check output
roslaunch dvrk_video gscam_v4l.launch \
  device:=/dev/video0 \
  camera_name:=endopsm1
```

**✅ Expected Result:**

- `/dev/video*` devices present
- Camera node launches successfully
- No permission errors

---

### Empty Camera Topics

**Symptom:**

- Camera topic exists but publishes no data
- `rostopic hz <topic>` shows 0 Hz or no messages
- Image viewer shows blank/black screen

**Diagnosis:**

```bash
# Check if topic exists
rostopic list | grep image_raw

# Check publication rate (should be ~30 Hz)
rostopic hz /endopsm1/image_raw

# Check topic info
rostopic info /endopsm1/image_raw

# View camera node output for errors
rosnode info /endopsm1_camera
```

**Solution:**

**Step 1: Verify Camera Node is Running**

```bash
# List active nodes
rosnode list | grep camera

# If camera node is missing, launch it
roslaunch dvrk_video gscam_v4l.launch \
  device:=/dev/video0 \
  camera_name:=endopsm1 \
  images_per_second:=30
```

**Step 2: Check Device Assignment**

```bash
# Identify correct video device
v4l2-ctl --list-devices

# Test different video device numbers if needed
roslaunch dvrk_video gscam_v4l.launch \
  device:=/dev/video2 \
  camera_name:=endopsm1
```

**Step 3: Restart Camera Pipeline**

```bash
# Kill existing camera node
rosnode kill /endopsm1_camera

# Clear any cached parameters
rosparam delete /endopsm1_camera

# Wait 5 seconds

# Relaunch
roslaunch dvrk_video gscam_v4l.launch \
  device:=/dev/video0 \
  camera_name:=endopsm1 \
  images_per_second:=30
```

**Verification:**

```bash
# Should show ~30 Hz
rostopic hz /endopsm1/image_raw

# Should show image
rosrun rqt_image_view rqt_image_view
```

**✅ Expected Result:**

- Topic publishes at ~30 Hz
- Images visible in viewer
- No error messages in camera node terminal

---

## Robot System Issues

### Robot Not Responding

**Symptom:**

- PSM arms do not respond to commands
- Teleoperation not working
- Robot appears frozen or idle
- Typically occurs after extended idle period

**Diagnosis:**

```bash
# Check if robot topics are publishing
rostopic list | grep dvrk

# Check robot state
rostopic echo /dvrk/PSM1/robot_state

# Check if roscore is responding
rostopic list
```

**Solution:**

**Step 1: Restart ROS Core**

```bash
# On Computer #1, stop roscore
# Press Ctrl+C in roscore terminal

# Wait 10 seconds for all nodes to shut down

# Restart roscore
roscore
```

**Step 2: Restart dVRK System**

```bash
# On Computer #2, stop dVRK system
# Press Ctrl+C in dvrk_system terminal

# Wait 15 seconds

# Restart dVRK system
cd ~/catkin_ws_dvrk/src/dvrk/dvrk_config_jhu/jhu-daVinci-Si
rosrun dvrk_robot dvrk_system \
  -j system-SUJ-ECM-MTML-PSM2-MTMR-PSM1-PSM3-Teleop-custom.json \
  -p 0.001 \
  -K
```

**Step 3: Re-home Robot Arms**

```bash
# In dVRK system terminal, press 'h' to home all arms
# Wait for homing to complete

# Press 'e' to enable power
```

**Step 4: Redo Hand-Eye Calibration**

```bash
# Required after restarting system
rosrun dvrk_camera_registration camera_registration.py
```

**Verification:**

```bash
# Check robot topics are active
rostopic hz /dvrk/PSM1/position_cartesian_current

# Try moving the robot (teleoperation or keyboard)
```

**✅ Expected Result:**

- Robot responds to commands
- Position topics publishing at ~1000 Hz
- Teleoperation functional

---

### Robot Teleop Drifting

**Symptom:**

- Robot drifts when master is stationary
- Requires frequent clutching to reset

**Diagnosis:**

```bash
# Monitor teleoperation state
rostopic echo /dvrk/MTMR/teleop_state

# Check for error accumulation
rostopic echo /dvrk/PSM1/position_cartesian_current
```

**Solution:**

**Step 1: Disable Robot in Console**

In the dVRK system terminal (Computer #2):

```bash
# Press 'e' to disable power
# Robot should go into idle state
```

**Step 2: Restart ROS Core**

```bash
# On Computer #1, stop roscore
# Press Ctrl+C

# Wait 10 seconds

# Restart roscore
roscore
```

**Step 3: Restart dVRK System**

```bash
# On Computer #2, restart dVRK system
cd ~/catkin_ws_dvrk/src/dvrk/dvrk_config_jhu/jhu-daVinci-Si
rosrun dvrk_robot dvrk_system \
  -j system-SUJ-ECM-MTML-PSM2-MTMR-PSM1-PSM3-Teleop-custom.json \
  -p 0.001 \
  -K
```

**Step 4: Recalibrate System**

```bash
# Redo hand-eye calibration
rosrun dvrk_camera_registration camera_registration.py
```

**Verification:**

```bash
# Test teleoperation with small movements
# Verify PSM follows MTM accurately
# Check that stationary MTM results in stationary PSM
```

**✅ Expected Result:**

- No drift when MTM is stationary
- PSM accurately follows MTM movements
- Position errors < 2mm

---

### Empty ROS Topics

**Symptom:**

- Robot topics exist but publish no data
- `rostopic hz <topic>` returns 0 Hz
- Position/state information not updating

**Diagnosis:**

```bash
# Check if topics exist
rostopic list | grep PSM1

# Check if data is being published
rostopic hz /dvrk/PSM1/position_cartesian_current

# Check robot state
rostopic echo /dvrk/PSM1/robot_state -n 1
```

**Solution:**

**Step 1: Verify Tools Are Connected**

**⚠️ CRITICAL:** All PSM arms must have tools attached for proper operation.

1. Check each PSM has a tool installed in the cannula
2. Verify tool is fully seated and locked
3. Check for tool recognition (LED indicators on arm)

**Step 2: Re-home Arms**

```bash
# In dVRK system console (Computer #2)
# Press 'h' to home all arms
# Wait for homing sequence to complete (green status indicators)
```

**Step 3: Enable Power**

```bash
# In dVRK system console
# Press 'e' to enable power
# Verify all arms show 'powered' state
```

**Step 4: Verify Topic Publishing**

```bash
# Each PSM should publish at ~1000 Hz
rostopic hz /dvrk/PSM1/position_cartesian_current
rostopic hz /dvrk/PSM2/position_cartesian_current
```

**Verification:**

```bash
# Check multiple robot topics
rostopic list | grep dvrk

# Verify data on each topic
rostopic echo /dvrk/PSM1/position_cartesian_current -n 1
```

**✅ Expected Result:**

- All PSM topics publishing at ~1000 Hz
- Robot state shows 'ENABLED' or 'READY'
- Position data updates correctly

**❌ If Still Not Working:**

- Verify correct tools for each PSM type
- Inspect tool for physical damage
- Check dVRK configuration JSON includes all PSMs

---

### dVRK System Fails to Start

**Symptom:**

- `dvrk_system` command exits with error
- Cannot connect to PSM/MTM arms
- Configuration file not found
- Hardware communication errors

**Diagnosis:**

```bash
# Verify configuration file exists
ls -l ~/catkin_ws_dvrk/src/dvrk/dvrk_config_jhu/jhu-daVinci-Si/*.json

# Check for hardware connection errors in terminal output

# Verify ROS environment
echo $ROS_MASTER_URI
echo $ROS_IP
```

**Solution:**

**Step 1: Verify Hardware Connections**

1. Check all power cables are connected
2. Verify FireWire/Ethernet cables to controllers
3. Check emergency stop is released
4. Verify all controller boxes are powered on

**Step 2: Check Configuration File Path**

```bash
# Navigate to config directory
cd ~/catkin_ws_dvrk/src/dvrk/dvrk_config_jhu/jhu-daVinci-Si

# Verify JSON file exists
ls -l system-SUJ-ECM-MTML-PSM2-MTMR-PSM1-PSM3-Teleop-custom.json

# Check file permissions
chmod 644 *.json
```

**Step 3: Verify ROS Network Configuration**

```bash
# On Computer #2, set correct ROS_MASTER_URI
export ROS_MASTER_URI=http://<Computer1_IP>:11311
export ROS_IP=<Computer2_IP>

# Test connectivity to ROS master
rostopic list
```

**Step 4: Launch with Verbose Output**

```bash
cd ~/catkin_ws_dvrk/src/dvrk/dvrk_config_jhu/jhu-daVinci-Si
rosrun dvrk_robot dvrk_system \
  -j system-SUJ-ECM-MTML-PSM2-MTMR-PSM1-PSM3-Teleop-custom.json \
  -p 0.001 \
  -K
  
# Watch for specific error messages
```

**Verification:**

```bash
# System should start without errors
# Check output for:
#   - "Connecting to PSM1... [OK]"
#   - "Connecting to PSM2... [OK]"
#   - "System ready"

# Verify robot topics exist
rostopic list | grep dvrk
```

**✅ Expected Result:**

- dVRK system launches successfully
- All configured arms connect without errors
- Keyboard control prompt appears
- Robot topics are published

---

## ROS Core Issues

### ROS Core Not Responding

**Symptom:**

- `rostopic list` hangs or times out
- Nodes cannot connect to ROS master
- "Unable to contact ROS master" errors

**Diagnosis:**

```bash
# Try to list topics (should timeout if core is dead)
timeout 5 rostopic list

# Check if roscore process is running
ps aux | grep roscore

# Verify network configuration
echo $ROS_MASTER_URI
```

**Solution:**

**Step 1: Ensure Only One ROS Core Instance**

```bash
# Find all roscore processes
ps aux | grep roscore

# Kill any existing roscore processes
pkill -9 roscore
pkill -9 rosmaster

# Wait 10 seconds for cleanup
sleep 10
```

**Step 2: Restart ROS Core**

```bash
# On Computer #1
roscore
```

**Step 3: Restart All Dependent Nodes**

After roscore is running, restart in this order:

1. dVRK system (Computer #2)
2. Camera nodes
3. Data collection nodes

**Verification:**

```bash
# Should return list of topics within 2 seconds
rostopic list

# Should show roscore node
rosnode list | grep rosout
```

**✅ Expected Result:**

- `rostopic list` responds immediately
- All nodes can connect to ROS master
- No timeout errors

---

### Multiple ROS Core Instances

**Symptom:**

- Conflicting topic data
- Nodes connecting to wrong ROS master
- Unpredictable behavior
- "Address already in use" error when starting roscore

**Diagnosis:**

```bash
# Find all roscore processes
ps aux | grep roscore

# Check if port 11311 is in use
netstat -tuln | grep 11311
```

**Solution:**

**Step 1: Kill All ROS Processes**

```bash
# Kill all roscore instances
pkill -9 roscore
pkill -9 rosmaster
pkill -9 roslaunch

# Verify all killed
ps aux | grep ros
```

**Step 2: Clean ROS Environment**

```bash
# Remove stale ROS logs and PID files
rm -rf ~/.ros/log/*

# Clear any temporary files
rosclean purge
```

**Step 3: Start Fresh ROS Core**

```bash
# On Computer #1 ONLY
roscore
```

**Step 4: Verify Single Instance**

```bash
# Should show only ONE roscore process
ps aux | grep roscore | grep -v grep
```

**Verification:**

```bash
# Check ROS master is responsive
rostopic list

# Verify correct master URI
echo $ROS_MASTER_URI
```

**✅ Expected Result:**

- Only one roscore process running
- All nodes connect to same ROS master
- No conflicting data

---

## Network and Communication Issues

### Cannot Find Video Devices

**Quick Reference:**

```bash
# List all video devices
ls -l /dev/video*

# Get detailed device information
v4l2-ctl --list-devices

# Expected output example:
# USB Camera (usb-0000:00:14.0-1):
#     /dev/video0
#     /dev/video1
# 
# Stereo Camera (usb-0000:00:14.0-2):
#     /dev/video2
#     /dev/video3
```

**If no devices found:**

- Reconnect USB cameras
- Reload USB video driver: `sudo rmmod uvcvideo && sudo modprobe uvcvideo`
- Check different USB ports

---

### ROS Network Configuration Problems

**Symptom:**

- Topics from Computer #2 not visible on Computer #1
- Nodes can't communicate across network

**Solution:**

```bash
# On Computer #2, verify environment variables
echo $ROS_MASTER_URI  # Should be http://<Computer1_IP>:11311
echo $ROS_IP          # Should be Computer #2's IP

# If incorrect, set them:
export ROS_MASTER_URI=http://192.168.1.100:11311  # Example
export ROS_IP=192.168.1.101                        # Example

# Add to ~/.bashrc for persistence
echo "export ROS_MASTER_URI=http://192.168.1.100:11311" >> ~/.bashrc
echo "export ROS_IP=192.168.1.101" >> ~/.bashrc

# Test connectivity
ping <Computer1_IP>
rostopic list  # Should show topics from both computers
```

---

## Additional Resources

### Official Documentation

- [dVRK Wiki](https://github.com/jhu-dvrk/sawIntuitiveResearchKit/wiki) - Official dVRK documentation
- [ROS Tutorials](http://wiki.ros.org/ROS/Tutorials) - Learn ROS fundamentals
- [ROS Troubleshooting](http://wiki.ros.org/ROS/Troubleshooting) - General ROS issues

### Community Support

- [dVRK Google Group](https://groups.google.com/forum/#!forum/dvrk-users) - User community and support
- [GitHub Issues](https://github.com/jhu-dvrk/sawIntuitiveResearchKit/issues) - Report bugs and feature requests

### Related Documentation

- **dVRK Hardware Setup Guide** - Initial system setup and calibration
- **dVRK Data Collection Guide** - Daily startup and recording procedures

---

**Maintainer:** [Jacob M. Delgado López]  
**Last Updated:** [02/15/2026]