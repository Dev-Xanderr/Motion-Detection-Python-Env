# Real-Time Pose and Object Detection Projects

This File is needed for the caffe model (https://limewire.com/d/y9b1m#GYUkne1Xc4)
- Download and extract it into the the folder with the projects


This repository contains three Python scripts demonstrating real-time computer vision applications using OpenCV, MediaPipe, and deep learning models. Each script processes webcam video input for distinctive functionalities such as pose estimation, holistic landmark detection, and object detection with motion-triggered image replacement.

---

## Files Overview

### 1. `mediapipe_holistic_arm_movement.py`

This script leverages **MediaPipe Holistic** to perform real-time detection of face, hand, and body landmarks using the webcam. It visualizes these landmarks with styled colors and implements logic to detect specific arm movements such as arms raised or lateral hand motion.

**Key Features:**
- Real-time holistic pose and landmark detection on face, hands, and body.
- Styled visualization of landmarks with color-coded connections.
- Arm movement analysis with console messaging.
- Uses MediaPipe Python API and OpenCV.

---

### 2. `object_detection_motion_replace.py`

This script performs real-time object detection using a pre-trained **TensorFlow SSD MobileNetV2** model via OpenCV's Deep Neural Network (DNN) module to detect persons, TVs, and keyboards. It incorporates simple motion detection by frame differencing and replaces detected objects with corresponding images only when significant movement occurs in the scene.

**Key Features:**
- Real-time object detection from webcam feed.
- Motion detection via frame differencing and thresholding.
- Conditional replacement of detected objects with custom images (person, TV, keyboard).
- Interactive display with OpenCV.

**Dependencies:** TensorFlow model files (.pb, .pbtxt) and class labels file (`coco_class_labels.txt`).

---

### 3. `Load-a-Caffe-Model.py`

This script loads a **Caffe** deep learning model for real-time human pose estimation using OpenCV's DNN module. It detects 15 keypoints of the human body, draws the skeleton and keypoints on the webcam feed, and implements logic to detect specific arm positions and movements.

**Key Features:**
- Real-time pose keypoint detection using a Caffe model.
- Drawing skeleton and labeled keypoints on camera frames.
- Detects if right arm or both arms are raised above shoulder level.
- Recognizes basic lateral arm movements (right to left and left to right).
- Visual feedback via OpenCV windows.

---

## Requirements

- Python 3.7 or higher
- OpenCV (`opencv-python`)
- MediaPipe (only for `mediapipe_holistic_arm_movement.py`)
- NumPy
- Pretrained model files for respective scripts:
  - TensorFlow SSD MobileNetV2 model and config for `object_detection_motion_replace.py`
  - Caffe prototxt and weights for `Load-a-Caffe-Model.py`

---

## Installation
pip install opencv-python numpy mediapipe


Download required model files and place them accordingly as specified in the scripts.

---

## Usage

Run the desired script in your terminal or within an IDE such as VSCode:

python mediapipe_holistic_arm_movement.py
python object_detection_motion_replace.py
python Load-a-Caffe-Model.py


Press the `ESC` key to quit the webcam video window in all scripts.

---

## Notes

- Ensure your webcam is connected and accessible.
- Adjust confidence thresholds and parameters in the code to improve detections based on lighting and environment.
- The scripts are primarily for demonstration and can be extended with additional actions or integrated into larger applications.

---

## License

This project is licensed under the Apache 2.0 License.

---

## Acknowledgments

- MediaPipe by Google
- OpenCV community
- TensorFlow SSD MobileNetV2 model authors
- Public Caffe pose estimation models

This README provides each script’s purpose, features, requirements, installation, and usage details, helping users understand and run your projects efficiently.
