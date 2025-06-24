# The Third London Summer School in Robotics and AI

This repo provides a tactile robot sorting system, which is for the Robotic Heckathon at the Third London Summer School in Robotics and AI.

🔧 **Hardware Requirement:** Please bring your own laptop, and make sure it has:
At least two USB-A ports (to connect both the robot and the tactile sensor).

💻 **Software Requirement:** Our code has been tested on Ubuntu 20.04 and 22.04. If you're using Windows or macOS, don't worry — the required tools (LeRobot, Hugging Face libraries, PyTorch, and Conda) support these platforms as well. But please note that there may be a port recognition issue on Windows.

## Envrionment Configuration
We recommand you to run this project on conda environment and Ubuntu. Please download and configure conda environment through this link (you can use conda/miniconda, if you haven't install, we recommand the latter). [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#linux)

Create a virtual environment with Python 3.10, using Miniconda
```
conda create -y -n SummerSchool2025 python=3.10 
```

Then activate your conda environment, you have to do this each time you open a shell to use : 
```
conda activate SummerSchool2025
```

Then, you can run the following command to configure the environment.
```
pip install -r requirements.txt
```

Note: For Apple M1 - the pyrealsense2 will through an exception use the commands below to resolve it.
```
pip install pyrealsense2-macosx opencv-python
```

## Step One: Tactile Sensor Manufacture and Test
### Manufacture
Manufacture process is shown as video.

### Test
Find the tactile camera id. Insert the camera into the USB-A port, and then run the following command in terminal. There will be two options: 1. Find camera ID, 2. Show camera feed. You should enter '1' in the terminal, and record the output camera id (e.g., 9) firstly. Then you can run the command again, and choose option 2 to show the camera real time.

```
python Step1_find_tactile.py
``` 

## Step Two: Object Texture Data Collection
Please run the following command to collect and organize image data from a camera for object texture classification tasks. When run, it presents two main options: (1) **Collect images for a class** — prompts you to enter a class label (e.g., "metal", "plastic"), number of samples to collect, and an optional camera ID (default is 0). It then captures images from the specified camera when you press the **spacebar**, and saves them to a common directory. (2) **Split data into train/test** — automatically divides the collected images into training and testing sets while preserving class distribution. This tool is useful for building datasets for image classification experiments and supports flexible camera input.
```
python Step2_data_collection.py
``` 

## Step Three: Texture Classifier Training and Testing
This step implements a full pipeline for training and testing a simple image classification model using handcrafted features, along with a live webcam-based prediction demo. It loads labeled image data from `data/train` and `data/test`, extracts basic features from each image (mean intensity, standard deviation, and edge count via Canny), and trains a **k-Nearest Neighbors (kNN)** classifier. After evaluating the classifier on the test set with both overall and per-class accuracy metrics, it optionally launches a **live webcam demo**, capturing frames and classifying them in real time based on the trained model. The code is modular and ready for extension, such as integrating deep learning models for feature extraction in place of the simple `extract_feature` function. Camera ID can be specified interactively before launching the live prediction.

```
python Step3_texture_classify.py
``` 

## Step Four: Tactile Robot Setup and Test
To build the connection between the robot and your computer, please run the following command to find the robot port.
```
python Step4_find_port.py
``` 

Through run the following command, you can use keyboard to control each joint of the SO-ARM-101 robot. And the tactile image is shown real-time at the same time. Please modify "robot-port" and "camera_id" based on the previous outputs.
```
python Step4_tactile_robot_test.py <robot_port> [camera_id]
``` 
Note: When you run this command first time, it will conduct robot calibration process at first.

## Step Five: Tactile Robot Sorting for Single Object
The "Step5_robot_controller" script defines a `RobotController` class that listens for robot movement commands over the **LCM (Lightweight Communications and Marshalling)** messaging system and sends corresponding joint actions to a connected SO101 robot. It supports commands for moving the robot to predefined regions or performing a grasping action. The script requires a valid serial port for communication with the robot, which can be provided either as a command-line argument or entered interactively at runtime (default: `/dev/ttyACM0`). Upon receiving a `ROBOT_COMMAND` message, it interprets the target area (e.g., `grasp_area`, `region_a`, etc.), executes the relevant joint motions, and publishes a `ROBOT_STATUS` message upon task completion. This tool is useful for integrating robot control into modular systems that rely on message-based task coordination.


The "Step5_tactile_policy" script implements a `PolicyNode` that integrates camera-based object classification with robot control using the **LCM messaging system**. It initializes a k-Nearest Neighbors (kNN) classifier trained on image features (mean intensity, standard deviation, and edge count) extracted from a training dataset. At runtime, the script captures frames from a camera (with configurable camera ID), triggers a "grasp\_area" command to the robot, then classifies the object in view. Based on the predicted class (e.g., *fabrics*, *metal*, *plastics*), it sends a sorting command to the corresponding region. A real-time matplotlib display provides visual feedback, including the current prediction. The camera ID can be supplied as a command-line argument or entered interactively, allowing flexible hardware setups for object sorting and policy automation. You can modify the `classfy_image` function in this script based on **Step Four**.

Please run the following commands on two different terminals in sequence.
```
python Step5_robot_controller.py <robot_port>
```

```
python Step5_tactile_policy.py [camera_id]
```

## Step Six: Continuous Tactile Robot Sorting
Modify the state machine part in `Step5_tactile_policy` script to achieve continuous tactile robot sorting.