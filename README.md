# The Third London Summer School in Robotics and AI

This repo provides a tactile robot sorting system, which is for the Robotic Heckathon at The Third London Summer School in Robotics and AI.

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

## Step One: Tactile Sensor Manufacture and Test
### Manufacture
Manufacture process is shown as video.

### Test
Find the tactile camera id. Insert the camera into the USB-A port, and then run the following command in terminal. There will be two options: 1. Find camera ID, 2. Show camera feed. You should enter '1' in the terminal, and record the output camera id (e.g., 9) firstly. Then you can run the command again, and choose option 2 to show the camera real time.

```
python Step1_find_tactile.py
``` 

## Step Two: Object Texture Data Collection
```
python Step2_data_collection.py
``` 

## Step Three: Texture Classifier Training and Testing
```
python Step3_texture_classify.py
``` 

## Step Four: Tactile Robot Setup and Test
```
python Step4_find_port.py
``` 


```
python Step4_tactile_robot_test.py
``` 
Note: When you run this command first time, it will conduct robot calibration process at first.

## Step Five: Tactile Robot Sorting for Single Object

```
python Step5_robot_controller.py
```

```
python Step5_tactile_policy.py
```

## Step Six: Continuous Tactile Robot Sorting
