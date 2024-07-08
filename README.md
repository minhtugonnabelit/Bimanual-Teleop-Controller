# Bimanual Teleop Controller

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Description

This package provides a bimanual teleoperation controller for controlling a robotic system with two arms. It allows users to remotely control the robot's arms using a teleoperation interface for separated arm mission and bimanual manipulation task. Current setup of this package is used for implementing the bimanual teleoperation scheme on the PR2 robot in order to neglect the complicated and unreliable multiple ROS-master communication.


## Installation


To get this work on PR2, please ensure that the ROS controller having on PR2 are up-to-dated as the joint group velocity controller interface does not exist in the old version driver of this robot. 

This package calculation depends heavily on the robotics toolbox provided by Peter Corke and Jesse Haviland.
The python module can be installed by standard pip tool
```bash
pip install robotics-toolbox-python
```

1. Clone the repository into ROS workspace:

   ```bash
   git clone https://github.com/your-username/bimanual_teleop_controller.git
   ```


2. Install required dependencies for ROS:
    ```bash
    rosdep install --from-paths src --ignore-src -r -y
    ```


## Contributors

* Initial work by **Anh Minh Tu** (https://github.com/minhtugonnabelit)