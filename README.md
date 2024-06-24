# Bimanual Teleop Controller

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Description

This package provides a bimanual teleoperation controller for controlling a robotic system with two arms. It allows users to remotely control the robot's arms using a teleoperation interface for separated arm mission and bimanual manipulation task.


## Installation

This package calculation depends heavily on the robotics toolbox provided by Peter Corke and Jesse Haviland.
The python module can be installed by standard pip tool
```bash
pip install robotics-toolbox-python
```

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/bimanual_teleop_controller.git
   ```
2. Install required dependencies for ROS:
    ```bash
    rosdep install --from-paths src --ignore-src -r -y
    ```


#### Note:
To sync time between host machine and ssh machine
``` bash
ssh -t host@X.X.X.X sudo date --set @$(date -u +%s) 
```

## Contributors

* Initial work by **Anh Minh Tu** (https://github.com/minhtugonnabelit)