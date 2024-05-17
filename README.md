# Bimanual Teleop Controller

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Description

This package provides a bimanual teleoperation controller for controlling a robotic system with two arms. It allows users to remotely control the robot's arms using a teleoperation interface for separated arm mission and bimanual manipulation task.

## Features

- Intuitive and user-friendly teleoperation interface
- Real-time control of both arms
- Support for various robotic systems and manipulators

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/bimanual_teleop_controller.git
   ```
2. Install required dependencies:
    ```bash
    rosdep install --from-paths src --ignore-src -r -y
    ```

#### Note:
To sync time between host machine and ssh machine
``` bash
ssh -t pr2@10.68.0.1 sudo date --set @$(date -u +%s) 
```

## Contributors

* Initial work by **Anh Minh Tu** (https://github.com/minhtugonnabelit)