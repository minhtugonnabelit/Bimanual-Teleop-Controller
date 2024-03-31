import numpy as np
import spatialmath as sm
import spatialgeometry as geometry
import roboticstoolbox as rtb
from swift import Swift
from scipy import linalg
from copy import deepcopy
import threading

from utility import reorder_values


class FakePR2:

    r"""
    ### Class to simulate PR2 on Swift environment

    This will initialize the Swift environment with robot model without any ROS components. 
    This model will be fed with joint states provided and update the visualization of the virtual frame . """

    def __init__(self, launch_visualizer) -> None:

        self._launch_visualizer = launch_visualizer
        self._robot = rtb.models.PR2()
        self._is_collapsed = False
        self._tool_offset = {
            'left': np.eye(4),
            'right': np.eye(4)
        }

        self._arms_frame = {
            'left': {
                'end': self._robot.grippers[1],
                'start': 'l_shoulder_pan_link'
            },
            'right': {
                'end': self._robot.grippers[0],
                'start': 'r_shoulder_pan_link'
            }
        }

        if self._launch_visualizer:
            self._env = Swift()
            self._env.launch()
            self.init_visualization()
            self._thread = threading.Thread(target=self.timeline)
            self._thread.start()

    def timeline(self):
        r"""
        Timeline function to update the visualization
        :return: None
        """
        while not self._is_collapsed:
            self._env.step(0.1)

    def set_constraints(self, virtual_pose: np.ndarray):
        r"""
        Set the kinematics constraints for the robot
        :param virtual_pose: virtual pose on robot base frame
        :return: None
        """

        self._tool_offset['left'] = linalg.inv(self._robot.fkine(
            self._robot.q, end=self._arms_frame['left']['end'])) @ virtual_pose
        self._tool_offset['right'] = linalg.inv(self._robot.fkine(
            self._robot.q, end=self._arms_frame['right']['end'])) @ virtual_pose

        return True

    def set_joint_states(self, joint_states: list):
        r"""
        Set the joint states of the arms only
        :param joint_states: list of joint states
        :return: None
        """

        right_js = reorder_values(joint_states[17:24])
        left_js = reorder_values(joint_states[31:38])

        self._robot.q[16:23] = right_js
        self._robot.q[23:30] = left_js
        self.q = deepcopy(self._robot.q)

        if self._launch_visualizer:

            self._left_ax.T = self._robot.fkine(
                self._robot.q, end=self._arms_frame['left']['end'], tool=self._tool_offset['left']).A
            self._right_ax.T = self._robot.fkine(
                self._robot.q, end=self._robot.grippers['right']['end'], tool=self._tool_offset['right']).A

    def init_visualization(self):
        r"""
        Initialize the visualization of the robot

        :return: None
        """

        self._left_ax = geometry.Axes(length=0.05, pose=self._robot.fkine(
            self._robot.q, end=self._arms_frame['left']['end'], tool=self._tool_offset['left']).A)

        self._right_ax = geometry.Axes(length=0.05, pose=self._robot.fkine(
            self._robot.q, end=self._arms_frame['right']['end'], tool=self._tool_offset['right']).A)

        # self._env.add(self._robot)
        # self._env.add(self._left_ax)
        # self._env.add(self._right_ax)

    def get_tool_pose(self, side):
        r"""
        Get the tool pose of the robot
        :param side: side of the robot

        :return: tool pose
        """

        return self._robot.fkine(self._robot.q, end=self._arms_frame[side]['end'], tool=self._tool_offset[side]).A

    def get_jacobian(self, side):
        r"""
        Get the Jacobian of the robot on the tool frame
        :param side: side of the robot

        :return: Jacobian
        """

        return self._robot.jacobe(self._robot.q, end=self._arms_frame[side]['end'], start=self._arms_frame[side]['start'], tool=self._tool_offset[side])

    def shutdown(self):
        r"""
        Get the joint states of the robot
        :return: joint states
        """
        print("Fake PR2 is collapsed")
        if self._launch_visualizer:
            self._is_collapsed = True
            self._thread.join()
