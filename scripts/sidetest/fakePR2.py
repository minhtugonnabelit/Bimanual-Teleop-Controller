import numpy as np
import spatialmath as sm
import spatialgeometry as geometry
import roboticstoolbox as rtb
from swift import Swift
from scipy import linalg
from copy import deepcopy
import threading

from bimanual_teleop_controller.utility import *


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
            'l': np.eye(4),
            'r': np.eye(4)
        }

        self._arms_frame = {
            'l': {
                'end': self._robot.grippers[1],
                'start': 'l_shoulder_pan_link'
            },
            'r': {
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
            self._env.step(1/CONTROL_RATE)

    def set_constraints(self, virtual_pose: np.ndarray):
        r"""
        Set the kinematics constraints for the robot
        :param virtual_pose: virtual pose on robot base frame
        :return: None
        """

        self._tool_offset['l'] = linalg.inv(self._robot.fkine(
            self._robot.q, end=self._arms_frame['l']['end'])) @ virtual_pose
        self._tool_offset['r'] = linalg.inv(self._robot.fkine(
            self._robot.q, end=self._arms_frame['r']['end'])) @ virtual_pose

        return True

    def set_states(self, joint_states: list, real_robot=True):
        r"""
        Set the joint states of the arms only
        :param joint_states: list of joint states
        :return: None
        """
        if real_robot:
            right_js = reorder_values(joint_states[17:24])
            left_js = reorder_values(joint_states[31:38])
        else:
            right_js = joint_states[0:7]
            left_js = joint_states[7:14]

        self._robot.q[16:23] = right_js
        self._robot.q[23:30] = left_js
        self.q = deepcopy(self._robot.q)

        if self._launch_visualizer:

            self._left_ax.T = self._robot.fkine(
                self._robot.q, end=self._arms_frame['l']['end'], tool=self._tool_offset['l']).A
            self._right_ax.T = self._robot.fkine(
                self._robot.q, end=self._arms_frame['r']['end'], tool=self._tool_offset['r']).A

    def init_visualization(self):
        r"""
        Initialize the visualization of the robot

        :return: None
        """

        self._left_ax = geometry.Axes(length=0.05, pose=self._robot.fkine(
            self._robot.q, end=self._arms_frame['l']['end'], tool=self._tool_offset['l']).A)

        self._right_ax = geometry.Axes(length=0.05, pose=self._robot.fkine(
            self._robot.q, end=self._arms_frame['r']['end'], tool=self._tool_offset['r']).A)

        self._env.add(self._robot)
        self._env.add(self._left_ax)
        self._env.add(self._right_ax)

    def get_tool_pose(self, side:str, offset=True):
        r"""
        Get the tool pose of the robot
        :param side: side of the robot
        :param offset: include tool offset or not

        :return: tool pose
        """
        tool = self._tool_offset[side] if offset else np.eye(4)

        return self._robot.fkine(self._robot.q, end=self._arms_frame[side]['end'], tool=tool).A
    
    def get_joint_positions(self, side : str):
        r"""
        Get the joint states of the robot
        :return: joint states
        """
        if side == 'l':
            return self._robot.q[23:30]
        else:
            return self._robot.q[16:23]

    def get_jacobian(self, side):
        r"""
        Get the Jacobian of the robot on the tool frame
        :param side: side of the robot

        :return: Jacobian
        """

        return self._robot.jacobe(self._robot.q, end=self._arms_frame[side]['end'], start=self._arms_frame[side]['start'], tool=self._tool_offset[side])

    def get_drift_compensation(self) -> np.ndarray:
        r"""
        Get the drift compensation for the robot
        :return: drift compensation velocities as joint velocities
        """
        v, _ = rtb.p_servo(self.get_tool_pose(side = 'l', offset=True), 
                        self.get_tool_pose(side = 'r', offset=True),
                        1, 0.01,
                        method='angle-axis')
        
        # get fix velocities for the drift for both linear and angular velocities
        qdot_fix_left = CalcFuncs.rmrc(self.get_jacobian('l'), v, w_thresh=0.05)
        qdot_fix_right = CalcFuncs.rmrc(self.get_jacobian('r'), -v, w_thresh=0.05)

        return np.r_[qdot_fix_left, qdot_fix_right]

    def shutdown(self):
        r"""
        Get the joint states of the robot
        :return: joint states
        """
        if self._launch_visualizer:
            self._is_collapsed = True
            self._thread.join()
