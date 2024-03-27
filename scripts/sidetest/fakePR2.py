import numpy as np
import spatialmath as sm
import spatialgeometry as geometry
import roboticstoolbox as rtb
from scipy import linalg
from swift import Swift
import threading


class FakePR2:

    r"""
    ### Class to simulate PR2 on Swift environment

    This will initialize the Swift environment with robot model without any ROS components. 
    This model will be fed with joint states provided and update the visualization of the virtual frame . """

    def __init__(self) -> None:

        self._robot = rtb.models.PR2()
        self._robot.q = np.zeros(31)

        self._is_collapsed = False
        self._constraints_is_set = False
        self.init_visualization()
        self.thread = threading.Thread(target=self.timeline)
        self.thread.start()

        
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

        # self._virtual_pose = virtual_pose
        self._joined_in_left = linalg.inv(self._robot.fkine(
            self._robot.q, end=self._robot.grippers[1])) @ virtual_pose
        self._joined_in_right = linalg.inv(self._robot.fkine(
            self._robot.q, end=self._robot.grippers[0])) @ virtual_pose
        
        self._ee_constraint = {
            "left": self._joined_in_left,
            "right": self._joined_in_right
        }

        self._constraints_is_set = True

        return True

    def set_joint_states(self, joint_states: list):
        r"""
        Set the joint states of the arms only
        :param joint_states: list of joint states
        :return: None
        """
        # print(joint_states)
        left_js = np.array(joint_states[17:24])
        right_js = np.array(joint_states[31:38])

        left_js[0], left_js[1], left_js[2], left_js[3], left_js[4] = left_js[1], left_js[2], left_js[0], left_js[4], left_js[3]
        right_js[0], right_js[1], right_js[2], right_js[3], right_js[4] = right_js[1], right_js[2], right_js[0], right_js[4], right_js[3]

        self._robot.q[16:23] = left_js
        self._robot.q[23:30] = right_js

        left_constraint = np.eye(4)
        right_constraint = np.eye(4)
        if self._constraints_is_set:    # If the constraints are set, then update the virtual frame from the middle point between the two end-effectors
            left_constraint = self._ee_constraint['left']
            right_constraint = self._ee_constraint['right']

        self._left_ax.T = self._robot.fkine(
            self._robot.q, end=self._robot.grippers[1], ).A @ left_constraint
        self._right_ax.T = self._robot.fkine(
            self._robot.q, end=self._robot.grippers[0], ).A @ right_constraint

    def init_visualization(self):
        r"""
        Initialize the visualization of the robot

        :return: None
        """

        self._env = Swift()
        self._env.set_camera_pose([1, 0, 1], [0, 0.5, 1])
        self._env.launch()

        if not self._constraints_is_set:    # If the constraints are not set, then visualize the virtual frame from each arm end-effector
            self._left_ax = geometry.Axes(length=0.05, pose=self._robot.fkine(
                self._robot.q, end=self._robot.grippers[1], ).A)
            self._right_ax = geometry.Axes(length=0.05, pose=self._robot.fkine(
                self._robot.q, end=self._robot.grippers[0], ).A)
        else:                           # If the constraints are set, then visualize the virtual frame from the middle point between the two end-effectors
            self._left_ax = geometry.Axes(length=0.05, pose=self._robot.fkine(
                self._robot.q, end=self._robot.grippers[1], ).A @ self._joined_in_left)
            self._right_ax = geometry.Axes(length=0.05, pose=self._robot.fkine(
                self._robot.q, end=self._robot.grippers[0], ).A @ self._joined_in_right)

        self._env.add(self._robot)
        self._env.add(self._left_ax)
        self._env.add(self._right_ax)



    def get_jacobian(self, side):
        r"""
        Get the Jacobian of the robot
        :param side: side of the robot

        :return: Jacobian
        """
        tool = sm.SE3(self._ee_constraint[side]) if self._constraints_is_set else None

        if side == 'left':
            return self._robot.jacobe(self._robot.q, end=self._robot.grippers[1], start="l_shoulder_pan_link", tool=tool)
        elif side == 'right':
            return self._robot.jacobe(self._robot.q, end=self._robot.grippers[0], start="r_shoulder_pan_link", tool=tool)
        else:
            return None

    def shutdown(self):
        r"""
        Get the joint states of the robot
        :return: joint states
        """
        print("Fake PR2 is collapsed")
        self._is_collapsed = True
        self.thread.join()