import numpy as np
import spatialmath as sm
import spatialmath.base as smb
import spatialgeometry as geometry
import roboticstoolbox as rtb
from scipy import linalg
from swift import Swift

import threading
from copy import deepcopy
import math
import pygame
import sys
# import time
import matplotlib.pyplot as plt
from enum import Enum


def joy_init():
    r"""
    Initialize the pygame library and joystick.

    Returns:
    - The joystick object.

    """

    pygame.init()
    joystick_count = pygame.joystick.get_count()
    if joystick_count == 0:
        raise Exception('No joystick found')
    else:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()

    return joystick


def joy_to_twist(joy, gain):
    r"""

    Convert the joystick data to a twist message using Pygame module.

    Parameters:
    - joy: The joystick object.
    - gain: The gain of the linear and angular velocities.

    Returns:
    - The twist message.

    """
    vx, vy, vz, r, p, y = 0, 0, 0, 0, 0, 0
    done = False

    if isinstance(joy, pygame.joystick.JoystickType):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        if joy.get_button(0):
            done = True

        vz = (joy.get_axis(2) + 1)/2 - (joy.get_axis(5) + 1)/2
        y = joy.get_button(1)*0.1 - joy.get_button(3)*0.1

        # Low pass filter
        vy = joy.get_axis(0) if abs(joy.get_axis(0)) > 0.1 else 0
        vx = joy.get_axis(1) if abs(joy.get_axis(1)) > 0.1 else 0
        r = joy.get_axis(3) if abs(joy.get_axis(3)) > 0.2 else 0
        p = joy.get_axis(4) if abs(joy.get_axis(4)) > 0.2 else 0

    else:

        if joy[1][-3]:
            done = True

        vz = (joy[0][5] + 1)/2 - (joy[0][2] + 1)/2
        y = joy[1][4]*0.1 - joy[1][5]*0.1

        # Low pass filter
        vy = joy[0][0] if abs(joy[0][0]) > 0.1 else 0
        vx = joy[0][1] if abs(joy[0][1]) > 0.1 else 0
        r = joy[0][3] if abs(joy[0][3]) > 0.2 else 0
        p = joy[0][4] if abs(joy[0][4]) > 0.2 else 0

    # ---------------------------------------------------------------------------#
    twist = np.zeros(6)
    twist[:3] = np.array([vx, vy, vz]) * gain[0]
    twist[3:] = np.array([r, p, y]) * gain[1]

    return twist, done


def adjoint(T):

    R = T[:3, :3]
    p = T[:3, -1]

    ad = np.eye(6, 6)
    ad[:3, :3] = R
    ad[3:, 3:] = R
    ad[:3, 3:] = np.cross(p, R)

    return ad


def manipulability(jacob):
    r"""
    Calculate the manipulability of the robot.

    Parameters:
    - jacob: The Jacobian matrix of the robot.

    Returns:
    - The manipulability of the robot.

    """

    return np.sqrt(np.linalg.det(jacob @ np.transpose(jacob)))


def rmrc(jacob, twist):
    r"""
    Calculate the joint velocities using the Resolved Motion Rate Control (RMRC) method.

    Parameters:
    - jacob: The Jacobian matrix of the robot.
    - twist: The desired twist of the robot.

    Returns:
    - The joint velocities of the robot.

    """

    # calculate manipulability
    w = np.sqrt(np.linalg.det(jacob @ np.transpose(jacob)))

    # set threshold and damping
    w_thresh = 0.08
    max_damp = 0.5

    # if manipulability is less than threshold, add damping
    damp = (1 - np.power(w/w_thresh, 2)) * max_damp if w < w_thresh else 0

    # calculate damped least square
    j_dls = np.transpose(jacob) @ np.linalg.inv(jacob @
                                                np.transpose(jacob) + damp * np.eye(6))

    # get joint velocities, if robot is in singularity, use damped least square
    qdot = j_dls @ np.transpose(twist)

    return qdot


def nullspace_projection(jacob):
    r"""
    Calculate the projection matrix on to the null space of the Jacobian matrix.

    Parameters:
    - jacob: The Jacobian matrix of the robot.

    Returns:
    - The projection matrix on to the null space of the Jacobian matrix.
    """

    return np.eye(jacob.shape[1]) - np.linalg.pinv(jacob) @ jacob


def duo_arm_qdot_constraint(jacob_l, jacob_r, twist, activate_nullspace=True):
    r"""
    Calculate the joint velocities using the Resolved Motion Rate Control (RMRC) method for a dual-arm robot.
    The Jacobian of the left and right arms are used to calculate the joint velocities that need to be presented on the same frame
    """

    qdot_left = rmrc(jacob_l, twist, )
    qdot_right = rmrc(jacob_r, twist, )
    # Combine the joint velocities of the left and right arms
    qdotc = np.concatenate([qdot_left, qdot_right], axis=0)
    # Combine the Jacobians of the left and right arms
    jacob_c = np.concatenate([jacob_l, -jacob_r], axis=1)

    if activate_nullspace:
        qdotc = nullspace_projection(jacob_c) @ qdotc

    qdot_l = qdotc[0:jacob_l.shape[1]]
    qdot_r = qdotc[jacob_l.shape[1]:]

    return qdot_l, qdot_r


def angle_axis_python(T, Td):
    r"""
    Computes the angle-axis representation of the error between two transformation matrices.

    Parameters:
    - T: The current transformation matrix.
    - Td: The desired transformation matrix.
    Returns:
    - e: A 6x1 vector representing the error in the angle-axis representation.
    - angle: The angle of rotation between the two frames.
    - axis: The axis of rotation between the two frames.

    """
    e = np.empty(6)
    e[:3] = Td[:3, -1] - T[:3, -1]
    R = Td[:3, :3] @ T[:3, :3].T
    li = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    ln = smb.norm(li)

    if smb.iszerovec(li):
        # diagonal matrix case
        if np.trace(R) > 0:
            # (1,1,1) case
            a = np.zeros((3,))
        else:
            a = np.pi / 2 * (np.diag(R) + 1)
    else:
        # non-diagonal matrix case
        a = math.atan2(ln, np.trace(R) - 1) * li / ln

    axis = li / ln
    angle = math.atan2(ln, np.trace(R) - 1)

    e[3:] = a

    return e, angle, axis


# Function for visualiztion of the frames in the 3D plot using Matplotlib
def add_frame_to_plot(ax, tf, label=''):
    r"""
    Adds a transformation frame to an existing 3D plot and returns the plotted objects.

    Parameters:
    - ax: A matplotlib 3D axis object where the frame will be added.
    - tf: 4x4 transformation matrix representing the frame's position and orientation.
    - label: A string label to identify the frame.

    Returns:
    - A list of matplotlib objects (quivers and text).
    """
    # Origin of the frame
    origin = np.array(tf[0:3, 3]).reshape(3, 1)

    # Directions of the frame's axes, transformed by R
    x_dir = tf[0:3, 0:3] @ np.array([[1], [0], [0]])
    y_dir = tf[0:3, 0:3] @ np.array([[0], [1], [0]])
    z_dir = tf[0:3, 0:3] @ np.array([[0], [0], [1]])

    # Plotting each axis using quiver for direction visualization
    quivers = []
    quivers.append(ax.quiver(*origin, *x_dir, color='r',
                   length=0.15, linewidth=1, normalize=True))
    quivers.append(ax.quiver(*origin, *y_dir, color='g',
                   length=0.15, linewidth=1, normalize=True))
    quivers.append(ax.quiver(*origin, *z_dir, color='b',
                   length=0.15, linewidth=1, normalize=True))

    # Optionally adding a label to the frame
    text = None
    if label:
        text = ax.text(*origin.flatten(), label, fontsize=12)

    return quivers, text


def animate_frame(tf, quivers, text, ax):

    if quivers:
        for q in quivers:
            q.remove()
    if text:
        text.remove()

    return add_frame_to_plot(ax, tf)


def reorder_values(data):
    r"""
    Reorder the joints based on the joint names to use with PR2 feedback data

    Parameters:
    - joints: The joint values.
    - joint_names: The joint names.

    Returns:
    - The reordered joints.
    """

    if len(data) != 7:
        raise ValueError("The length of the data should be 7")

    data_array = np.asarray(data)
    data_array[0], data_array[1], data_array[2], data_array[3], data_array[
        4] = data_array[1], data_array[2], data_array[0], data_array[4], data_array[3]

    return data_array.tolist()


def plot_joint_velocities(actual_data: np.ndarray, desired_data, distance_data):

    fig, ax = plt.subplots(2, 4)

    reformat_actual_data = list()
    reformat_desired_data = list()

    for i in range(7):
        joint_vel_actual_data = list()
        joint_vel_desired_data = list()
        for j in range(len(actual_data)):
            joint_vel_actual_data.append(actual_data[j][i])
            joint_vel_desired_data.append(desired_data[j][i])
        reformat_actual_data.append(joint_vel_actual_data)
        reformat_desired_data.append(joint_vel_desired_data)

    print(len(reformat_actual_data[0]))
    print(len(reformat_desired_data[0]))

    ax[0, 0].plot(reformat_actual_data[0], 'r', linewidth=1)
    ax[0, 0].plot(reformat_desired_data[0], 'b', linewidth=1)
    ax[0, 0].set_title(f"Joint {0}")

    ax[0, 1].plot(reformat_actual_data[1], 'r', linewidth=1)
    ax[0, 1].plot(reformat_desired_data[1], 'b', linewidth=1)
    ax[0, 1].set_title(f"Joint {1}")

    ax[0, 2].plot(reformat_actual_data[2], 'r', linewidth=1)
    ax[0, 2].plot(reformat_desired_data[2], 'b', linewidth=1)
    ax[0, 2].set_title(f"Joint {2}")

    ax[0, 3].plot(reformat_actual_data[3], 'r', linewidth=1)
    ax[0, 3].plot(reformat_desired_data[3], 'b', linewidth=1)
    ax[0, 3].set_title(f"Joint {3}")

    ax[1, 0].plot(reformat_actual_data[4], 'r', linewidth=1)
    ax[1, 0].plot(reformat_desired_data[4], 'b', linewidth=1)
    ax[1, 0].set_title(f"Joint {4}")

    ax[1, 1].plot(reformat_actual_data[5], 'r', linewidth=1)
    ax[1, 1].plot(reformat_desired_data[5], 'b', linewidth=1)
    ax[1, 1].set_title(f"Joint {5}")

    ax[1, 2].plot(reformat_actual_data[6], 'r', linewidth=1)
    ax[1, 2].plot(reformat_desired_data[6], 'b', linewidth=1)
    ax[1, 2].set_title(f"Joint {6}")

    ax[1, 3].plot(distance_data, 'g', linewidth=1)

    fig.legend(['Actual', 'Desired'])

    plt.show()


class FakePR2:

    r"""
    ### Class to simulate PR2 on Swift environment

    This will initialize the Swift environment with robot model without any ROS components. 
    This model will be fed with joint states provided and update the visualization of the virtual frame . """

    def __init__(self, launch_visualizer) -> None:

        print('bruh1')
        self._robot = rtb.models.PR2()
        self._is_collapsed = False
        self._constraints_is_set = False
        self._virtual_pose = None
        self._launch_visualizer = launch_visualizer
        if self._launch_visualizer:
            self.init_visualization()
            self.thread = threading.Thread(target=self.timeline)
            self.thread.start()

    def timeline(self):
        r"""
        Timeline function to update the visualization
        :return: None
        """
        self._env.launch()
        while not self._is_collapsed:
            self._env.step(0.1)

    def set_constraints(self, virtual_pose: np.ndarray):
        r"""
        Set the kinematics constraints for the robot
        :param virtual_pose: virtual pose on robot base frame
        :return: None
        """

        self._virtual_pose = virtual_pose
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
        right_js = reorder_values(joint_states[17:24])
        left_js = reorder_values(joint_states[31:38])

        self._robot.q[16:23] = right_js
        self._robot.q[23:30] = left_js
        self.q = deepcopy(self._robot.q)

        if self._launch_visualizer:
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
        print('bruh')

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

        print('brhu2')

        # self._env.add(self._robot)
        # self._env.add(self._left_ax)
        # self._env.add(self._right_ax)

    def get_virtual_pose(self):
        r"""
        Get the virtual pose of the robot
        :return: virtual pose
        """
        return self._virtual_pose

    def get_tool_pose(self, side):
        r"""
        Get the tool pose of the robot
        :param side: side of the robot

        :return: tool pose
        """
        if side == 'left':
            return self._robot.fkine(self._robot.q, end=self._robot.grippers[1], ).A @ self._ee_constraint[side] if self._constraints_is_set else self._robot.fkine(self._robot.q, end=self._robot.grippers[1], ).A
        elif side == 'right':
            return self._robot.fkine(self._robot.q, end=self._robot.grippers[0], ).A @ self._ee_constraint[side] if self._constraints_is_set else self._robot.fkine(self._robot.q, end=self._robot.grippers[0], ).A
        else:
            return None

    def get_jacobian(self, side):
        r"""
        Get the Jacobian of the robot on the tool frame
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
        if self._launch_visualizer:
            self._is_collapsed = True
            self.thread.join()
