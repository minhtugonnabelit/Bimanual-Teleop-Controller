import numpy as np
import spatialmath as sm
import spatialmath.base as smb
import spatialgeometry as geometry
import roboticstoolbox as rtb
from scipy import linalg, optimize
from swift import Swift
import threading

from copy import deepcopy
import math
import pygame
import sys
import matplotlib.pyplot as plt

TWIST_GAIN = [0.2, 0.2]
CONTROL_RATE = 20
SAMPLE_STATES = {
    'left': [np.pi/4, np.pi/6, np.pi/2, -np.pi/2, np.pi/6, -np.pi/4, np.pi/2],
    'right': [-np.pi/4, np.pi/6, -np.pi/2, -np.pi/2, -np.pi/6, -np.pi/4, np.pi/2]
}
NEUTRAL_STATES = {
    'left': [0.05592020315366142, 0.4115547023030020313, 1.223241480964399752, -0.75718229886988179, 0.25000010026008326, -0.48229593735634957, 1.573265592638103776],
    'right': [-0.05869937106810763, 0.4107752715756987882, -1.223126457438489645, -0.75897762731364821, -0.25000005892831325, -0.4851061342000067, -1.5713531640700703562,]
}
JOINT_NAMES = {
    "left": [
        "l_shoulder_pan_joint",
        "l_shoulder_lift_joint",
        "l_upper_arm_roll_joint",
        "l_elbow_flex_joint",
        "l_forearm_roll_joint",
        "l_wrist_flex_joint",
        "l_wrist_roll_joint",
    ],
    "right": [
        "r_shoulder_pan_joint",
        "r_shoulder_lift_joint",
        "r_upper_arm_roll_joint",
        "r_elbow_flex_joint",
        "r_forearm_roll_joint",
        "r_wrist_flex_joint",
        "r_wrist_roll_joint",
    ]
}


class CalcFuncs():

    def __init__(self):
        pass

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

    def rmrc(jacob, twist, w_thresh=0.08):
        r"""
        Calculate the joint velocities using the Resolved Motion Rate Control (RMRC) method.

        Parameters:
        - jacob: The Jacobian matrix of the robot.
        - twist: The desired twist of the robot.

        Returns:
        - The joint velocities of the robot.

        """

        # calculate manipulability
        w = CalcFuncs.manipulability(jacob)

        # set threshold and damping
        max_damp = 0.5

        # if manipulability is less than threshold, add damping
        damp = (1 - np.power(w/w_thresh, 2)) * max_damp if w < w_thresh else 0

        # calculate damped least square
        j_dls = np.transpose(jacob) @ np.linalg.inv(jacob @
                                                    np.transpose(jacob) + np.power(damp,1) * np.eye(6))

        # get joint velocities, if robot is in singularity, use damped least square
        qdot = j_dls @ np.transpose(twist)

        return qdot

    def nullspace_projector(m):
        r"""
        Calculate the projection matrix on to the null space of the Jacobian matrix.

        Parameters:
        - m: The matrix to construct the projector on to its nullspace.

        Returns:
        - The projector matrix for a joint vector onto N(m).
        """

        return np.eye(m.shape[1]) - np.linalg.pinv(m) @ m

    def duo_arm_qdot_constraint(jacob_l, jacob_r, twist, activate_nullspace=True):
        r"""
        Calculate the joint velocities using the Resolved Motion Rate Control (RMRC) method for a dual-arm robot.
        The Jacobian of the left and right arms are used to calculate the joint velocities that need to be presented on the same frame
        """

        qdot_left = CalcFuncs.rmrc(jacob_l, twist)
        qdot_right = CalcFuncs.rmrc(jacob_r, twist)

        # Combine the joint velocities of the left and right arms
        qdotc = np.r_[qdot_left, qdot_right]

        # Combine the Jacobians of the left and right arms
        jacob_c = np.c_[jacob_l, -jacob_r]

        if activate_nullspace:
            nullspace_projection_matrix = np.eye(
                jacob_c.shape[1]) - np.linalg.pinv(jacob_c) @ jacob_c
            qdotc = nullspace_projection_matrix @ qdotc

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
        li = np.array([R[2, 1] - R[1, 2], R[0, 2] -
                      R[2, 0], R[1, 0] - R[0, 1]])
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


class AnimateFuncs():


    @classmethod
    def add_frame_to_plot(cls, ax, tf, label=''):
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

    @classmethod
    def animate_frame(cls, tf, quivers, text, ax):

        if quivers:
            for q in quivers:
                q.remove()
        if text:
            text.remove()

        return AnimateFuncs.add_frame_to_plot(ax, tf)


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
            print("Button 0 is pressed")
            done = True

        vz = (lpf(joy.get_axis(2) + 1,))/2 - (lpf(joy.get_axis(5) + 1,))/2
        y = joy.get_button(1)*0.1 - joy.get_button(3)*0.1

        # Low pass filter
        vy = lpf(joy.get_axis(1))  # if abs(joy.get_axis(0)) > 0.1 else 0
        vx = lpf(joy.get_axis(1))  # if abs(joy.get_axis(1)) > 0.1 else 0
        r = lpf(joy.get_axis(3))  # if abs(joy.get_axis(3)) > 0.2 else 0
        p = lpf(joy.get_axis(4))  # if abs(joy.get_axis(4)) > 0.2 else 0

    else:

        if joy[1][-3]:
            done = True

        vz = ((lpf(joy[0][5] + 1)) - (lpf(joy[0][2] + 1)))/2
        y = 0

        # Low pass filter
        vy = lpf(joy[0][0])
        vx = lpf(joy[0][1])
        r = lpf(joy[0][3])
        p = lpf(joy[0][4])

    # ---------------------------------------------------------------------------#
    twist = np.zeros(6)
    twist[:3] = np.array([vx, vy, vz]) * gain[0]
    twist[3:] = np.array([r, p, y]) * gain[1]

    return twist, done


def lpf(value, threshold=0.2):
    r"""
    Low pass filter for the joystick data.
    """

    return value if abs(value) > threshold else 0


def map_interval(x):
    return 0.03 * (x + 1)


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


def plot_joint_velocities(actual_data: np.ndarray, desired_data: np.ndarray, dt=0.001, title='Joint Velocities'):

    actual_data = np.array(actual_data)
    desired_data = np.array(desired_data)
    # distance_data = np.array(distance_data)

    # Adjusted figsize for better visibility
    fig, ax = plt.subplots(2, 4, figsize=(18, 10))

    # Prepare data
    time_space = np.linspace(0, len(actual_data) * dt, len(actual_data))

    # Plot settings
    colors = ['r', 'b']  # Red for actual, Blue for desired
    labels = ['Actual', 'Desired']
    data_types = [actual_data, desired_data]

    for i in range(7):  # Assuming there are 7 joints
        joint_axes = ax[i // 4, i % 4]
        for data, color, label in zip(data_types, colors, labels):
            joint_data = data[:, i]
            if joint_data.shape[0] != len(time_space):
                time_space = np.linspace(
                    0, len(joint_data) * dt, len(joint_data))
            joint_axes.plot(time_space, joint_data, color,
                            linewidth=1, label=label if i == 0 else "")

            # Max and Min annotations
            max_value = np.max(joint_data)
            min_value = np.min(joint_data)
            max_time = time_space[np.argmax(joint_data)]
            min_time = time_space[np.argmin(joint_data)]

            joint_axes.annotate(f'Max: {max_value:.2f}', xy=(max_time, max_value), xytext=(10, 0),
                                textcoords='offset points', ha='center', va='bottom', color=color)
            joint_axes.annotate(f'Min: {min_value:.2f}', xy=(min_time, min_value), xytext=(10, -10),
                                textcoords='offset points', ha='center', va='top', color=color)

        joint_axes.set_title(JOINT_NAMES[title][i])

    fig.legend(['Actual', 'Desired'], loc='upper right')
    fig.suptitle(title)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    return fig, ax


def plot_manip_and_drift(constraint_distance: float, manipulabity_threshold: float, drift: np.ndarray, manip_l: np.ndarray, manip_r: list, dt=0.001):
    r"""
    Plot the manipulability and drift data for the PR2 robot.

    Parameters:
    - constraint_distance: The constraint distance data.
    - drift: The drift data.
    - manip_l: The manipulability data for the left arm.
    - manip_r: The manipulability data for the right arm.
    - dt: The time interval.

    Returns:
    - The figure and axes objects.
    """


    # Prepare data
    fig, ax = plt.subplots(2, 2, figsize=(18, 10))
    time_space = np.linspace(0, len(drift) * dt, len(drift))
    manip_axes = ax[0, 0]
    drift_axes = ax[0, 1]

    # Plot manipulability data    # Plot drift data
    if len(manip_r) != len(time_space):
        time_space = np.linspace(0, len(manip_r)
                                 * dt, len(manip_r) + 1)

    if len(manip_l) != len(time_space):
        time_space = np.linspace(0, len(manip_l)
                                 * dt, len(manip_l) + 1)
        
    manip_axes.plot(time_space, manip_l, 'r', linewidth=1)
    manip_axes.plot(time_space, manip_r, 'b', linewidth=1)
    manip_axes.set_title('Manipulability graph')
    manip_axes.set_xlabel('Time')
    manip_axes.set_ylabel('Manipulability')
    manip_axes.legend(['Left arm', 'Right arm'])
    manip_axes.axhline(y=manipulabity_threshold, color='k',
                       linewidth=1, linestyle='--')
    
    manip_axes.annotate(f'Min Left {np.min(manip_l):.4f}', 
                        xy=(time_space[np.argmin(manip_l)], np.min(manip_l)), 
                        xytext=(10, 0), textcoords='offset points', 
                        ha='center', va='bottom', color='r')
    
    manip_axes.annotate(f'Min Right {np.min(manip_r):.4f}', 
                        xy=(time_space[np.argmin(manip_r)], np.min(manip_r)), 
                        xytext=(10, -10), textcoords='offset points', 
                        ha='center', va='top', color='b')

    # Plot drift data
    if len(drift) != len(time_space):
        time_space = np.linspace(0, len(drift)
                                 * dt, len(drift) + 1)
        
    drift_axes.plot(time_space, drift, 'k', linewidth=1)
    drift_axes.set_title('Drift graph')
    drift_axes.set_xlabel('Time')
    drift_axes.set_ylabel('Distance')
    drift_axes.set_ylim([constraint_distance - 0.2, constraint_distance + 0.2])
    drift_axes.axhline(y=constraint_distance, color='r', linewidth=1)

    drift_axes.annotate(f'Constraint {constraint_distance:.4f}',
                        xy=(time_space[time_space.size//2], 0.35),
                        xytext=(10, 0),
                        textcoords='offset points',
                        ha='center', va='bottom', color='r')

    drift_axes.annotate(f'Max {np.max(drift):.4f}',
                        xy=(time_space[np.argmax(drift)], np.max(drift)),
                        xytext=(10, 0), textcoords='offset points',
                        ha='center', va='bottom', color='k')

    drift_axes.annotate(f'Min {np.min(drift):.4f}',
                        xy=(time_space[np.argmin(drift)], np.min(drift)),
                        xytext=(10, -10), textcoords='offset points',
                        ha='center', va='top', color='k')

    return fig, ax

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
        return True
    
    