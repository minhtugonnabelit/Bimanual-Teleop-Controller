from typing import Union

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
import time
import matplotlib.pyplot as plt

import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

ArrayLike = Union[list, np.ndarray, tuple, set]

SAMPLE_STATES = {
    'r': np.deg2rad([-40, 30, -90, -91, -31, -39, 91]),
    'l': np.deg2rad([39, 27, 91, -92, 31, -38, 85])}

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


class CalcFuncs:
    r"""
    Class to calculate the necessary functions for the robot control.
    """
    
    @staticmethod    
    def adjoint(T):

        R = T[:3, :3]
        p = T[:3, -1]

        ad = np.eye(6, 6)
        ad[:3, :3] = R
        ad[3:, 3:] = R
        ad[3:, :3] = np.cross(p, R)

        return ad

    @staticmethod    
    def manipulability(jacob):
        r"""
        Calculate the manipulability of the robot.

        Parameters:
        - jacob: The Jacobian matrix of the robot.

        Returns:
        - The manipulability of the robot.

        """

        return np.sqrt(np.linalg.det(jacob @ np.transpose(jacob)))
    
    @staticmethod    
    def rmrc(jacob, twist, w_thresh=0.08):
        r"""
        Calculate the joint velocities using the Resolved Motion Rate Control (RMRC) method.

        Parameters:
        - jacob: The Jacobian matrix of the robot.
        - twist: The desired twist of the robot.

        Returns:
        - The joint velocities of the robot.

        """

        w = CalcFuncs.manipulability(jacob)

        max_damp = 1
        # damp = (1 - np.power(w/w_thresh, 2)) * max_damp if w < w_thresh else 0
        damp = np.power(1 - w/w_thresh, 2) * max_damp if w < w_thresh else 0

        j_dls = np.transpose(jacob) @ np.linalg.inv(jacob @
                                                    np.transpose(jacob) + damp * np.eye(6))

        qdot = j_dls @ np.transpose(twist)
        return qdot
    
    @staticmethod
    def nullspace_projector(m):
        r"""
        Calculate the projection matrix on to the null space of the Jacobian matrix.

        Parameters:
        - m: The matrix to construct the projector on to its nullspace.

        Returns:
        - The projector matrix for a joint vector onto N(m).
        """

        return np.eye(m.shape[1]) - np.linalg.pinv(m) @ m
    
    @staticmethod
    def duo_arm_qdot_constraint(jacob_l, jacob_r, twist, activate_nullspace=True):
        r"""
        Calculate the joint velocities using the Resolved Motion Rate Control (RMRC) method for a dual-arm robot.
        The Jacobian of the left and right arms are used to calculate the joint velocities that need to be presented on the same frame
        """

        qdot_left = CalcFuncs.rmrc(jacob_l, twist)
        qdot_right = CalcFuncs.rmrc(jacob_r, twist)

        qdotc = np.r_[qdot_left, qdot_right]
        jacob_c = np.c_[jacob_l, -jacob_r]

        if activate_nullspace:
            nullspace_projection_matrix = np.eye(
                jacob_c.shape[1]) - np.linalg.pinv(jacob_c) @ jacob_c
            qdotc = nullspace_projection_matrix @ qdotc

        qdot_l = qdotc[0:jacob_l.shape[1]]
        qdot_r = qdotc[jacob_l.shape[1]:]

        return qdot_l, qdot_r
    
    @staticmethod
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
                a = np.zeros((3,))
            else:
                a = np.pi / 2 * (np.diag(R) + 1)
        else:
            # non-diagonal matrix case
            a = math.atan2(ln, np.trace(R) - 1) * li / ln

        axis = li / ln
        angle = math.atan2(ln, np.trace(R) - 1)

        e[3:] = a

        return e
    
    @staticmethod
    def pd_servo(
        wTe, 
        wTep, 
        prev_error: np.ndarray, 
        gain_p: Union[float, ArrayLike] = 1.0, 
        gain_d: Union[float, ArrayLike] = 0.0,
        threshold=0.1, 
        method="rpy",
        dt=0.001
    ):
        """
        Position-based servoing.

        Returns the end-effector velocity which will cause the robot to approach
        the desired pose.

        :param wTe: The current pose of the end-effecor in the base frame.
        :type wTe: SE3 or ndarray
        :param wTep: The desired pose of the end-effecor in the base frame.
        :type wTep: SE3 or ndarray
        :param cumulative_linear_error: The cumulative linear error of the robot.
        :type cumulative_linear_error: ndarray(3)
        :param cumulative_angular_error: The cumulative angular error of the robot.
        :type cumulative_angular_error: ndarray(3)
        :param gain_p: The proportional gain for the controller. Can be vector corresponding to each
            axis, or scalar corresponding to all axes.
        :type gain_p: float, or array-like
        :param gain_i: The integral gain for the controller. Can be vector corresponding to each
            axis, or scalar corresponding to all axes.
        :param threshold: The threshold or tolerance of the final error between
            the robot's pose and desired pose
        :type threshold: float
        :param method: The method used to calculate the error. Default is 'rpy' -
            error in the end-effector frame. 'angle-axis' - error in the base frame
            using angle-axis method.
        :type method: string: 'rpy' or 'angle-axis'

        :returns v: The velocity of the end-effecotr which will casue the robot
            to approach wTep
        :rtype v: ndarray(6)
        :returns arrived: True if the robot is within the threshold of the final
            pose
        :rtype arrived: bool

        """

        if isinstance(wTe, sm.SE3):
            wTe = wTe.A

        if isinstance(wTep, sm.SE3):
            wTep = wTep.A

        if method == "rpy":
            # Pose difference
            eTep = np.linalg.inv(wTe) @ wTep
            e = np.empty(6)

            # Translational error
            e[:3] = eTep[:3, -1]

            # Angular error
            e[3:] = smb.tr2rpy(eTep, unit="rad", order="zyx", check=False)
        else:
            e = CalcFuncs.angle_axis_python(wTe, wTep)

        # Calculate the derivative of the error
        d_error = (e - prev_error) / dt

        if smb.isscalar(gain_p):
            kp = gain_p * np.eye(6)
        else:
            kp = np.diag(gain_p)

        if smb.isscalar(gain_d):
            kd = gain_d * np.eye(6)
        else:
            kd = np.diag(gain_d)

        v = kp @ e

        vd = kd @ d_error

        v = v + vd

        arrived = True if np.sum(np.abs(e)) < threshold else False

        return v, arrived, e
    
    @staticmethod
    def weight(x, k):
        r"""
        Weighting function for smooth transition between two states.

        Parameters:
        - x: The input value.
        - k: The gain for steepness of the transition.

        Returns:
        - The weighted value.
        """

        return 1 / (1 + np.exp(-k * (x - 0.5)))
    
    @staticmethod
    def weight_vector(x, k):

        return np.array([CalcFuncs.weight(xi, k) for xi in x])
    
    @staticmethod
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
    
    @staticmethod
    def debounce(last_switch_time, debounce_interval=1):
        current_time = time.time()
        if current_time - last_switch_time >= debounce_interval:
            last_switch_time = current_time
            return True, last_switch_time
        else:
            return False, last_switch_time
    
    @staticmethod
    def lerp(start, end, t):
        return start + t * (end - start)


class AnimateFuncs:
    r"""
    
    """

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


class FakePR2:

    r"""
    ### Class to simulate PR2 on Swift environment

    This will initialize the Swift environment with robot model without any ROS components. 
    This model will be fed with joint states provided and update the visualization of the virtual frame . """

    def __init__(self, control_rate, launch_visualizer) -> None:

        self._control_rate = control_rate
        self._launch_visualizer = launch_visualizer
        self._robot = rtb.models.PR2()
        self._is_collapsed = False
        self._drift_error = np.zeros(6)
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


        # Get the joint limits and form the symmetric soft limits
        qmin, qmax = self.get_joint_limits_all()
        self.qmid = (qmin + qmax) / 2

        # Normal joint with soft limits start at 80% and end at 90% of the joint limits
        self.soft_limit_start = ((qmax - qmin)/2) * 0.75
        self.soft_limit_end = ((qmax - qmin)/2) * 0.9

        # Except for the shoulder lift joints of both arms
        self.soft_limit_start[1] = ((qmax[1] - qmin[1])/2) * 0.7
        self.soft_limit_start[8] = deepcopy(self.soft_limit_start[1])

        self.soft_limit_range = (self.soft_limit_end - self.soft_limit_start)

        if self._launch_visualizer:
            self._env = Swift()
            self._env.launch()
            self.init_visualization()
            self._thread = threading.Thread(target=self.timeline)
            self._thread.start()

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
    
    def timeline(self):
        r"""
        Timeline function to update the visualization
        :return: None
        """
        while not self._is_collapsed:
            self._env.step(1/self._control_rate)

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
            right_js = CalcFuncs.reorder_values(joint_states[17:24])
            left_js = CalcFuncs.reorder_values(joint_states[31:38])
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

    def get_tool_pose(self, side: str, offset=True):
        r"""
        Get the tool pose of the robot
        :param side: side of the robot
        :param offset: include tool offset or not

        :return: tool pose
        """
        tool = self._tool_offset[side] if offset else np.eye(4)

        return self._robot.fkine(self._robot.q, end=self._arms_frame[side]['end'], tool=tool).A

    def get_joint_positions(self, side: str):
        r"""
        Get the joint states of the robot
        :return: joint states
        """
        if side == 'l':
            return self._robot.q[23:30]
        else:
            return self._robot.q[16:23]
        
    def get_joint_limits(self, side):
        r"""
        Get the joint limits of the robot
        :param side: side of the robot
        :return: joint limits
        """
        return self._robot.qlim[0][16:23] if side == 'r' else self._robot.qlim[0][23:30], self._robot.qlim[1][16:23] if side == 'r' else self._robot.qlim[1][23:30]
        
    def get_joint_limits_all(self):
        r"""
        Get the joint limits of the robot
    
        First 7 joints are for the left arm and the next 7 joints are for the right arm
        :return: joint limits   
        """

        qmin_l, qmax_l = self.get_joint_limits('l')
        qmin_r, qmax_r = self.get_joint_limits('r')
        qmin = np.r_[qmin_l, qmin_r]
        qmax = np.r_[qmax_l, qmax_r]

        return qmin, qmax

    def get_jacobian(self, side):
        r"""
        Get the Jacobian of the robot on the tool frame
        :param side: side of the robot

        :return: Jacobian
        """
        return self._robot.jacobe(self._robot.q, end=self._arms_frame[side]['end'], start=self._arms_frame[side]['start'], tool=self._tool_offset[side])

    def get_twist_in_tool_frame(self, side, twist):
                        
        tool = self.get_tool_pose(side) 
        twist_converted= CalcFuncs.adjoint(np.linalg.inv(tool)) @ twist

        return twist_converted

    def joint_limits_damper(self, qdot, dt, steepness=10):
        r"""
        Repulsive potential field for joint limits for both arms
        :param qdot: joint velocities
        :param steepness: steepness of the transition
        :return: repulsive velocity potential field 
        """
        # Get the joint positions for next step
        q = np.r_[self.get_joint_positions(
            'l'), self.get_joint_positions('r')] + qdot * dt
        
        x = np.zeros(qdot.shape[0])
        for i in range(len(x)):
            qi, qm, qsls, qsle, qslr = q[i], self.qmid[i],  self.soft_limit_start[i], self.soft_limit_end[i], self.soft_limit_range[i]
            a = np.abs(qi - qm)
            x[i] = np.round((a-qsls) / qslr , 4)
            # if a < qsls:
            #     x[i] = 0
            # elif a < qsle and a > qsls:
            #     x[i] = np.round(np.abs(a-qsls) / qslr , 4)
            # else:
            #     x[i] = 1
            # print(x[i])

        weights = CalcFuncs.weight_vector(x, steepness)
        qdot_repulsive = - weights.max() * qdot
        print(weights.max(), np.where(weights == weights.max())[0])

        return qdot_repulsive, weights.max(), np.where(weights == weights.max())[0]
    
    def joint_limits_damper_side(self, side, qdot, dt, steepness=10):
        r"""
        Repulsive potential field for joint limits for both arms
        :param qdot: joint velocities
        :param steepness: steepness of the transition
        :return: repulsive velocity potential field 
        """
        # Get the joint positions for next step
        q = self.get_joint_positions(side) + qdot * dt
        arm_offset = 0 if side == 'l' else 7
        
        x = np.zeros(qdot.shape[0])
        for i in range(len(x)):
            qi, qm, qsls, qsle, qslr = q[i], self.qmid[i+arm_offset],  self.soft_limit_start[i+arm_offset], self.soft_limit_end[i+arm_offset], self.soft_limit_range[i+arm_offset]
            a = np.abs(qi - qm)
            if a < qsls:
                x[i] = 0
            elif a < qsle and a > qsls:
                x[i] = np.round(np.abs(a-qsls) / qslr , 4)
            else:
                x[i] = 1

        weights = CalcFuncs.weight_vector(x, steepness)
        qdot_repulsive = - weights.max() * qdot

        return qdot_repulsive, weights.max(), np.where(weights == weights.max())[0]
    
    # def joint_limits_damper_left(self, qdot, dt, steepness=10):
    #     r"""
    #     Repulsive potential field for joint limits for both arms
    #     :param qdot: joint velocities
    #     :param steepness: steepness of the transition
    #     :return: repulsive velocity potential field 
    #     """
    #     # Get the joint positions for next step
    #     q = self.get_joint_positions('l') + qdot * dt
        
    #     x = np.zeros(qdot.shape[0])
    #     for i in range(len(x)):
    #         qi, qm, qsls, qsle, qslr = q[i], self.qmid[i],  self.soft_limit_start[i], self.soft_limit_end[i], self.soft_limit_range[i]
    #         a = np.abs(qi - qm)
    #         if a < qsls:
    #             x[i] = 0
    #         elif a < qsle and a > qsls:
    #             x[i] = np.round(np.abs(a-qsls) / qslr , 4)
    #         else:
    #             x[i] = 1

    #     weights = CalcFuncs.weight_vector(x, steepness)
    #     qdot_repulsive = - weights.max() * qdot

    #     return qdot_repulsive, weights.max()
    
    # def joint_limits_damper_right(self, qdot, dt, steepness=10):

    #     # Get the joint positions for next step
    #     q = self.get_joint_positions('r') + qdot * dt

    #     x = np.zeros(qdot.shape[0])
    #     for i in range(len(x)):
    #         qi, qm, qsls, qsle, qslr = q[i], self.qmid[i+7],  self.soft_limit_start[i+7], self.soft_limit_end[i+7], self.soft_limit_range[i+7]
    #         a = np.abs(qi - qm)
    #         if a < qsls:
    #             x[i] = 0
    #         elif a < qsle and a > qsls:
    #             x[i] = np.round(np.abs(a-qsls) / qslr , 4)
    #         else:
    #             x[i] = 1

    #     weights = CalcFuncs.weight_vector(x, steepness)
    #     qdot_repulsive = - weights.max() * qdot

    #     return qdot_repulsive, weights.max()


    def task_drift_compensation(self,  gain_p = 5, gain_d = 0.5, on_taskspace=True) -> np.ndarray:
        r"""
        Get the drift compensation for the robot
        :return: drift compensation velocities as joint velocities
        """
    
        v, _, self._drift_error = CalcFuncs.pd_servo(self.get_tool_pose(side='l', offset=True),
                                                    self.get_tool_pose(side='r', offset=True),
                                                    self._drift_error,
                                                    gain_p=gain_p,
                                                    gain_d=gain_d,
                                                    threshold=0.001,
                                                    method='angle-axis',
                                                    dt=1/self._control_rate)

        if on_taskspace:
            return v
        else:
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


class ROSUtils:

    @staticmethod
    def call_service(service_name, service_type, **kwargs):
        r"""
        Call a service in ROS
        :param service_name: name of the service
        :param service_type: type of the service
        :param request: request to the service
        :return: response from the service
        """
        rospy.wait_for_service(service_name)
        try:
            service = rospy.ServiceProxy(service_name, service_type)
            response = service(**kwargs)
            return response
        except rospy.ServiceException as e:
            rospy.logerr_once("Service call failed: %s" % e)
            return None
        
    @staticmethod
    def create_joint_traj_msg(joint_names: list, dt: float, traj_frame_id : str, joint_states: list = None, qdot: list = None, q: list = None):
        joint_traj = JointTrajectory()
        joint_traj.header.stamp = rospy.Time.now()
        joint_traj.header.frame_id = traj_frame_id
        joint_traj.joint_names = joint_names

        traj_point = JointTrajectoryPoint()
        if q is not None:
            traj_point.positions = q
        else:
            traj_point.positions = CalcFuncs.reorder_values(
                joint_states) + qdot * dt
            traj_point.velocities = qdot

        traj_point.time_from_start = rospy.Duration(dt)
        joint_traj.points = [traj_point]

        return joint_traj


def joy_init():
    pygame.init()
    joystick_count = pygame.joystick.get_count()
    if joystick_count == 0:
        raise Exception('No joystick found')
    else:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()

    return joystick

def joy_to_twist(joy, gain, base=False):

    vx, vy, vz, r, p, y = 0, 0, 0, 0, 0, 0
    done = False
    agressive = 0

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
        vy = lpf(joy.get_axis(1)) 
        vx = lpf(joy.get_axis(1)) 
        r = lpf(joy.get_axis(3))  
        p = lpf(joy.get_axis(4)) 

    else:

        trigger_side = 5 if not base else 2
        agressive =  (-lpf(joy[0][trigger_side]) + 1)/2 

        # Low pass filter
        vy = - lpf(joy[0][0]) / np.abs(lpf(joy[0][0])) if lpf(joy[0][0]) != 0 else 0
        vx = - lpf(joy[0][1]) / np.abs(lpf(joy[0][1])) if lpf(joy[0][1]) != 0 else 0
        y = joy[1][2] - joy[1][1] # button X and B

        if not base:
            vz = joy[1][3] - joy[1][0] # button Y and A
            r = lpf(joy[0][3]) / np.abs(lpf(joy[0][3])) if lpf(joy[0][3]) != 0 else 0
            p = - lpf(joy[0][4]) / np.abs(lpf(joy[0][4])) if lpf(joy[0][4]) != 0 else 0

        
    # ---------------------------------------------------------------------------#
    twist = np.zeros(6)
    twist[:3] = np.array([vx, vy, vz]) * gain[0] * agressive
    twist[3:] = np.array([r, p, y]) * gain[1] * agressive
    return twist, done

def lpf(value, threshold=0.1):
    return value if abs(value) > threshold else 0

def plot_joint_velocities(actual_data: np.ndarray, desired_data: np.ndarray, dt=0.001, title='Joint Velocities'):

    actual_data = np.array(actual_data)
    desired_data = np.array(desired_data)

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

    return fig, ax

def plot_manip_and_drift(constraint_distance: float, 
                         manipulabity_threshold: float, 
                         joint_limits : np.ndarray, 
                         joint_positions : np.ndarray ,
                         joint_velocities: np.ndarray, 
                         drift: np.ndarray, 
                         manip: np.ndarray, 
                         dt=0.001):
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
    plt.rcParams['text.usetex'] = True
    plt.rcParams.update({'font.size': 10})  # Adjust font size smaller if necessary
    plt.rcParams['figure.facecolor'] = 'white'


    # Prepare data
    fig, ax = plt.subplots(3, 2, figsize=(15, 20))
    time_space = np.linspace(0, len(drift) * dt, len(drift))
    manip_axes = ax[0, 0]
    drift_axes = ax[0, 1]
    manip_l = manip[0]
    manip_r = manip[1]    
    joint_pos_axes = ax[1, :]
    joint_vel_axes = ax[2, :]

    plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.15, hspace=0.5, wspace=0.2)  # Adjust horizontal spacing
    joint_limits = {
        'left': [joint_limits[0][:7], joint_limits[1][:7]],
        'right': [joint_limits[0][7:], joint_limits[1][7:]]     
    }

    # Plot joint positions

    if len(joint_positions[0]) != len(time_space):
        time_space = np.linspace(0, len(joint_positions['l'])
                                 * dt, len(joint_positions['l']) + 1)

    for i, side in enumerate(['left', 'right']):
        # for j in range(7):  # Assuming there are 7 joints
        #     joint_data = np.array([d[j] for d in joint_positions[i]])
        #     joint_pos_axes[i].plot(time_space, joint_data, label=f'Joint {j+1}')
        #     joint_pos_axes[i].axhline(y=joint_limits[side][0][j], color='r', linestyle='--')  # Lower limit
        #     joint_pos_axes[i].axhline(y=joint_limits[side][1][j], color='g', linestyle='--')  # Upper limit

        joint_data = np.array([d[5] for d in joint_positions[i]])
        joint_pos_axes[i].plot(time_space, joint_data, label=f'Joint {5+1}')
        joint_pos_axes[i].axhline(y=joint_limits[side][0][5], color='r', linestyle='--')  # Lower limit
        joint_pos_axes[i].axhline(y=joint_limits[side][1][5], color='g', linestyle='--')  # Upper limit
        joint_pos_axes[i].set_xlabel('Time [s]')
        joint_pos_axes[i].set_ylabel('Joint Position [rad]')
        joint_pos_axes[i].set_title(f'{side.capitalize()} Arm Joint Positions')
        joint_pos_axes[i].legend()

    # Plot joint velocities
    for i, side in enumerate(['left', 'right']):
        for j in range(7):  # Assuming there are 7 joints
            joint_data = np.array([d[j] for d in joint_velocities[side]])
            joint_vel_axes[i].plot(time_space, joint_data, label=f'Joint {j+1}')
        joint_vel_axes[i].set_title(f'{side.capitalize()} Arm Joint Velocities')
        joint_vel_axes[i].set_xlabel('Time [s]')
        joint_vel_axes[i].set_ylabel('Joint Velocity [rad/s]')
        joint_vel_axes[i].legend()

    # Plot manipulability data    # Plot drift data
    if len(manip_r) != len(time_space):
        time_space = np.linspace(0, len(manip_r)
                                 * dt, len(manip_r) + 1)

    if len(manip_l) != len(time_space):
        time_space = np.linspace(0, len(manip_l)
                                 * dt, len(manip_l) + 1)

    manip_axes.plot(time_space, manip_l, 'r', linewidth=1, label='Left' )
    manip_axes.plot(time_space, manip_r, 'b', linewidth=1, label='Right')
    manip_axes.set_title('Manipulability')
    manip_axes.set_xlabel('Time [s]')
    manip_axes.set_ylabel(r'$\omega$')
    manip_axes.axhline(y=manipulabity_threshold, color='k',
                       linewidth=1, linestyle='--', label='Threshold')

    manip_axes.annotate(f'Min Left {np.min(manip_l):.2f}',
                        xy=(time_space[np.argmin(manip_l)], np.min(manip_l)),
                        xytext=(10, 0), textcoords='offset points',
                        ha='center', va='bottom', color='r')

    manip_axes.annotate(f'Min Right {np.min(manip_r):.2f}',
                        xy=(time_space[np.argmin(manip_r)], np.min(manip_r)),
                        xytext=(10, -10), textcoords='offset points',
                        ha='center', va='top', color='b')
    
    manip_axes.legend()


    # Plot drift data
    if len(drift) != len(time_space):
        time_space = np.linspace(0, len(drift)
                                 * dt, len(drift) + 1)

    drift_axes.plot(time_space, drift, 'k', linewidth=1)
    drift_axes.set_title('Drift graph')
    drift_axes.set_xlabel('Time [s]')
    drift_axes.set_ylabel('Distance [m]')
    drift_axes.set_ylim([constraint_distance - 0.05, constraint_distance + 0.05]) 
    drift_axes.axhline(y=constraint_distance, color='r', linewidth=1, label=f'Constraint = {constraint_distance:.4f}')
    drift_axes.axhline(y=np.max(drift), color='k', linewidth=1, linestyle='--', label=f'Max drift = {np.max(drift):.4f}')
    drift_axes.axhline(y=np.min(drift), color='k', linewidth=1, linestyle='--', label=f'Min drift = {np.min(drift):.4f}')
    drift_axes.legend()

    # drift_axes.annotate(f'Constraint {constraint_distance:.4f}',
    #                     xy=(time_space[time_space.size//2], constraint_distance),
    #                     xytext=(10, 0),
    #                     textcoords='offset points',
    #                     ha='center', va='bottom', color='r')

    # drift_axes.annotate(f'Max {np.max(drift):.4f}',
    #                     xy=(time_space[np.argmax(drift)], np.max(drift)),
    #                     xytext=(10, 0), textcoords='offset points',
    #                     ha='center', va='bottom', color='k')

    # drift_axes.annotate(f'Min {np.min(drift):.4f}',
    #                     xy=(time_space[np.argmin(drift)], np.min(drift)),
    #                     xytext=(10, -10), textcoords='offset points',
    #                     ha='center', va='top', color='k')

    return fig, ax

