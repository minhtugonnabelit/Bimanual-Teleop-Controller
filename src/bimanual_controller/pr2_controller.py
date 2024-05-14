# /usr/bin/env python3

# This file contains the PR2Controller class that is used to control the PR2 robot
#
# The PR2Controller class is used to control the PR2 robot.
# It is responsible for setting up the robot model, initializing the arms, and handling the joint states and joystick messages.
# It also provides functions to move the robot to a neutral position, open and close the grippers, and send joint velocities to the robot.

import rospy
import tf
from tf import TransformerROS as tfROS

import actionlib
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState, Joy
from geometry_msgs.msg import TwistStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from pr2_mechanism_msgs.srv import SwitchController, UnloadController
from pr2_controllers_msgs.msg import Pr2GripperCommand, JointTrajectoryAction, JointTrajectoryGoal
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal

import numpy as np
import spatialmath as sm
import roboticstoolbox as rtb
import matplotlib.pyplot as plt
from scipy import linalg, optimize

from bimanual_controller.utility import *
from bimanual_controller.arm_controller import ArmController


class PR2Controller:
    r"""
    Class to control the PR2 robot
    """

    def __init__(self, name, log_level, rate):

        # Initialize the robot model to handle constraints and calculate Jacobians
        rospy.init_node(name, log_level=log_level, anonymous=True)
        self._virtual_robot = FakePR2(launch_visualizer=False)
        self._rate = rospy.Rate(rate)
        self._dt = 1/rate

        self.right_arm = ArmController(arm='r',
                                       arm_group_joint_names=JOINT_NAMES['right'],
                                       arm_group_controller_name="/r_arm_joint_group_velocity_controller",
                                       controller_cmd_type=Float64MultiArray,
                                       enable_gripper=True,
                                       gripper_cmd_type=Pr2GripperCommand)

        self.left_arm = ArmController(arm='l',
                                      arm_group_joint_names=JOINT_NAMES['left'],
                                      arm_group_controller_name="/l_arm_joint_group_velocity_controller",
                                      controller_cmd_type=Float64MultiArray,
                                      enable_gripper=True,
                                      gripper_cmd_type=Pr2GripperCommand)

        # Initialize the joint states subscriber
        self._joint_states = None
        self._joint_state_sub = rospy.Subscriber(
            '/joint_states', JointState, self.__joint_state_callback)

        # Initialize the joystick subscriber
        self._joy_msg = None
        self._joystick_sub = rospy.Subscriber(
            '/joy', Joy, self.__joystick_callback)

        self._hydra_joy_msg = {
            'l': None,
            'r': None
        }
        self._hydra_joy_sub = {
            'l': rospy.Subscriber('/hydra_left_joy', Joy, self.__hydra_joystick_callback, callback_args='l'),
            'r': rospy.Subscriber('/hydra_right_joy', Joy, self.__hydra_joystick_callback, callback_args='r')
        }
        self._controller_frame_id = {
            'l': 'hydra_left_grab',
            'r': 'hydra_right_grab'
        }
        self._hydra_base_frame_id = 'hydra_base'

        # Initialize the buffer for the joint velocities recording
        self.constraint_distance = 0
        self._offset_distance = []
        self._manipulability = [[], []]
        self._q_record = [[], []]
        self._constraint_is_set = False
        self._qdot_record = {
            'left': [],
            'right': []
        }

        self._qdot_record_PID = {
            'left': {'desired': [],  'actual': []},
            'right': {'desired': [],  'actual': []}
        }

        # Initialize the transform listener
        self._tf_listener = tf.TransformListener()
        self._tf_broadcaster = tf.TransformBroadcaster()

        rospy.loginfo('Controller ready to go')
        rospy.on_shutdown(self.__clean)

    def __clean(self):
        r"""
        Clean up function with dedicated shutdown procedure"""

        self._virtual_robot.shutdown()
        rospy.loginfo('Shutting down the virtual robot')
        PR2Controller.kill_jg_vel_controller()
        self.move_to_neutral()

        self.joint_limits = self._virtual_robot.get_joint_limits_all()
        fig3, ax3 = plot_manip_and_drift(
            self.constraint_distance,
            self.manip_thresh,
            self.joint_limits,
            self._q_record,
            self._qdot_record,
            self._offset_distance,
            self._manipulability,
            dt=self._dt)

        plt.show()

    def sleep(self):
        r"""
        Sleep the controller
        """

        self._rate.sleep()

    def set_kinematics_constraints(self):
        r"""
        Set the kinematics constraints for the robot
        :return: bool value of the service call and the virtual pose
        """
        tf.TransformListener().waitForTransform(
            'base_link', 'l_gripper_tool_frame', rospy.Time(), rospy.Duration(4.0))

        left_pose = self._tf_listener.lookupTransform(
            'base_footprint',
            'l_gripper_tool_frame',
            rospy.Time(0))

        left_pose = tfROS.fromTranslationRotation(
            tfROS,
            translation=left_pose[0],
            rotation=left_pose[1])

        right_pose = self._tf_listener.lookupTransform(
            'base_footprint',
            'r_gripper_tool_frame',
            rospy.Time(0))

        right_pose = tfROS.fromTranslationRotation(
            tfROS,
            translation=right_pose[0],
            rotation=right_pose[1])

        virtual_pose = np.eye(4)
        virtual_pose[:3, -1] = (left_pose[:3, -1] + right_pose[:3, -1]) / 2
        constraint_distance = np.linalg.norm(
            left_pose[:3, -1] - right_pose[:3, -1])

        self._virtual_robot.set_constraints(virtual_pose)
        return True, virtual_pose, constraint_distance

    def set_manip_thresh(self, manip_thresh):
        r"""
        Set the manipulability threshold for the robot
        :param manip_thresh: manipulability threshold
        :return: None
        """
        self.manip_thresh = manip_thresh

    def move_to_neutral(self):
        r"""
        Move the robot to neutral position
        :return: None
        """

        client_r = actionlib.SimpleActionClient(
            'r_arm_controller/follow_joint_trajectory',
            FollowJointTrajectoryAction)
        client_r.wait_for_server()
        goal_r = FollowJointTrajectoryGoal()
        goal_r.trajectory = PR2Controller._create_joint_traj_msg(
            JOINT_NAMES['right'],
            3,
            q=SAMPLE_STATES['right'])
        
        client_r.send_goal(goal_r)
        client_r.wait_for_result()

        client_l = actionlib.SimpleActionClient(
            'l_arm_controller/follow_joint_trajectory', 
            FollowJointTrajectoryAction)
        client_l.wait_for_server()
        goal_l = FollowJointTrajectoryGoal()
        goal_l.trajectory = PR2Controller._create_joint_traj_msg(
            JOINT_NAMES['left'], 
            3, 
            q=SAMPLE_STATES['left'])
        
        client_l.send_goal(goal_l)
        client_l.wait_for_result()

        return client_l.wait_for_result()

    # Callback functions

    def __joint_state_callback(self, msg: JointState):
        r"""
        Callback function for the joint state subscriber
        :param msg: JointState message
        :return: None
        """

        self._joint_states = msg
        self._virtual_robot.set_states(self._joint_states.position)

    def __joystick_callback(self, msg: Joy):
        r"""
        Callback function for the joystick subscriber
        :param msg: Joy message
        :return: None
        """

        self._joy_msg = (msg.axes, msg.buttons)

    def __hydra_joystick_callback(self, msg: Joy, side: str):
        r"""
        Callback function for the hydra joystick subscriber
        :param msg: Joy message
        :param side: side of the robot
        :return: None
        """

        self._hydra_joy_msg[side] = (msg.axes, msg.buttons)

    # Getters

    def get_joy_msg(self):
        r"""
        Get the joystick message
        :return: Joy message
        """

        return self._joy_msg

    def get_hydra_joy_msg(self, side: str):
        r"""
        Get the hydra joystick message
        :param side: side of the robot
        :return: Joy message
        """

        return self._hydra_joy_msg[side]

    def get_twist(self, side: str, synced=False, gain=[1, 1]):
        r"""
        Get the twist of the specified side hydra grab frame in the hydra base frame
        :param side: side of the robot
        :param synced: flag to check if the transform is synced

        :return: TwistStamped message
        """
        try:
            self._tf_listener.waitForTransform(
                self._controller_frame_id[side], self._hydra_base_frame_id, rospy.Time(), rospy.Duration(20))
            synced = True
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("Waiting for tf")
            synced = False

        if not synced:
            return None, synced

        twist = self._tf_listener.lookupTwist(
            self._controller_frame_id[side], self._hydra_base_frame_id, rospy.Time(), rospy.Duration(1/CONTROL_RATE)) if synced else None

        xdot = np.zeros(6)
        xdot[:3] = np.array(twist[0]) * gain[0]
        xdot[3:] = np.array(twist[1]) * gain[1]

        return xdot, synced

    def get_joint_states(self):
        r"""
        Get the joint states
        :return: JointState message
        """

        return self._joint_states

    def get_jacobian(self, side: str):
        r"""
        Get the Jacobian of the side
        :param side: side of the robot
        :return: Jacobian matrix
        """

        return self._virtual_robot.get_jacobian(side)

    def joint_limit_damper(self, qdot, steepness=10) -> list:
        r"""
        joint limit avoidance mechanism with speed scaling factor calculated based on
        how close individual joint to its limit. We then get a list of 2xn scaling factor that range from 0 to 1.
        The factor will always be 1 unless a joint get into soft limit, thus lead to the general
        factor that applied entirely to set of joint velocity

        Args:
            qdot (list): Joint velocities

        Returns:
            list: Joint velocities with joint limit avoidance mechanism applied
        """
        joint_limits_damper, max_weights = self._virtual_robot.joint_limits_damper(
            qdot, steepness)
        if max_weights > 0.75:
            rospy.logwarn(
                f"Joint limit avoidance mechanism is applied with max weight: {max_weights}")

        return joint_limits_damper

    def task_drift_compensation(self, gain=5, taskspace_compensation=True):
        r"""
        Task drift compensation mechanism that calculate the drift vector between two end-effectors
        and then apply the RMRC to the drift vector to get the joint velocity that will be applied to the robot

        Args:
            gain (int, optional): Gain of the RMRC. Defaults to 5.
            taskspace_compensation (bool, optional): Flag to indicate if the compensation is in task space. Defaults to True.

        Returns:
            list: Joint velocities with task drift compensation mechanism applied
        """

        return self._virtual_robot.task_drift_compensation(gain, taskspace_compensation)

    @ staticmethod
    def __call_service(service_name: str, service_type: str, **kwargs):
        r"""
        Call the service
        :param service_name: name of the service
        :param service_type: type of the service
        :param kwargs: additional arguments
        :return: bool value of the service call
        """

        rospy.wait_for_service(service_name)
        try:
            service = rospy.ServiceProxy(service_name, service_type)
            response = service(**kwargs)
            return response.ok
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    @ staticmethod
    def _create_joint_traj_msg(joint_names: list, dt: float, joint_states: list = None, qdot: list = None, q: list = None):
        r"""
        Create a joint trajectory message.
        If q desired is not provided, the joint velocities are used to calculate the joint positions

        Args:
            joint_names (list): List of joint names.
            dt (float): Time duration.
            joint_states (list, optional): List of joint states. Defaults to None.
            qdot (list, optional): List of joint velocities. Defaults to None.
            q (list, optional): List of joint positions. Defaults to None.

        Returns:
            JointTrajectory: JointTrajectory message.
        """

        joint_traj = JointTrajectory()
        joint_traj.header.stamp = rospy.Time.now()
        joint_traj.header.frame_id = 'torso_lift_link'
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

    @ staticmethod
    def start_jg_vel_controller():
        r"""
        Switch the controllers
        :return: bool value of the service call
        """

        switched = PR2Controller.__call_service('pr2_controller_manager/switch_controller',
                                                SwitchController,
                                                start_controllers=[
                                                    'r_arm_joint_group_velocity_controller',
                                                    'l_arm_joint_group_velocity_controller'],
                                                stop_controllers=[
                                                    'r_arm_controller',
                                                    'l_arm_controller'],
                                                strictness=1)

        return switched

    @ staticmethod
    def kill_jg_vel_controller():
        r"""
        Switch the controllers
        :return: bool value of the service call
        """
        rospy.loginfo(
            'Switching controllers and unloading velocity controllers')

        switched = PR2Controller.__call_service('pr2_controller_manager/switch_controller',
                                                SwitchController,
                                                start_controllers=[
                                                    'r_arm_controller',
                                                    'l_arm_controller'],
                                                stop_controllers=[
                                                    'r_arm_joint_group_velocity_controller',
                                                    'l_arm_joint_group_velocity_controller'],
                                                strictness=1)

        PR2Controller.__call_service('pr2_controller_manager/unload_controller',
                                     UnloadController, name='l_arm_joint_group_velocity_controller')
        PR2Controller.__call_service('pr2_controller_manager/unload_controller',
                                     UnloadController, name='r_arm_joint_group_velocity_controller')

        rospy.loginfo('Controllers switched and unloaded')

        return switched

    # Record function

    def store_drift(self):
        r"""
        Store the drift in the buffer
        """
        self._offset_distance.append(
            np.linalg.norm(
                self._virtual_robot.get_tool_pose(side=self.left_arm.get_arm_name(), offset=False)[:3, -1] -
                self._virtual_robot.get_tool_pose(side=self.right_arm.get_arm_name(), offset=False)[:3, -1]))

    def store_manipulability(self):
        r"""
        Store the manipulability in the buffer
        """

        self._manipulability[0].append(CalcFuncs.manipulability(
            self.get_jacobian(self.left_arm.get_arm_name())))
        self._manipulability[1].append(CalcFuncs.manipulability(
            self.get_jacobian(self.right_arm.get_arm_name())))

    def store_constraint_distance(self, distance: float):
        r"""
        Store the constraint distance in the buffer

        Args:
            distance (float): Distance between the end-effectors.
        """

        self.constraint_distance = distance

    def store_joint_velocities_for_PID_tuner(self, side: str, qdot: list):
        r"""
        Store the joint velocities in the buffer for PID tunner function

        Args:
            side (str): Side of the robot.
            qdot (list): List of joint velocities.
        """

        self._qdot_record_PID[side]['desired'].append(qdot)

        if side == 'left':
            self._qdot_record_PID[side]['actual'].append(
                CalcFuncs.reorder_values(self._joint_states.velocity[31:38]))
        else:
            self._qdot_record_PID[side]['actual'].append(
                CalcFuncs.reorder_values(self._joint_states.velocity[17:24]))

    def store_joint_velocities(self, side: str, qdot: list):
        r"""
        Store the joint velocities in the buffer

        Args:
            side (str): Side of the robot.
            qdot (list): List of joint velocities.
        """

        self._qdot_record[side].append(qdot)

    def store_joint_positions(self):
        r"""
        Store the joint positions in the buffer

        Args:
            side (str): Side of the robot.
            q (list): List of joint positions.
        """
        self._q_record[0].append(CalcFuncs.reorder_values(
            self._joint_states.position[31:38]))
        self._q_record[1].append(CalcFuncs.reorder_values(
            self._joint_states.position[17:24]))

        # self._q_record[side].append(self._virtual_robot.get_joint_positions(side))
