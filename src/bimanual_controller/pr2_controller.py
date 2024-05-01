import tf
import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState, Joy
from geometry_msgs.msg import TwistStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from pr2_mechanism_msgs.srv import SwitchController, UnloadController
from pr2_controllers_msgs.msg import Pr2GripperCommand

import numpy as np
import spatialmath as sm
import roboticstoolbox as rtb
import matplotlib.pyplot as plt
import time
from scipy import linalg, optimize

from bimanual_controller.utility import *
from bimanual_controller.fakePR2 import FakePR2


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

        # Initialize the buffer for the joint velocities recording
        self._qdot_record = {
            'left': {'desired': [],  'actual': []},
            'right': {'desired': [],  'actual': []}
        }
        self.constraint_distance = 0
        self._offset_distance = []
        self._constraint_is_set = False

        # Initialize the gripper command publishers
        self._gripper_cmd_pub = {
            'right': rospy.Publisher('r_gripper_controller/command', Pr2GripperCommand, queue_size=1),
            'left': rospy.Publisher('l_gripper_controller/command', Pr2GripperCommand, queue_size=1)
        }

        # Initialize arms trajectory controllers publishers
        self._arm_traj_control_pub = {
            'left': rospy.Publisher('l_arm_controller/command', JointTrajectory, queue_size=1),
            'right': rospy.Publisher('r_arm_controller/command', JointTrajectory, queue_size=1)
        }

        # Initialize the joint group velocity controller publisher
        self._arms_vel_controller_pub = {
            'right': rospy.Publisher('/r_arm_joint_group_velocity_controller/command', Float64MultiArray, queue_size=1),
            'left': rospy.Publisher('/l_arm_joint_group_velocity_controller/command', Float64MultiArray, queue_size=1)
        }

        # Initialize the joint states subscriber
        self._joint_states = None
        self._joint_state_sub = rospy.Subscriber(
            '/joint_states', JointState, self.__joint_state_callback)

        # Initialize the joystick subscriber
        self._joy_msg = None
        self._joystick_sub = rospy.Subscriber(
            '/joy', Joy, self.__joystick_callback)

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
        fig1, ax1 = plot_joint_velocities(
            self._qdot_record['left']['actual'], self._qdot_record['left']['desired'], distance_data=self._offset_distance, constraint_distance = self.constraint_distance, dt=self._dt, title='left')
        fig2, ax2 = plot_joint_velocities(
            self._qdot_record['right']['actual'], self._qdot_record['right']['desired'], distance_data=self._offset_distance, constraint_distance = self.constraint_distance, dt=self._dt, title='right')
        plt.show()

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

        left_pose = tf.TransformerROS.fromTranslationRotation(
            tf.TransformerROS,
            translation=left_pose[0],
            rotation=left_pose[1])

        right_pose = self._tf_listener.lookupTransform(
            'base_footprint',
            'r_gripper_tool_frame',
            rospy.Time(0))

        right_pose = tf.TransformerROS.fromTranslationRotation(
            tf.TransformerROS,
            translation=right_pose[0],
            rotation=right_pose[1])

        virtual_pose = np.eye(4)
        virtual_pose[:3, -1] = (left_pose[:3, -1] + right_pose[:3, -1]) / 2
        constraint_distance = np.linalg.norm(left_pose[:3, -1] - right_pose[:3, -1])

        self._virtual_robot.set_constraints(virtual_pose)
        return True, virtual_pose, constraint_distance

    def move_to_neutral(self):
        r"""
        Move the robot to neutral position
        :return: None
        """
        self._arm_traj_control_pub['right'].publish(
            PR2Controller.__create_joint_traj_msg(JOINT_NAMES['right'], 3, q=SAMPLE_STATES['right']))
        self._arm_traj_control_pub['left'].publish(
            PR2Controller.__create_joint_traj_msg(JOINT_NAMES['left'], 3, q=SAMPLE_STATES['left']))

    # Gripper functions

    def open_gripper(self, side: str):
        r"""
        Close the gripper
        :param side: side of the robot
        :return: None
        """

        msg = Pr2GripperCommand()
        msg.position = 0.08
        msg.max_effort = 10.0
        self._gripper_cmd_pub[side].publish(msg)

    def close_gripper(self, side: str):
        r"""
        Close the gripper
        :param side: side of the robot
        :return: None
        """

        msg = Pr2GripperCommand()
        msg.position = 0.0
        msg.max_effort = 10.0
        self._gripper_cmd_pub[side].publish(msg)

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

    # Getters
    def get_joy_msg(self):
        r"""
        Get the joystick message
        :return: Joy message
        """

        return self._joy_msg

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
    
    def get_drift_compensation(self) -> np.ndarray:
        r"""
        Get the drift compensation for the robot
        :return: drift compensation velocities as joint velocities
        """
        v, _ = rtb.p_servo(self._virtual_robot.get_tool_pose('left', offset=True), 
                        self._virtual_robot.get_tool_pose('right', offset=True),
                        1, 0.01,
                        method='angle-axis')
        
        # get fix velocities for the drift for both linear and angular velocities
        qdot_fix_left = CalcFuncs.rmrc(self._virtual_robot.get_jacobian('left'), v, w_thresh=0.05)
        qdot_fix_right = CalcFuncs.rmrc(self._virtual_robot.get_jacobian('right'), -v, w_thresh=0.05)

        return np.r_[qdot_fix_left, qdot_fix_right]

    def send_joint_velocities(self, side: str, qdot: list):
        r"""
        Send the joint velocities to the robot
        :param side: side of the robot
        :param qdot: list of joint velocities
        :return: None
        """

        self._arms_vel_controller_pub[side].publish(
            PR2Controller.__joint_group_command_to_msg(qdot))

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def __joint_group_command_to_msg(values: list):
        r"""
        Convert the joint group command to Float64MultiArray message

        Args:
            values (list): List of joint values

        Returns:
            Float64MultiArray: Float64MultiArray message
        """

        msg = Float64MultiArray()
        msg.data = values
        return msg

    @staticmethod
    def __create_joint_traj_msg(joint_names: list, dt: float, joint_states: list = None, qdot: list = None, q: list = None):
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
        joint_traj.header.frame_id = 'torso_lift_link'
        joint_traj.joint_names = joint_names

        traj_point = JointTrajectoryPoint()
        if q is not None:
            traj_point.positions = q
        else:
            traj_point.positions = reorder_values(joint_states) + qdot * dt
            traj_point.velocities = qdot

        traj_point.time_from_start = rospy.Duration(dt)
        joint_traj.points = [traj_point]

        return joint_traj

    def store_joint_velocities(self, side: str, qdot: list):
        r"""
        Store the joint velocities in the buffer

        Args:
            side (str): Side of the robot.
            qdot (list): List of joint velocities.
        """

        self._qdot_record[side]['desired'].append(qdot)

        if side == 'left':
            self._qdot_record[side]['actual'].append(
                reorder_values(self._joint_states.velocity[31:38]))
        else:
            self._qdot_record[side]['actual'].append(
                reorder_values(self._joint_states.velocity[17:24]))

    def store_drift(self):
        r"""
        Store the drift in the buffer
        """
        self._offset_distance.append(np.linalg.norm(self._virtual_robot.get_tool_pose(
            'left', offset=False)[:3, -1] - self._virtual_robot.get_tool_pose('right', offset=False)[:3, -1]))

    def store_constraint_distance(self, distance: float):
        r"""
        Store the constraint distance in the buffer

        Args:
            distance (float): Distance between the end-effectors.
        """

        self.constraint_distance = distance


    def sleep(self):
        r"""
        Sleep the controller
        """

        self._rate.sleep()
