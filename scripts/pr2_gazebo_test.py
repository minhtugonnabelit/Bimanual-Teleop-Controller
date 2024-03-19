import numpy as np
from scipy import linalg
from enum import Enum
import threading

import spatialmath as sm
import matplotlib.pyplot as plt
import spatialgeometry as geometry
import roboticstoolbox as rtb
from swift import Swift

# Import custom utility functions
from utility import *
from fakePR2 import FakePR2

import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState, Joy
from std_msgs.msg import Float64MultiArray, Float64
import tf


LEFT_SAMPLE_JOINTSTATES = [np.pi/6,
                           np.pi/6,
                           np.pi/3,
                           -np.pi/2,
                           0,
                           -np.pi/4,
                           np.pi/2]

# LEFT_SAMPLE_JOINTSTATES = [0, 0, 0, 0, 0, 0, np.pi]
#
RIGHT_SAMPLE_JOINTSTATES = [-np.pi/6,
                            np.pi/6,
                            -np.pi/3,
                            -np.pi/2,
                            0,
                            -np.pi/4,
                            np.pi/2]

# RIGHT_SAMPLE_JOINTSTATES = [0, 0, 0, 0, 0, 0, np.pi]

# class CONTROL_MODE(Enum):
#     POSITION = 0
#     VELOCITY = 1
#     ACCELERATION = 2
#     EFFORT = 3


class PR2BiCoor:

    JOINT_NAMES = {
        'left': [
            "l_shoulder_pan_joint",
            "l_shoulder_lift_joint",
            "l_upper_arm_roll_joint",
            "l_elbow_flex_joint",
            "l_forearm_roll_joint",
            "l_wrist_flex_joint",
            "l_wrist_roll_joint",
        ],
        'right': [
            "r_shoulder_pan_joint",
            "r_shoulder_lift_joint",
            "r_upper_arm_roll_joint",
            "r_elbow_flex_joint",
            "r_forearm_roll_joint",
            "r_wrist_flex_joint",
            "r_wrist_roll_joint",
        ]}

    def __init__(self):

        # Initialize the robot model
        self._robot = FakePR2()

        # Initialize the joint states subscriber
        self._joint_states = None
        rospy.wait_for_message('/joint_states', JointState)
        self._joint_state_sub = rospy.Subscriber(
            '/joint_states', JointState, self._joint_state_callback)
        
        # Initialize the joystick subscriber
        self._joy_msg = None
        rospy.wait_for_message('/joy', Joy)
        self._joystick_sub = rospy.Subscriber(
            '/joy', Joy, self._joystick_callback)
        

        # Initialize arms controllers publishers
        # left_arm_pub = rospy.Publisher(
        #     'l_arm_controller/command', JointTrajectory, queue_size=1)
        # right_arm_pub = rospy.Publisher(
        #     'r_arm_controller/command', JointTrajectory, queue_size=1)

        # self._arm_control_pub = {
        #     'left': left_arm_pub,
        #     'right': right_arm_pub
        # }

        right_arm_vel_pub = rospy.Publisher(
            'r_joint_group_vel_controller/command', Float64MultiArray, queue_size=1)
        left_arm_vel_pub = rospy.Publisher(
            'l_joint_group_vel_controller/command', Float64MultiArray, queue_size=1)

        self._arm_vel_pub = {
            'left': left_arm_vel_pub,
            'right': right_arm_vel_pub
        }

        # Initialize the transform listener
        self._tf_listener = tf.TransformListener()
        rospy.on_shutdown(self._clean)

        self._rate = rospy.Rate(10)


    def set_kinematics_constraints(self):
        r"""
        Set the kinematics constraints for the robot
        :return: None
        """

        left_pose = self._tf_listener.lookupTransform(
            'l_gripper_tool_frame', 'base_link', rospy.Time(0))
        left_pose = tf.TransformerROS.fromTranslationRotation(
            left_pose[0], left_pose[1])
        # left_pose = self.posestamped_to_SE3(left_pose[0], left_pose[1])

        right_pose = self._tf_listener.lookupTransform(
            'r_gripper_tool_frame', 'base_link', rospy.Time(0))
        right_pose = tf.TransformerROS.fromTranslationRotation(
            right_pose[0], right_pose[1])
        # right_pose = self.posestamped_to_SE3(right_pose[0], right_pose[1])

        virtual_pose = np.eye(4, 4)
        virtual_pose[0:3, 3] = (left_pose[:3, -1] + right_pose[:3, -1]) / 2

        self.robot.set_constraints(virtual_pose)

    # def send_command(self, side, control_mode, value, duration):
    #     r"""
    #     Send the command to the robot
    #     :param joint_names: list of joint names
    #     :param control_mode: control mode
    #     :param value: control value
    #     :param duration: duration of the command
    #     :return: None
    #     """
    #     msg = JointTrajectory()
    #     msg.header.frame_id = 'torso_lift_link'
    #     msg.header.stamp = rospy.Time.now()
    #     msg.joint_names = self.JOINT_NAMES[side]
    #     point = JointTrajectoryPoint()
    #     if control_mode == CONTROL_MODE.POSITION:
    #         point.positions = value
    #     elif control_mode == CONTROL_MODE.VELOCITY:
    #         point.velocities = value
    #     elif control_mode == CONTROL_MODE.ACCELERATION:
    #         point.accelerations = value
    #     elif control_mode == CONTROL_MODE.EFFORT:
    #         point.effort = value
    #     point.time_from_start = rospy.Duration(duration)
    #     msg.points.append(point)

    #     self._arm_control_pub[side].publish(msg)

    def send_command(self, side: str, values: list):

        msg = Float64MultiArray()
        msg.data = values
        self._arm_vel_pub[side].publish(msg)
        # pass

    def _joint_state_callback(self, msg):
        r"""
        Callback function for the joint state subscriber
        :param msg: JointState message
        :return: None
        """

        self._joint_states = msg.position
        self._robot.set_joint_states(self.joint_states)

    def _joystick_callback(self, msg):
        r"""
        Callback function for the joystick subscriber
        :param msg: Joy message
        :return: None
        """

        self._joy_msg = msg

    def joy_to_twist(self, joy, gain):
        r"""
        Convert the joystick input to twist
        :param joy: joystick input
        :param gain: gain
        :return: twist
        """

        twist = np.zeros(6)
        twist[0] = joy.axes[1] * gain[0]
        twist[1] = joy.axes[0] * gain[0]
        twist[2] = joy.axes[4] * gain[0]
        twist[3] = joy.axes[3] * gain[1]
        twist[4] = joy.axes[2] * gain[1]
        twist[5] = joy.axes[5] * gain[1]

        return twist

    def move_to_neutral(self):
        r"""
        Move the robot to neutral position
        :return: None
        """

        while not rospy.is_shutdown():
            self.send_command('right', CONTROL_MODE.POSITION,
                              RIGHT_SAMPLE_JOINTSTATES, 3)
            self.send_command('left', CONTROL_MODE.POSITION,
                              LEFT_SAMPLE_JOINTSTATES, 3)
            self._rate.sleep()

    def bimanual_controller(self):
        r"""
        Bimanual controller
        :return: None
        """

        pass

    def _clean(self):

        self._robot.q()


if __name__ == "__main__":

    # Initialize the ROS node
    rospy.init_node('test_command')
    controller = PR2BiCoor()
    controller.move_to_neutral()
