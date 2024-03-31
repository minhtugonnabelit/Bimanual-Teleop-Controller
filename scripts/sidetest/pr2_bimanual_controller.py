import numpy as np
from enum import Enum

import tf
import rospy
from std_msgs.msg import Float64MultiArray, Float64, Header
from sensor_msgs.msg import JointState, Joy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# Import custom utility functions
from utility import *


LEFT_SAMPLE_JOINTSTATES = [np.pi/6,
                           np.pi/6,
                           np.pi/3,
                           -np.pi/2,
                           0,
                           -np.pi/4,
                           np.pi/2]

# LEFT_SAMPLE_JOINTSTATES = [0, 0, 0, 0, 0, 0, np.pi]

RIGHT_SAMPLE_JOINTSTATES = [-np.pi/6,
                            np.pi/6,
                            -np.pi/3,
                            -np.pi/2,
                            0,
                            -np.pi/4,
                            np.pi/2]

# RIGHT_SAMPLE_JOINTSTATES = [0, 0, 0, 0, 0, 0, np.pi]


class CONTROL_MODE(Enum):
    POSITION = 0
    VELOCITY = 1
    ACCELERATION = 2
    EFFORT = 3


class PR2BimanualController:

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

    def __init__(self, rate=5):

        # Initialize the robot model
        rospy.init_node('test_command', anonymous=True)
        rospy.loginfo('Initializing the robot model')
        self._robot = FakePR2()
        print('bruh3')
        self._constraint_is_set = False
        self._rate = rospy.Rate(rate)
        rospy.loginfo('Finish setting up PR2 on Swift')

        # Initialize the joint states subscriber
        self._joint_states = None
        self._joint_state_sub = rospy.Subscriber(
            '/joint_states', JointState, self._joint_state_callback)

        # Initialize the joystick subscriber
        self._joy_msg = None
        self._joystick_sub = rospy.Subscriber(
            '/joy', Joy, self._joystick_callback)

        # Initialize arms controllers publishers
        left_arm_pub = rospy.Publisher(
            'l_arm_controller/command', JointTrajectory, queue_size=1)
        right_arm_pub = rospy.Publisher(
            'r_arm_controller/command', JointTrajectory, queue_size=1)

        self._arm_control_pub = {
            'left': left_arm_pub,
            'right': right_arm_pub
        }

        # Initialize arms velocity publishers
        right_arm_vel_pub = rospy.Publisher(
            'r_joint_group_vel_controller/command', Float64MultiArray, queue_size=1)
        left_arm_vel_pub = rospy.Publisher(
            'l_joint_group_vel_controller/command', Float64MultiArray, queue_size=1)

        self._arm_vel_pub = {
            'left': left_arm_vel_pub,
            'right': right_arm_vel_pub
        }

        left_gripper_pub = rospy.Publisher(
            'l_gripper_controller/command', 
        )

        # Initialize the transform listener
        self._tf_listener = tf.TransformListener()
        rospy.on_shutdown(self._clean)

    def _clean(self):

        self._robot.shutdown()

    def _joint_state_callback(self, msg):
        r"""
        Callback function for the joint state subscriber
        :param msg: JointState message
        :return: None
        """

        self._joint_states = msg.position
        self._robot.set_joint_states(self._joint_states)

    def _joystick_callback(self, msg):
        r"""
        Callback function for the joystick subscriber
        :param msg: Joy message
        :return: None
        """

        self._joy_msg = (msg.axes, msg.buttons)

    @staticmethod
    def _joint_group_command_to_msg(values: list):

        msg = Float64MultiArray()
        msg.data = values
        return msg

    def set_constraints(self):
        r"""
        Set the kinematics constraints for the robot
        :return: Boolean value indicating if the constraints are set
        """

        left_pose = self._tf_listener.lookupTransform(
            'base_footprint', 'l_gripper_tool_frame', rospy.Time(0))
        left_pose = tf.TransformerROS.fromTranslationRotation(
            tf.TransformerROS, translation=left_pose[0], rotation=left_pose[1])

        right_pose = self._tf_listener.lookupTransform(
            'base_footprint', 'r_gripper_tool_frame', rospy.Time(0))
        right_pose = tf.TransformerROS.fromTranslationRotation(
            tf.TransformerROS, translation=right_pose[0], rotation=right_pose[1])

        virtual_pose = np.eye(4)
        virtual_pose[:3, -1] = (left_pose[:3, -1] + right_pose[:3, -1]) / 2

        self._robot.set_constraints(virtual_pose)
        return True

    def send_traj_command(self, side, control_mode, value, duration):
        r"""
        Send the command to the robot
        :param joint_names: list of joint names
        :param control_mode: control mode
        :param value: control value
        :param duration: duration of the command
        :return: None
        """
        msg = JointTrajectory()
        msg.header.frame_id = 'torso_lift_link'
        msg.header.stamp = rospy.Time.now()
        msg.joint_names = self.JOINT_NAMES[side]
        point = JointTrajectoryPoint()
        if control_mode == CONTROL_MODE.POSITION:
            point.positions = value
        elif control_mode == CONTROL_MODE.VELOCITY:
            point.velocities = value
        elif control_mode == CONTROL_MODE.ACCELERATION:
            point.accelerations = value
        elif control_mode == CONTROL_MODE.EFFORT:
            point.effort = value
        point.time_from_start = rospy.Duration(duration)
        msg.points.append(point)

        self._arm_control_pub[side].publish(msg)

    def move_to_neutral(self):
        r"""
        Move the robot to neutral position
        :return: None
        """

        self.send_traj_command('right', CONTROL_MODE.POSITION,
                               RIGHT_SAMPLE_JOINTSTATES, 3)
        self.send_traj_command('left', CONTROL_MODE.POSITION,
                               LEFT_SAMPLE_JOINTSTATES, 3)

    # -------------------------------------------------#
    ###     Functions for testing the controller    ###
    # -------------------------------------------------#

    def bmcp_test(self):

        rospy.wait_for_message('/joy', Joy)
        while not rospy.is_shutdown():

            if self._joy_msg[1][-3]:
                self.move_to_neutral()

            if (self._joy_msg[1][4] * self._joy_msg[1][5]) and not self._constraint_is_set:
                self._constraint_is_set = self.set_constraints()

            if self._constraint_is_set:  # If the constraints are set, then perform the control loop for bi-manipulation
                
                qdot_l = np.zeros(7)
                qdot_r = np.zeros(7)

                if self._joy_msg[1][5]:  # Safety trigger

                    twist, done = joy_to_twist(self._joy_msg, [0.1, 0.1])
                    jacob_left = self._robot.get_jacobian('left')
                    jacob_right = self._robot.get_jacobian('right')

                    qdot_l, qdot_r = duo_arm_qdot_constraint(
                        jacob_left, jacob_right, twist, activate_nullspace=True)

                self._arm_vel_pub['left'].publish(
                    PR2BimanualController._joint_group_command_to_msg(qdot_l))
                self._arm_vel_pub['right'].publish(
                    PR2BimanualController._joint_group_command_to_msg(qdot_r))

            self._rate.sleep()

    def single_joint_test(self):

        V = 0.6
        qdot = np.zeros(14)

        rospy.wait_for_message('/joy', Joy)
        while not rospy.is_shutdown():

            dir = self._joy_msg[0][1] / np.abs(self._joy_msg[0]
                                               [1]) if np.abs(self._joy_msg[0][1]) > 0.1 else 0
            if self._joy_msg[1][4]:

                for i in range(7):
                    qdot[i+7] = V * dir * self._joy_msg[1][i]

            if self._joy_msg[1][5]:

                for i in range(7):
                    qdot[i] = V * dir * self._joy_msg[1][i]

            if not (self._joy_msg[1][4] + self._joy_msg[1][5]):
                qdot = np.zeros(14)

            self._arm_vel_pub['left'].publish(
                PR2BimanualController._joint_group_command_to_msg(qdot[7:]))
            self._arm_vel_pub['right'].publish(
                PR2BimanualController._joint_group_command_to_msg(qdot[:7]))

            self._rate.sleep()
