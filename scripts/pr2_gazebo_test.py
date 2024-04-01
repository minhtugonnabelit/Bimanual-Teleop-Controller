import tf
import rospy
from std_msgs.msg import Float64MultiArray, Float64
from sensor_msgs.msg import JointState, Joy
from geometry_msgs.msg import TwistStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from pr2_mechanism_msgs.srv import SwitchController, LoadController

import numpy as np
import spatialmath as sm
import roboticstoolbox as rtb
import matplotlib.pyplot as plt
import time
from enum import Enum
from scipy import linalg, optimize

# Import custom utility functions
from bimanual_controller.utility import *
from bimanual_controller.fakePR2 import FakePR2


SAMPLE_STATES = {
    'left': [np.pi/6, np.pi/6, np.pi/3, -np.pi/2, 0, -np.pi/4, np.pi/2],
    'right': [-np.pi/6, np.pi/6, -np.pi/3, -np.pi/2, 0, -np.pi/4, np.pi/2]
}

NEUTRAL_STATES = {
    'left': [0.05592020315366142, 0.4115547023030020313, 0.23241480964399752, -0.75718229886988179, 0.25000010026008326, -0.48229593735634957, 1.573265592638103776],
    'right': [-0.05869937106810763, 0.4107752715756987882, -0.23126457438489645, -0.75897762731364821, -0.25000005892831325, -0.4851061342000067, -1.5713531640700703562,]
}


class CONTROL_MODE(Enum):
    POSITION = 0
    VELOCITY = 1
    ACCELERATION = 2
    EFFORT = 3


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
        self._virtual_robot = FakePR2(launch_visualizer=False)
        self._rate = rospy.Rate(20)

        # Initialize arms trajectory controllers publishers
        left_arm_pub = rospy.Publisher(
            'l_arm_controller/command', JointTrajectory, queue_size=1)
        right_arm_pub = rospy.Publisher(
            'r_arm_controller/command', JointTrajectory, queue_size=1)
        self._arm_control_pub = {
            'left': left_arm_pub,
            'right': right_arm_pub
        }

        # # Initialize arms velocity publishers

        # right_arm_vel_pub = rospy.Publisher(
        #     'r_joint_group_vel_controller/command', Float64MultiArray, queue_size=1)
        # left_arm_vel_pub = rospy.Publisher(
        #     'l_joint_group_vel_controller/command', Float64MultiArray, queue_size=1)

        # self._arm_vel_pub = {
        #     'left': left_arm_vel_pub,
        #     'right': right_arm_vel_pub
        # }

        self._joint_group_vel_pub = rospy.Publisher(
            'pr2_joint_group_vel_controller/command', Float64MultiArray, queue_size=1)

        # Initialize the joint states subscriber
        self._joint_states = None
        self._joint_state_sub = rospy.Subscriber(
            '/joint_states', JointState, self._joint_state_callback)

        # Initialize the joystick subscriber
        self._joy_msg = None
        self._joystick_sub = rospy.Subscriber(
            '/joy', Joy, self._joystick_callback)

        self._vel_cmd_msg = None
        self._vel_cmd_sub = rospy.Subscriber(
            'pr2_joint_group_vel_controller/command', Float64MultiArray, self._velocities_command_callback)

        # # Initialize the twist subscriber
        # self._twist_msg = {
        #     'left': None,
        #     'right': None
        # }
        # self._twist_sub = {
        #     'left': rospy.Subscriber(
        #         '/l_arm_servo_server/delta_twist_cmds', TwistStamped, self._twist_callback, callback_args='left'),
        #     'right': rospy.Subscriber(
        #         '/r_arm_servo_server/delta_twist_cmds', TwistStamped, self._twist_callback, callback_args='right')
        # }

        # Initialize the transform listener
        self._tf_listener = tf.TransformListener()

        # Initialize the buffer for the joint velocities recording
        self._qdot_record = {
            'left': {
                'desired': [],
                'actual': []
            },
            'right': {
                'desired': [],
                'actual': []
            }}

        self._offset_distance = []
        self._constraint_is_set = False

        rospy.loginfo('Controller ready to go')
        rospy.on_shutdown(self._clean)

    def set_kinematics_constraints(self):
        r"""
        Set the kinematics constraints for the robot
        :return: None
        """
        tf.TransformListener().waitForTransform(
            'base_link', 'l_gripper_tool_frame', rospy.Time(), rospy.Duration(4.0))

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

        self._virtual_robot.set_constraints(virtual_pose)
        return True, virtual_pose

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

    def send_command(self, side: str, values: list):

        msg = Float64MultiArray()
        msg.data = values
        self._arm_vel_pub[side].publish(msg)

    def move_to_neutral(self):
        r"""
        Move the robot to neutral position
        :return: None
        """

        # while not rospy.is_shutdown():
        self.send_traj_command('right', CONTROL_MODE.POSITION,
                               SAMPLE_STATES['right'], 1)
        self.send_traj_command('left', CONTROL_MODE.POSITION,
                               SAMPLE_STATES['left'], 1)

    def teleop_test(self):
        r"""
        This test control loop function is used to perform coordination control on PR2 arms using single PS4 Joystick
        """

        rospy.loginfo('Start teleop')
        rospy.wait_for_message('/joy', Joy)

        done = False
        while not done:

            if self._joy_msg[1][-3]:

                self.move_to_neutral()


            if (self._joy_msg[1][4] * self._joy_msg[1][5]) and not self._constraint_is_set:

                self._constraint_is_set, _ = self.set_kinematics_constraints()
                rospy.loginfo('Constraint is set, switching controllers')
                rospy.sleep(1)
                PR2BiCoor._start_jg_vel_controller()


            if self._constraint_is_set:

                qdot = np.zeros(14)
                qdot_l = np.zeros(7)
                qdot_r = np.zeros(7)

                twist, done = joy_to_twist(self._joy_msg, [0.3, 0.3])
                if self._joy_msg[1][5]:
                
                    start_time = time.time()

                    jacob_right = self._virtual_robot.get_jacobian('right')
                    jacob_left = self._virtual_robot.get_jacobian('left')

                    qdot_l, qdot_r = duo_arm_qdot_constraint(
                        jacob_left, jacob_right, twist, activate_nullspace=True)

                    qdot = np.concatenate([qdot_r, qdot_l])

                    exec_time = time.time() - start_time
                    rospy.loginfo(f'Execution time: {exec_time}')
                
                msg = PR2BiCoor._joint_group_command_to_msg(qdot)
                self._joint_group_vel_pub.publish(msg)


            if done:
                rospy.loginfo('Done')
                rospy.signal_shutdown('Done')

            self._rate.sleep()

    def bimanual_teleop(self):
        r"""
        This test control loop function is used to control each PR2 arms using Razer Hydra motion controllers

        There will be two states for the control loop:
        1. Both arms are controlled by the Razer Hydra motion controllers independently
        2. Both arms are switched to bi-manual control mode with a constraint set in the middle of the arms
        """

        while not rospy.is_shutdown():

            # Get PR2 configuration to neutral joint position

            self._rate.sleep()

        pass

    def home(self):
        r"""
        Move the robot to home position
        :return: None
        """
        while not rospy.is_shutdown():
            self.move_to_neutral()
            self._rate.sleep()

    def path_trakcing_test(self):

        updated_joined_left = self._virtual_robot.get_tool_pose('left')
        arrived = False
        home = False
        qdot = np.zeros(14)
        qdot_l = np.zeros(7)
        qdot_r = np.zeros(7)
        target = np.eye(4)

        while not rospy.is_shutdown():

            if self._joy_msg[1][-3]:

                self.move_to_neutral()

            if (self._joy_msg[1][4] * self._joy_msg[1][5]) and not self._constraint_is_set:

                self._constraint_is_set, pose = self.set_kinematics_constraints()
                target = pose @ sm.SE3(0.2, 0, 0).A
                rospy.loginfo('constraint is set')

            if self._constraint_is_set:

                updated_joined_left = self._virtual_robot.get_tool_pose('left')
                middle_twist, arrived = rtb.p_servo(updated_joined_left,
                                                    target,
                                                    gain=0.1,
                                                    threshold=0.01,
                                                    method='angle-axis')  # Servoing in the virtual middle frame using angle-axis representation for angular error

                jacob_left = self._virtual_robot.get_jacobian('left')
                jacob_right = self._virtual_robot.get_jacobian('right')

                # Calculate the joint velocities using the Resolved Motion Rate Control (RMRC) method with the projection onto nullspace of Constraint Jacobian
                qdot_l, qdot_r = duo_arm_qdot_constraint(
                    jacob_left, jacob_right, middle_twist, activate_nullspace=True)

                # Visualization of the frames
                updated_joined_left = self._virtual_robot.get_tool_pose('left')

                qdot = np.concatenate([qdot_r, qdot_l])
                msg = PR2BiCoor._joint_group_command_to_msg(qdot)
                self._joint_group_vel_pub.publish(msg)

            self._qdot_record['left']['desired'].append(qdot_l)
            self._qdot_record['left']['actual'].append(
                reorder_values(self._joint_states.velocity[31:38]))

            self._qdot_record['right']['desired'].append(qdot_r)
            self._qdot_record['right']['actual'].append(
                reorder_values(self._joint_states.velocity[17:24]))

            if arrived:
                print('Arrived')
                break

            self._rate.sleep()

    @staticmethod
    def _joint_group_command_to_msg(values: list):
        r"""
        Convert the joint group command to Float64MultiArray message
        :param values: list of joint velocities
        :return: Float64MultiArray message
        """

        msg = Float64MultiArray()
        msg.data = values
        return msg

    # Callback functions

    def _joint_state_callback(self, msg: JointState):
        r"""
        Callback function for the joint state subscriber
        :param msg: JointState message
        :return: None
        """

        self._joint_states = msg
        self._virtual_robot.set_joint_states(self._joint_states.position)

    def _joystick_callback(self, msg: Joy):
        r"""
        Callback function for the joystick subscriber
        :param msg: Joy message
        :return: None
        """

        self._joy_msg = (msg.axes, msg.buttons)

    def _velocities_command_callback(self, msg: Float64MultiArray):
        r"""
        Callback function for the velocities command subscriber
        :param msg: Float64MultiArray message
        :return: None
        """

        self._qdot_record['right']['desired'].append(msg.data[:7])
        self._qdot_record['right']['actual'].append(
            reorder_values(self._joint_states.velocity[17:24]))
        self._qdot_record['left']['desired'].append(msg.data[7:])
        self._qdot_record['left']['actual'].append(
            reorder_values(self._joint_states.velocity[31:38]))

        self._offset_distance.append(np.linalg.norm(self._virtual_robot.get_tool_pose(
            'left')[:3, -1] - self._virtual_robot.get_tool_pose('right')[:3, -1]))

    def _twist_callback(self, msg: TwistStamped, side: str):
        r"""
        Callback function for the twist subscriber
        :param msg: TwistStamped message
        :param side: side of the arm
        :return: None
        """

        self._twist_msg[side] = msg

    # Service call function
    @staticmethod
    def _call(service_name: str, service_type: str, **kwargs):
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

    def _start_jg_vel_controller():
        r"""
        Switch the controllers
        :return: bool value of the service call
        """

        switched = PR2BiCoor._call('pr2_controller_manager/switch_controller',
                        SwitchController,
                        start_controllers=['pr2_joint_group_vel_controller',],
                        stop_controllers=['l_arm_controller','r_arm_controller'],
                        strictness=2)
        
        return switched
    
    def _kill_jg_vel_controller():
        r"""
        Switch the controllers
        :return: bool value of the service call
        """

        switched = PR2BiCoor._call('pr2_controller_manager/switch_controller',
                        SwitchController,
                        start_controllers=['l_arm_controller', 'r_arm_controller'],
                        stop_controllers=['pr2_joint_group_vel_controller'],
                        strictness=2)
        
        return switched

    # Clean up function

    def _clean(self):
        r"""
        Clean up function with dedicated shutdown procedure"""

        self._virtual_robot.shutdown()
        rospy.loginfo('Shutting down the robot')
        PR2BiCoor._kill_jg_vel_controller()
        fig1, ax1 = plot_joint_velocities(
            self._qdot_record['left']['actual'], self._qdot_record['left']['desired'], distance_data=self._offset_distance, dt=0.05, title='Left Arm')
        fig2, ax2 = plot_joint_velocities(
            self._qdot_record['right']['actual'], self._qdot_record['right']['desired'], distance_data=self._offset_distance, dt=0.05, title='Right Arm')
        plt.show()


if __name__ == "__main__":

    rospy.init_node('bcmp_test', log_level=rospy.INFO, anonymous=True,)
    rospy.logdebug('Command node initialized')
    controller = PR2BiCoor()
    controller.teleop_test()
