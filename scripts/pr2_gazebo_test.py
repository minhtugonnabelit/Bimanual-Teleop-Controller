import numpy as np
from enum import Enum

# Import custom utility functions
from utility import *
from fakePR2 import FakePR2

import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState, Joy
from std_msgs.msg import Float64MultiArray, Float64, Header
import tf

from scipy import linalg
import spatialmath as sm
import roboticstoolbox as rtb


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
        self._robot = FakePR2(launch_visualizer=False)
        self._constraint_is_set = False
        self._rate = rospy.Rate(20)

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

        # Initialize the transform listener
        self._tf_listener = tf.TransformListener()
        self._desired_qdot_left = list()
        self._actual_qdot_left = list()
        self._desired_qdot_right = list()
        self._actual_qdot_right = list()
        rospy.on_shutdown(self._clean)

    def set_kinematics_constraints(self):
        r"""
        Set the kinematics constraints for the robot
        :return: None
        """
        tf.TransformListener().waitForTransform(
            'base_link', 'l_gripper_tool_frame', rospy.Time(), rospy.Duration(4.0))
        
        left_pose = self._tf_listener.lookupTransform(
            'base_link', 'l_gripper_tool_frame', rospy.Time(0))
        left_pose = tf.TransformerROS.fromTranslationRotation(
            tf.TransformerROS, translation=left_pose[0], rotation=left_pose[1])

        right_pose = self._tf_listener.lookupTransform(
            'base_link', 'r_gripper_tool_frame', rospy.Time(0))
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

    def send_command(self, side: str, values: list):

        msg = Float64MultiArray()
        msg.data = values
        self._arm_vel_pub[side].publish(msg)

    def joint_group_command_to_msg(self, values: list):

        msg = Float64MultiArray()
        msg.data = values
        return msg
        # self._arm_vel_pub[side].publish(msg)

    def _joint_state_callback(self, msg : JointState):
        r"""
        Callback function for the joint state subscriber
        :param msg: JointState message
        :return: None
        """

        self._joint_states = msg
        self._robot.set_joint_states(self._joint_states.position)

    def _joystick_callback(self, msg : Joy):
        r"""
        Callback function for the joystick subscriber
        :param msg: Joy message
        :return: None
        """

        self._joy_msg = (msg.axes, msg.buttons)

    def move_to_neutral(self):
        r"""
        Move the robot to neutral position
        :return: None
        """

        # while not rospy.is_shutdown():
        self.send_traj_command('right', CONTROL_MODE.POSITION,
                               RIGHT_SAMPLE_JOINTSTATES, 1)
        self.send_traj_command('left', CONTROL_MODE.POSITION,
                               LEFT_SAMPLE_JOINTSTATES, 1)
        
        

    def start(self):

        rospy.wait_for_message('/joy', Joy)
        while not rospy.is_shutdown():

            if self._joy_msg[1][-3]:
                self.move_to_neutral()

            if  (self._joy_msg[1][4] * self._joy_msg[1][5]) and not self._constraint_is_set:
                self._constraint_is_set = self.set_kinematics_constraints()

            if self._constraint_is_set:
                
                qdot = np.zeros(14)
                qdot_l = np.zeros(7)
                qdot_r = np.zeros(7)

                if self._joy_msg[1][5]:
                    start_time = time.time()
                    twist, done = joy_to_twist(self._joy_msg, [0.1, 0.1])
                    jacob_left = self._robot.get_jacobian('left')
                    jacob_right = self._robot.get_jacobian('right')

                    qdot_l, qdot_r = duo_arm_qdot_constraint(
                        jacob_left, jacob_right, twist, activate_nullspace=True)

                    qdot = np.concatenate([qdot_r, qdot_l])

                    exec_time = time.time() - start_time
                    print('Execution time: ', exec_time)

                self._actual_qdot_left.append(reorder_values(self._joint_states.velocity[17:24]))
                self._desired_qdot_left.append(qdot_r)

                msg = self.joint_group_command_to_msg(qdot)
                self._joint_group_vel_pub.publish(msg)


            self._rate.sleep()

    def home(self):
        r"""
        Move the robot to home position
        :return: None
        """
        while not rospy.is_shutdown():
            self.move_to_neutral()
            self._rate.sleep()
        # self.move_to_neutral()

    def path_trakcing_test(self):

        updated_joined_left = self._robot.get_tool_pose('left')
        # rospy.wait_for_message('/joy', Joy)
        arrived = False
        home = False
        qdot = np.zeros(14)
        qdot_l = np.zeros(7)
        qdot_r = np.zeros(7)
        target = np.eye(4)

        while not rospy.is_shutdown() :

            if not self._constraint_is_set:
                self._constraint_is_set = self.set_kinematics_constraints()    
                pose = self._robot.get_virtual_pose()
                target = pose @ sm.SE3(0.2, 0, 0).A
                print('constraint is set')
                
            updated_joined_left = self._robot.get_tool_pose('left')
            middle_twist, arrived = rtb.p_servo(updated_joined_left,
                                                            target,
                                                            gain=0.1,
                                                            threshold=0.01,
                                                            method='angle-axis')  # Servoing in the virtual middle frame using angle-axis representation for angular error
            
            jacob_left = self._robot.get_jacobian('left')
            jacob_right = self._robot.get_jacobian('right')

            # Calculate the joint velocities using the Resolved Motion Rate Control (RMRC) method with the projection onto nullspace of Constraint Jacobian
            qdot_l, qdot_r = duo_arm_qdot_constraint(jacob_left, jacob_right, middle_twist, activate_nullspace=True)

            # Visualization of the frames
            updated_joined_left = self._robot.get_tool_pose('left')

            qdot = np.concatenate([qdot_r, qdot_l])
            msg = self.joint_group_command_to_msg(qdot)
            self._joint_group_vel_pub.publish(msg)
            
            self._actual_qdot_left.append(reorder_values(self._joint_states.velocity[17:24]))
            self._desired_qdot_left.append(qdot_r)

            self._actual_qdot_right.append(reorder_values(self._joint_states.velocity[31:38]))
            self._desired_qdot_right.append(qdot_l)

            if arrived:
                print('Arrived')
                break 

            self._rate.sleep()

    def _clean(self):

        self._robot.shutdown()
        print('Shutting down the robot')
        plot_joint_velocities(self._actual_qdot_left, self._desired_qdot_left)
        plot_joint_velocities(self._actual_qdot_right, self._desired_qdot_right)


if __name__ == "__main__":

    # Initialize the ROS node
    rospy.init_node('test_command', disable_signals=True)
    controller = PR2BiCoor()
    # controller.path_trakcing_test()
    # controller.home()
    controller.start()
