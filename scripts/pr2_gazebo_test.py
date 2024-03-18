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
from sensor_msgs.msg import JointState
import tf


LEFT_SAMPLE_JOINTSTATES = [ np.pi/6, 
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


# class FakePR2:

#     r"""
#     ### Class to simulate PR2 on Swift environment

#     This will initialize the Swift environment with robot model without any ROS components. 
#     This model will be fed with joint states provided and update the visualization of the virtual frame . """

#     def __init__(self) -> None:

#         self._robot = rtb.models.PR2()
#         self._robot.q = np.zeros(31)

#         self._constraints_is_set = False
#         self.init_visualization()

#         pass

#     def timeline(self):
#         r"""
#         Timeline function to update the visualization
#         :return: None
#         """

#         self._env.step()

#     def set_constraints(self, virtual_pose: np.ndarray):
#         r"""
#         Set the kinematics constraints for the robot
#         :param virtual_pose: virtual pose on robot base frame
#         :return: None
#         """

#         self._virtual_pose = virtual_pose
#         self._joined_in_left = linalg.inv(sm.SE3(self._robot.fkine(
#             self._robot.q, end=self._robot.grippers[1], ).A)) @ virtual_pose
#         self._joined_in_right = linalg.inv(sm.SE3(self._robot.fkine(
#             self._robot.q, end=self._robot.grippers[0], ).A)) @ virtual_pose

#         self._constraints_is_set = True

#         return True
#         # pass

#     def set_joint_states(self, joint_states: list):
#         r"""
#         Set the joint states of the arms only
#         :param joint_states: list of joint states
#         :return: None
#         """
#         print(joint_states)

#         self._robot.q[16:23] = joint_states[17:24]
#         self._robot.q[23:30] = joint_states[31:38]

#         left_constraint = np.eye(4, 4)
#         right_constraint = np.eye(4, 4)
#         if self._constraints_is_set:    # If the constraints are set, then update the virtual frame from the middle point between the two end-effectors
#             left_constraint = self._joined_in_left
#             right_constraint = self._joined_in_right

#         self._left_ax.T = self._robot.fkine(
#             self._robot.q, end=self._robot.grippers[1], ).A @ left_constraint
#         self._right_ax.T = self._robot.fkine(
#             self._robot.q, end=self._robot.grippers[0], ).A @ right_constraint

#         # self._env.step()

#     def init_visualization(self):
#         r"""
#         Initialize the visualization of the robot

#         :return: None
#         """

#         self._env = Swift()
#         self._env.set_camera_pose([1, 0, 1], [0, 0.5, 1])
#         self._env.launch()

#         if not self._constraints_is_set:    # If the constraints are not set, then visualize the virtual frame from each arm end-effector
#             self._left_ax = geometry.Axes(length=0.05, pose=self._robot.fkine(
#                 self._robot.q, end=self._robot.grippers[1], ).A)
#             self._right_ax = geometry.Axes(length=0.05, pose=self._robot.fkine(
#                 self._robot.q, end=self._robot.grippers[0], ).A)
#         else:                           # If the constraints are set, then visualize the virtual frame from the middle point between the two end-effectors
#             self._left_ax = geometry.Axes(length=0.05, pose=self._robot.fkine(
#                 self._robot.q, end=self._robot.grippers[1], ).A @ self._joined_in_left)
#             self._right_ax = geometry.Axes(length=0.05, pose=self._robot.fkine(
#                 self._robot.q, end=self._robot.grippers[0], ).A @ self._joined_in_right)

#         self._env.add(self._robot)
#         self._env.add(self._left_ax)
#         self._env.add(self._right_ax)

#         self.thread = threading.Thread(target=self.timeline)
#         self.thread.start()

#     def get_jacobian(self, side, tool=None):
#         r"""
#         Get the Jacobian of the robot
#         :param side: side of the robot
#         :param tool: tool frame

#         :return: Jacobian
#         """

#         if side == 'left':
#             return self._robot.jacobe(self._robot.q, end=self._robot.grippers[1], start="l_shoulder_pan_link", tool=tool)
#         elif side == 'right':
#             return self._robot.jacobe(self._robot.q, end=self._robot.grippers[0], start="r_shoulder_pan_link", tool=tool)
#         else:
#             return None

#     def q(self):
#         r"""
#         Get the joint states of the robot
#         :return: joint states
#         """

#         self.thread.join()

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
        self.robot = FakePR2()

        # Initialize the joint states subscriber
        self.joint_states = None
        rospy.wait_for_message('/joint_states', JointState)
        self._joint_state_sub = rospy.Subscriber(
            '/joint_states', JointState, self._joint_state_callback)

        # Initialize arms controllers publishers
        left_arm_pub = rospy.Publisher(
            'l_arm_controller/command', JointTrajectory, queue_size=1)
        right_arm_pub = rospy.Publisher(
            'r_arm_controller/command', JointTrajectory, queue_size=1)

        self._arm_control_pub = {
            'left': left_arm_pub,
            'right': right_arm_pub
        }

        self._rate = rospy.Rate(10)

        # Initialize the transform listener
        self._tf_listener = tf.TransformListener()
        rospy.on_shutdown(self.clean)


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

    def send_command(self, side, control_mode, value, duration):
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

    def _joint_state_callback(self, msg):
        r"""
        Callback function for the joint state subscriber
        :param msg: JointState message
        :return: None
        """

        self.joint_states = msg.position
        self.robot.set_joint_states(self.joint_states)

    def _joystick_callback(self, msg):
        r"""
        Callback function for the joystick subscriber
        :param msg: Joy message
        :return: None
        """

        self.joy_msg = msg

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
            self.send_command('right', CONTROL_MODE.POSITION, RIGHT_SAMPLE_JOINTSTATES, 3)
            self.send_command('left', CONTROL_MODE.POSITION, LEFT_SAMPLE_JOINTSTATES, 3)
            self._rate.sleep()

    def bimanual_controller(self):
        r"""
        Bimanual controller
        :return: None
        """



        pass

    def clean(self):

        self.robot.q()



if __name__ == "__main__":

    # Initialize the ROS node
    rospy.init_node('test_command')
    controller = PR2BiCoor()
    controller.move_to_neutral()


# # # Set the initial pose of the end-effectors
# # left_tip_pose = pr2.fkine(pr2.q, end=pr2.grippers[1], ).A
# # right_tip_pose = pr2.fkine(pr2.q, end=pr2.grippers[0], ).A
# # l2r = linalg.inv(sm.SE3(left_tip_pose)) @ right_tip_pose
# # left_ax = geometry.Axes(length=0.05, pose=left_tip_pose, color=[1, 0, 0])
# # right_ax = geometry.Axes(length=0.05, pose=right_tip_pose)
# # # Extract the middle point between the two tools
# # joined = np.eye(4,4)
# # joined[0:3, 3] = (left_tip_pose[0:3, 3] + right_tip_pose[0:3, 3]) / 2
# # joined_ax = geometry.Axes(length=0.05, pose=joined,)
# # joined_in_left = linalg.inv(sm.SE3(left_tip_pose)) @ joined
# # joined_in_right = linalg.inv(sm.SE3(right_tip_pose)) @ joined

# w_l = list()
# w_r = list()
# df = list()
# dt = 0.015

# joy = joy_init()
# LIN_G = 0.02
# ANG_G = 0.05

# done = False

# while not done:

#     # ---------------------------------------------------------------------------#
#     # SECTION TO HANDLE THE JOYSTICK INPUT
#     twist, done = joy_to_twist(joy, [LIN_G, ANG_G], done)
#     print(cur_config)

#     # ---------------------------------------------------------------------------#
#     # SECTION TO PERFORMS TWIST TRANSFORMATION IN A RIGID BODY MOTION
#     jacob_l = pr2.jacobe(cur_config, end=pr2.grippers[1], start="l_shoulder_pan_link", tool=sm.SE3(
#         joined_in_left))  # Jacobian of the left arm within tool frame
#     jacob_r = pr2.jacobe(cur_config, end=pr2.grippers[0], start="r_shoulder_pan_link", tool=sm.SE3(
#         joined_in_right))  # Jacobian of the right arm within tool frame

#     w_l.append(manipulability(jacob_l))
#     w_r.append(manipulability(jacob_r))

#     # Calculate the joint velocities using the Resolved Motion Rate Control (RMRC) method with the projection onto nullspace of Constraint Jacobian
#     qdot_l, qdot_r = duo_arm_qdot_constraint(
#         jacob_l, jacob_r, twist, activate_nullspace=True)

#     # ---------------------------------------------------------------------------#
#     # SECTION TO UPDATE VISUALIZATION AND RECORD THE NECESSARY DATA
#     pr2.q[16:23] = pr2.q[16:23] + qdot_r * dt  # right arm
#     pr2.q[23:30] = pr2.q[23:30] + qdot_l * dt   # left arm

#     # Send the joint angles to the robot
#     send_command(RIGHT_JOINT_NAMES, 1, qdot_r * dt, 0.05)
#     send_command(LEFT_JOINT_NAMES, 1, qdot_l * dt, 0.05)

#     # check if any of the joint angles reach the joint limits

#     for i in range(16, 30):
#         if pr2.q[i] > pr2.qlim[1, i] or pr2.q[i] < pr2.qlim[0, i]:
#             print(f"Joint angle of {i} reach the joint limit")
#             # done = True

#     # Visualization of the frames
#     updated_joined_left = pr2.fkine(
#         pr2.q, end=pr2.grippers[1], ).A @ joined_in_left
#     updated_joined_right = pr2.fkine(
#         pr2.q, end=pr2.grippers[0], ).A @ joined_in_right
#     left_ax.T = updated_joined_left
#     right_ax.T = updated_joined_right

#     # Record the distance between offset frames of each arm to  observe the drift of tracked frame
#     dis = np.linalg.norm(
#         updated_joined_left[0:3, 3] - updated_joined_right[0:3, 3])
#     df.append(dis)

#     env.step()


# # Record and plot the distance between offset frames of each arm to  observe the drift of tracked frame
# fig, ax = plt.subplots(2, 2)
# ax[0, 0].plot(w_l, 'r', linewidth=1)
# ax[0, 0].plot(w_r, 'b', linewidth=1)
# ax[0, 0].set_title('Manipulability graph')
# ax[0, 0].set_xlabel('Time')
# ax[0, 0].set_ylabel('Manipulability')
# ax[0, 0].legend(['Left arm', 'Right arm'])

# ax[0, 1].plot(np.diff(w_l), 'r', linewidth=1)
# ax[0, 1].plot(np.diff(w_r), 'b', linewidth=1)
# ax[0, 1].set_title('wdot')
# ax[0, 1].set_xlabel('Time')
# ax[0, 1].set_ylabel('Manipulability rate')
# ax[0, 1].legend(['Left arm', 'Right arm'])

# ax[1, 1].plot(df, 'k', linewidth=1)
# ax[1, 1].set_title('Drift graph')
# ax[1, 1].set_xlabel('Time')
# ax[1, 1].set_ylabel('Distance')


# plt.show()
# # env.hold()
