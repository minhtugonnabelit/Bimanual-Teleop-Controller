import numpy as np
from scipy import linalg
from enum import Enum

import spatialmath as sm
import matplotlib.pyplot as plt
import spatialgeometry as geometry
import roboticstoolbox as rtb
from swift import Swift

# Import custom utility functions
from utility import *

import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
import tf

# Initialize the ROS node
rospy.init_node('test_command')
pub = rospy.Publisher('r_arm_controller/command', JointTrajectory, queue_size=1 )
rate = rospy.Rate(10)
msg = JointTrajectory()


class CONTROL_MODE(Enum):
    POSITION = 0
    VELOCITY = 1
    ACCELERATION = 2
    EFFORT = 3

LEFT_JOINT_NAMES = [
    "l_shoulder_pan_joint",
    "l_shoulder_lift_joint",
    "l_upper_arm_roll_joint",
    "l_elbow_flex_joint",
    "l_forearm_roll_joint",
    "l_wrist_flex_joint",
    "l_wrist_roll_joint",
]

RIGHT_JOINT_NAMES = [
    "r_shoulder_pan_joint",
        "r_shoulder_lift_joint",
        "r_upper_arm_roll_joint"
        "r_forearm_roll_joint",
        "r_elbow_flex_joint",
        "r_wrist_flex_joint",
        "r_wrist_roll_joint",
]

class PR2BiCoor:
    
    def __init__(self):
        
        self.joint_states = None
        rospy.wait_for_message('/joint_states', JointState)
        self.joint_state_sub = rospy.Subscriber('/joint_states', JointState, self.joint_state_callback)

        self.PR2 = rtb.models.PR2()
        self.q[16:23] = self.joint_states[17:24]
        self.q[23:30] = self.joint_states[31:38]

        self._tf_listener = tf.TransformListener()

        env = Swift()
        env.set_camera_pose([1, 0, 1], [0, 0.5, 1])
        env.launch()

        pass

    def set_kinematics_constraints(self):

        left_pose = self._tf_listener.lookupTransform('l_gripper_tool_frame', 'base_link', rospy.Time(0))
        left_pose = self.posestamped_to_SE3(left_pose[0], left_pose[1])
        right_pose = self._tf_listener.lookupTransform('r_gripper_tool_frame', 'base_link', rospy.Time(0))
        right_pose = self.posestamped_to_SE3(right_pose[0], right_pose[1])

        virtual_pose = np.eye(4,4)
        virtual_pose[0:3, 3] = (left_pose.A[:3,-1]) + np.array(right_pose.A[:3,-1]) / 2
        

        joined_ax = geometry.Axes(length=0.05, pose=virtual_pose,)
        joined_in_left = linalg.inv(sm.SE3(left_pose)) @ virtual_pose
        joined_in_right = linalg.inv(sm.SE3(right_pose)) @ virtual_pose

    def posestamped_to_SE3(pose):

        T = sm.SE3()
        T[:3,-1] = np.array(pose[0])
        T[:3,:3] = tf.TransformerROS.fromTranslationRotation(pose[0], pose[1])

        return T

    def get_joint_names(self, side):
        if side == 'left':
            return LEFT_JOINT_NAMES
        elif side == 'right':
            return RIGHT_JOINT_NAMES
        else:
            return None
        
    def send_command(self, joint_names : list, control_mode, value, duration):
        
        r"""
        Send the command to the robot
        :param joint_names: list of joint names
        :param control_mode: control mode
        :param value: control value
        :param duration: duration of the command
        :return: None
        """
        msg.header.frame_id = 'torso_lift_link'
        msg.header.stamp = rospy.Time.now()
        msg.joint_names = joint_names
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
        pub.publish(msg)

    def joint_state_callback(self, msg):
        # print(data)
        self.joint_states = msg.position


                               


# # Set the initial pose of the end-effectors
# left_tip_pose = pr2.fkine(pr2.q, end=pr2.grippers[1], ).A
# right_tip_pose = pr2.fkine(pr2.q, end=pr2.grippers[0], ).A
# l2r = linalg.inv(sm.SE3(left_tip_pose)) @ right_tip_pose
# left_ax = geometry.Axes(length=0.05, pose=left_tip_pose, color=[1, 0, 0])
# right_ax = geometry.Axes(length=0.05, pose=right_tip_pose)

# # Extract the middle point between the two tools
# joined = np.eye(4,4)
# joined[0:3, 3] = (left_tip_pose[0:3, 3] + right_tip_pose[0:3, 3]) / 2
# joined_ax = geometry.Axes(length=0.05, pose=joined,)
# joined_in_left = linalg.inv(sm.SE3(left_tip_pose)) @ joined
# joined_in_right = linalg.inv(sm.SE3(right_tip_pose)) @ joined

env.add(pr2)
env.add(left_ax)
env.add(right_ax)

w_l = list()
w_r = list()
df = list()
dt = 0.015

joy = joy_init()
LIN_G = 0.02
ANG_G = 0.05

done  = False

while not done:

    # ---------------------------------------------------------------------------#
    # SECTION TO HANDLE THE JOYSTICK INPUT
    twist, done = joy_to_twist(joy, [LIN_G, ANG_G], done)
    print(cur_config)

    # ---------------------------------------------------------------------------#
    # SECTION TO PERFORMS TWIST TRANSFORMATION IN A RIGID BODY MOTION
    jacob_l = pr2.jacobe(cur_config, end=pr2.grippers[1], start="l_shoulder_pan_link", tool=sm.SE3(joined_in_left))  # Jacobian of the left arm within tool frame
    jacob_r = pr2.jacobe(cur_config, end=pr2.grippers[0], start="r_shoulder_pan_link", tool=sm.SE3(joined_in_right))  # Jacobian of the right arm within tool frame

    w_l.append(manipulability(jacob_l))
    w_r.append(manipulability(jacob_r))
    
    # Calculate the joint velocities using the Resolved Motion Rate Control (RMRC) method with the projection onto nullspace of Constraint Jacobian
    qdot_l, qdot_r = duo_arm_qdot_constraint(jacob_l, jacob_r, twist, activate_nullspace=True)

    # ---------------------------------------------------------------------------#
    # SECTION TO UPDATE VISUALIZATION AND RECORD THE NECESSARY DATA
    pr2.q[16:23] = pr2.q[16:23] + qdot_r * dt  # right arm
    pr2.q[23:30] = pr2.q[23:30] + qdot_l * dt   # left arm

    # Send the joint angles to the robot
    send_command(RIGHT_JOINT_NAMES, 1, qdot_r * dt, 0.05)
    send_command(LEFT_JOINT_NAMES, 1, qdot_l * dt, 0.05)

    # check if any of the joint angles reach the joint limits

    for i in range(16, 30):
        if pr2.q[i] > pr2.qlim[1, i] or pr2.q[i] < pr2.qlim[0, i]:
            print(f"Joint angle of {i} reach the joint limit")
            # done = True

    # Visualization of the frames
    updated_joined_left = pr2.fkine(pr2.q, end=pr2.grippers[1], ).A @ joined_in_left
    updated_joined_right = pr2.fkine(pr2.q, end=pr2.grippers[0], ).A @ joined_in_right
    left_ax.T = updated_joined_left
    right_ax.T = updated_joined_right

    # Record the distance between offset frames of each arm to  observe the drift of tracked frame
    dis = np.linalg.norm(updated_joined_left[0:3, 3] - updated_joined_right[0:3, 3])
    df.append(dis)

    env.step()


# Record and plot the distance between offset frames of each arm to  observe the drift of tracked frame
fig, ax = plt.subplots(2,2)
ax[0,0].plot(w_l, 'r', linewidth=1)
ax[0,0].plot(w_r, 'b', linewidth=1)
ax[0,0].set_title('Manipulability graph')
ax[0,0].set_xlabel('Time')
ax[0,0].set_ylabel('Manipulability')
ax[0,0].legend(['Left arm', 'Right arm'])

ax[0,1].plot(np.diff(w_l), 'r', linewidth=1)
ax[0,1].plot(np.diff(w_r), 'b', linewidth=1)
ax[0,1].set_title('wdot')
ax[0,1].set_xlabel('Time')
ax[0,1].set_ylabel('Manipulability rate')
ax[0,1].legend(['Left arm', 'Right arm'])

ax[1,1].plot(df, 'k', linewidth=1)
ax[1,1].set_title('Drift graph')
ax[1,1].set_xlabel('Time')
ax[1,1].set_ylabel('Distance')


plt.show()
# env.hold()

