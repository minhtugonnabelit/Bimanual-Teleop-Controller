from typing import Union

import tf
import rospy
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from std_msgs.msg import Float64MultiArray
from pr2_controllers_msgs.msg import  JointControllerState, Pr2GripperCommand
from copy import deepcopy

from bimanual_controller.utility import *


class ArmController:
    r"""
    Class to control the arm of the robot
    """
        
    def __init__(self,
                 arm: str,
                 arm_group_joint_names: list,
                 arm_group_controller_name: str,
                 controller_cmd_type: Union[Float64MultiArray,],
                 gripper_cmd_type : Union[Pr2GripperCommand,],
                 robot_base_frame: str = 'base_link'):

        self._name = arm
        self._joint_names = arm_group_joint_names
        self._gripper_cmd_type = gripper_cmd_type
        self._robot_base_frame = robot_base_frame
        joint_controller_name = arm_group_controller_name + "/command"
        gripper_controller_name = arm + '_gripper_controller/command'

        self._joint_controller_pub = rospy.Publisher(
            joint_controller_name, controller_cmd_type, queue_size=10)
        self._gripper_controller_pub = rospy.Publisher(
            gripper_controller_name, gripper_cmd_type, queue_size=1) 
        self._client = actionlib.SimpleActionClient(
            '/' + arm + '_arm_controller/follow_joint_trajectory',
            FollowJointTrajectoryAction)
        self._client.wait_for_server()
        
        self._tf_listener = tf.TransformListener()
        self._tf_broadcaster = tf.TransformBroadcaster()

        self._gripper_state = None
        self._gripper_state_sub = rospy.Subscriber("/" + arm + '_gripper_controller/state', JointControllerState, self.gripper_state_callback)

    def get_arm_name(self):
        return self._name
    
    def get_arm_joint_names(self):
        return self._joint_names
    
    def get_gripper_transform(self):
        pose_in_base_frame = self._tf_listener.lookupTransform(
            self._robot_base_frame,
            self._name + '_gripper_tool_frame',
            rospy.Time(0))

        pose = tf.TransformerROS.fromTranslationRotation(
            tf.TransformerROS,
            translation=pose_in_base_frame[0],
            rotation=pose_in_base_frame[1])
        
        return pose

    def send_joint_command(self, joint_command):
        joint_command_msg = Float64MultiArray()
        joint_command_msg.data = joint_command
        self._joint_controller_pub.publish(joint_command_msg)

    def open_gripper(self):
        msg = self._gripper_cmd_type()
        msg.position = deepcopy(self._gripper_state) + 0.005
        msg.max_effort = 200.0
        self._gripper_controller_pub.publish(msg)

    def close_gripper(self):
        msg = self._gripper_cmd_type()
        msg.position = deepcopy(self._gripper_state) - 0.005
        msg.max_effort = 200.0
        self._gripper_controller_pub.publish(msg)

    def gripper_state_callback(self, data : JointControllerState):
        self._gripper_state = data.process_value

    def move_to_neutral(self, state):
        goal = FollowJointTrajectoryGoal()
        goal.trajectory = ROSUtils.create_joint_traj_msg(
            self._joint_names,
            2,
            self._robot_base_frame,
            q=state)
        
        self._client.send_goal(goal)
        return self._client.wait_for_result()
    
