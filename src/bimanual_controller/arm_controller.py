import tf
import rospy
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from pr2_controllers_msgs.msg import Pr2GripperCommand, JointControllerState
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
                 controller_cmd_type,
                 gripper_cmd_type,
                 robot_base_frame: str = 'base_link'):

        self.name = arm
        self.joint_names = arm_group_joint_names
        self.gripper_cmd_type = gripper_cmd_type
        self.robot_base_frame = robot_base_frame
        joint_controller_name = arm_group_controller_name + "/command"
        gripper_controller_name = arm + '_gripper_controller/command'

        self.joint_controller_pub = rospy.Publisher(
            joint_controller_name, controller_cmd_type, queue_size=10)
        self.gripper_controller_pub = rospy.Publisher(
            gripper_controller_name, gripper_cmd_type, queue_size=1) 
        
        # Initialize the transform listener
        self._tf_listener = tf.TransformListener()
        self._tf_broadcaster = tf.TransformBroadcaster()

        self._gripper_state = None
        self.gripper_state_sub = rospy.Subscriber("/" + arm + '_gripper_controller/state', JointControllerState, self.gripper_state_callback)

    def get_arm_name(self):
        return self.name
    
    def get_arm_joint_names(self):
        return self.joint_names
    
    def get_gripper_transform(self):
        pose_in_base_frame = self._tf_listener.lookupTransform(
            self.robot_base_frame,
            self.name + '_gripper_tool_frame',
            rospy.Time(0))

        pose = tf.TransformerROS.fromTranslationRotation(
            tf.TransformerROS,
            translation=pose_in_base_frame[0],
            rotation=pose_in_base_frame[1])
        
        return pose

    def send_joint_command(self, joint_command):
        joint_command_msg = Float64MultiArray()
        joint_command_msg.data = joint_command
        self.joint_controller_pub.publish(joint_command_msg)

    def open_gripper(self):
        msg = self.gripper_cmd_type()
        msg.position = deepcopy(self._gripper_state) + 0.005
        msg.max_effort = 100.0
        self.gripper_controller_pub.publish(msg)

    def close_gripper(self):
        msg = self.gripper_cmd_type()
        msg.position = deepcopy(self._gripper_state) - 0.005
        msg.max_effort = 100.0
        self.gripper_controller_pub.publish(msg)

    def gripper_state_callback(self, data : JointControllerState):
        self._gripper_state = data.process_value

    def move_to_neutral(self):

        client = actionlib.SimpleActionClient(
            '/' + self.name + '_arm_controller/follow_joint_trajectory',
            FollowJointTrajectoryAction)
        client.wait_for_server()
        goal = FollowJointTrajectoryGoal()
        goal.trajectory = ArmController._create_joint_traj_msg(
            self.joint_names,
            3,
            q=SAMPLE_STATES[self.name])
        
        client.send_goal(goal)
        client.wait_for_result()
    
    @ staticmethod
    def _create_joint_traj_msg(joint_names: list, dt: float, joint_states: list = None, qdot: list = None, q: list = None):
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