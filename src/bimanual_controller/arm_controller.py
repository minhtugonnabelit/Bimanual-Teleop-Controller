
import rospy
from std_msgs.msg import Float64MultiArray
from pr2_controllers_msgs.msg import Pr2GripperCommand, JointControllerState
from copy import deepcopy

class ArmController:
    r"""
    Class to control the arm of the robot
    """
        
    def __init__(self,
                 arm: str,
                 arm_group_joint_names: list,
                 arm_group_controller_name: str,
                 controller_cmd_type,
                 enable_gripper: bool = False,
                 gripper_cmd_type=None
                 ):

        self.name = arm
        self.joint_names = arm_group_joint_names
        self.gripper_cmd_type = gripper_cmd_type
        self.joint_controller_name = arm_group_controller_name + "/command"

        self.joint_controller_pub = rospy.Publisher(
            self.joint_controller_name, controller_cmd_type, queue_size=10)
        self.gripper_controller_pub = rospy.Publisher(
            "/" + arm + '_gripper_controller/command', Pr2GripperCommand, queue_size=1) 
        
        self._state = None
        self.gripper_state_sub = rospy.Subscriber("/" + arm + '_gripper_controller/state', JointControllerState, self.gripper_state_callback)

    def get_arm_name(self):
        return self.name
    
    def get_arm_joint_names(self):
        return self.joint_names
    
    def get_joint_controller_name(self):
        return self.joint_controller_name

    def send_joint_command(self, joint_command):
        joint_command_msg = Float64MultiArray()
        joint_command_msg.data = joint_command
        self.joint_controller_pub.publish(joint_command_msg)

    def open_gripper(self):
        r"""
        open the gripper
        :param side: side of the robot
        :return: None
        """

        msg = Pr2GripperCommand()
        msg.position = deepcopy(self._state) + 0.005
        msg.max_effort = 50.0
        rospy.loginfo("Opening Gripper")
        self.gripper_controller_pub.publish(msg)

    def close_gripper(self):
        r"""
        Close the gripper
        :param side: side of the robot
        :return: None
        """

        msg = Pr2GripperCommand()
        msg.position = deepcopy(self._state) - 0.005
        msg.max_effort = 50.0
        rospy.loginfo("Closing Gripper")
        self.gripper_controller_pub.publish(msg)

    def gripper_state_callback(self, data : JointControllerState):

        self._state = data.process_value