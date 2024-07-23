# import time
import numpy as np

import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState, Joy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from pr2_controllers_msgs.msg import Pr2GripperCommand
from pr2_mechanism_msgs.srv import SwitchController

# from fakePR2 import FakePR2
from bimanual_teleop_controller.utility import *
from bimanual_teleop_controller.fake_pr2 import FakePR2

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

class PR2VelControl():

    def __init__(self, rate = 5):

        self._fake_pr2 = FakePR2(launch_visualizer=False)  

        # self._pub = rospy.Publisher('pr2_joint_group_vel_controller/command', Float64MultiArray, queue_size=1)  
        self._pub = {
            'left': rospy.Publisher('/l_arm_joint_group_velocity_controller/command', Float64MultiArray, queue_size=1) ,
            'right': rospy.Publisher('/r_arm_joint_group_velocity_controller/command', Float64MultiArray, queue_size=1)  
        }
        self._l_pos_pub = rospy.Publisher('l_arm_controller/command', JointTrajectory, queue_size=1)
        self._r_pos_pub = rospy.Publisher('r_arm_controller/command', JointTrajectory, queue_size=1)

        
        self._joint_states = None
        self._joint_states_sub = rospy.Subscriber('joint_states', JointState, self._joint_states_callback)

        self._joy = None    
        self._joy_sub = rospy.Subscriber('joy', Joy, self._joy_callback)

        self.rate = rospy.Rate(rate)
        self._dt = 1/rate

        rospy.on_shutdown(self.clearnup)

    def _joint_states_callback(self, msg: JointState):
        self._joint_states = msg.position
        self._fake_pr2.set_joint_states(self._joint_states)

    def _joy_callback(self, msg: Joy):
        self._joy = (msg.axes, msg.buttons)
    
    def _command_to_mg(self,  values: list,):

        msg = Float64MultiArray()
        msg.data = values 

        return msg
    
    def joint_traj_to_msg(self, values: list):

        if (values == np.zeros(14)).all():
            return

        r_joint_traj = JointTrajectory()
        # r_joint_traj.header.stamp = rospy.Time.now()
        r_joint_traj.header.frame_id = "torso_lit_link"
        r_joint_traj.joint_names = [
            "r_shoulder_pan_joint",
            "r_shoulder_lift_joint",
            "r_upper_arm_roll_joint",
            "r_elbow_flex_joint",
            "r_forearm_roll_joint",
            "r_wrist_flex_joint",
            "r_wrist_roll_joint",
        ]

        r_traj_point = JointTrajectoryPoint()
        r_traj_point.positions = reorder_values(self._joint_states[17:24]) + values[:7] * self._dt
        r_traj_point.velocities = values[:7]
        r_traj_point.time_from_start = rospy.Duration(self._dt)
        r_joint_traj.points = [r_traj_point]    

        l_joint_traj = JointTrajectory()
        # l_joint_traj.header.stamp = rospy.Time.now()
        l_joint_traj.header.frame_id = "torso_lit_link"
        l_joint_traj.joint_names = [
            "l_shoulder_pan_joint",
            "l_shoulder_lift_joint",
            "l_upper_arm_roll_joint",
            "l_elbow_flex_joint",
            "l_forearm_roll_joint",
            "l_wrist_flex_joint",
            "l_wrist_roll_joint",
        ]

        l_traj_point = JointTrajectoryPoint()
        l_traj_point.positions = reorder_values(self._joint_states[31:38]) + values[7:] * self._dt
        l_traj_point.velocities = values[7:]
        l_traj_point.time_from_start = rospy.Duration(self._dt)
        l_joint_traj.points = [r_traj_point]    

        self._r_pos_pub.publish(r_joint_traj)
        self._l_pos_pub.publish(l_joint_traj)
    
    
    def left_neutral(self):

        l_neutral = JointTrajectory()
        l_neutral.joint_names = [
            "l_shoulder_pan_joint",
            "l_shoulder_lift_joint",
            "l_upper_arm_roll_joint",
            "l_elbow_flex_joint",
            "l_forearm_roll_joint",
            "l_wrist_flex_joint",
            "l_wrist_roll_joint",
        ]

        l_neutral_point = JointTrajectoryPoint()
        l_neutral_point.positions = LEFT_SAMPLE_JOINTSTATES
        l_neutral_point.time_from_start = rospy.Duration(1)
        l_neutral.points = [l_neutral_point]

        self._l_pos_pub.publish(l_neutral)

    def right_neutral(self):

        r_neutral = JointTrajectory()
        r_neutral.joint_names = [
            "r_shoulder_pan_joint",
            "r_shoulder_lift_joint",
            "r_upper_arm_roll_joint",
            "r_elbow_flex_joint",
            "r_forearm_roll_joint",
            "r_wrist_flex_joint",
            "r_wrist_roll_joint",
        ]

        r_neutral_point = JointTrajectoryPoint()
        r_neutral_point.positions = RIGHT_SAMPLE_JOINTSTATES
        r_neutral_point.time_from_start = rospy.Duration(1)
        r_neutral.points = [r_neutral_point]

        self._r_pos_pub.publish(r_neutral)
    
    
    def joy_to_qdot(self):
        
        rospy.wait_for_message('/joy', Joy)

        V = 0.3
        qdot = np.zeros(14)

        while not rospy.is_shutdown():
            
            qdot = np.zeros(14)

            dir = self._joy[0][1] / np.abs(self._joy[0][1]) if np.abs(self._joy[0][1]) > 0.4 else 0
            if self._joy[1][4]:
                
                for i in range(7):
                    qdot[i+7] = V * dir * self._joy[1][i]


            if self._joy[1][5]:

                for i in range(7):
                    qdot[i] = V * dir * self._joy[1][i]


            self._pub['right'].publish(self._command_to_mg(qdot[:7])) 
            self._pub['left'].publish(self._command_to_mg(qdot[7:])) 

            self.rate.sleep()

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

    @staticmethod
    def _kill_jg_vel_controller():
        r"""
        Switch the controllers
        :return: bool value of the service call
        """

        switched = PR2VelControl._call('pr2_controller_manager/switch_controller',
                                   SwitchController,
                                   start_controllers=[
                                       'l_arm_controller', 'r_arm_controller'],
                                   stop_controllers=[
                                       'pr2_joint_group_vel_controller'],
                                   strictness=1)

        return switched

    def clearnup(self):
        self._fake_pr2.shutdown()
        # self._kill_jg_vel_controller()


if __name__ == '__main__':
    rospy.init_node('pr2_vel_control')
    pr2 = PR2VelControl()
    pr2.joy_to_qdot()