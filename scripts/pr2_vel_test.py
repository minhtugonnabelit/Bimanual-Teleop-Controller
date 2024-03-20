# import time
import numpy as np

import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState, Joy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from fakePR2 import FakePR2
from utility import *

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

    def __init__(self):

        self._fake_pr2 = FakePR2()  

        self._pub = rospy.Publisher('pr2_joint_group_vel_controller/command', Float64MultiArray, queue_size=1)  
        self._l_pos_pub = rospy.Publisher('l_arm_controller/command', JointTrajectory, queue_size=1)
        self._r_pos_pub = rospy.Publisher('r_arm_controller/command', JointTrajectory, queue_size=1)

        
        self._joint_states = None
        self._joint_states_sub = rospy.Subscriber('joint_states', JointState, self._joint_states_callback)

        self._joy = None    
        self._joy_sub = rospy.Subscriber('joy', Joy, self._joy_callback)

        self.rate = rospy.Rate(50)

        rospy.on_shutdown(self._fake_pr2.shutdown)

    def _joint_states_callback(self, msg: JointState):
        self._joint_states = msg.position
        self._fake_pr2.set_joint_states(self._joint_states)

    def _joy_callback(self, msg: Joy):
        self._joy = (msg.axes, msg.buttons)
    
    def _command_to_mg(self,  values: list,):

        msg = Float64MultiArray()
        msg.data = values 

        return msg
    
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

        V = 0.6
        qdot = np.zeros(14)

        while not rospy.is_shutdown():
            
                
            dir = self._joy[0][1] / np.abs(self._joy[0][1]) if np.abs(self._joy[0][1]) > 0.1 else 0
            if self._joy[1][4]:
                
                if self._joy[0][-1]: 
                    self.left_neutral()
                else:
                    for i in range(7):
                        qdot[i+7] = V * dir * self._joy[1][i]


            if self._joy[1][5]:

                if self._joy[0][-1]: 
                    self.right_neutral()
                else:
                    for i in range(7):
                        qdot[i] = V * dir * self._joy[1][i]

            if not (self._joy[1][4] + self._joy[1][5]): qdot = np.zeros(14)


            msg = self._command_to_mg(qdot) 
            self._pub.publish(msg)

            self.rate.sleep()


if __name__ == '__main__':
    rospy.init_node('pr2_vel_control')
    pr2 = PR2VelControl()
    pr2.joy_to_qdot()