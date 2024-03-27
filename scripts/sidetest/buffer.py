# import time
import numpy as np

import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState, Joy

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


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

        self._pub = rospy.Publisher('pr2_joint_group_vel_controller/command', Float64MultiArray, queue_size=1)  
        self._l_pos_pub = rospy.Publisher('l_arm_controller/command', JointTrajectory, queue_size=1)
        self._r_pos_pub = rospy.Publisher('r_arm_controller/command', JointTrajectory, queue_size=1)
        
        self._joint_states = None
        self._joint_states_sub = rospy.Subscriber('joint_states', JointState, self._joint_states_callback)

        self._joy = None    
        self._joy_sub = rospy.Subscriber('joy', Joy, self._joy_callback)

        self.rate = rospy.Rate(50)

    def _joint_states_callback(self, msg: JointState):
        self._joint_states = msg.position

    def _joy_callback(self, msg: Joy):
        self._joy = (msg.axes, msg.buttons)
    
    def _command_to_mg(self,  values: list,):

        msg = Float64MultiArray()
        msg.data = values if len(values) == 14 else np.zeros(14)

        return msg
    
    def move_to_neutral(self):

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

        self._l_pos_pub.publish(l_neutral)
        self._r_pos_pub.publish(r_neutral)
    
    def joy_to_qdot(self):
        
        rospy.wait_for_message('/joy', Joy)

        V = 0.5

        while not rospy.is_shutdown():
            
            trigger = self._joy[1][4]
            dir = self._joy[0][1] / np.abs(self._joy[0][1]) if np.abs(self._joy[0][1]) > 0.1 else 0

            qdot = [
                V * dir * self._joy[1][0],
                V * dir * self._joy[1][1],
                V * dir * self._joy[1][2],
                V * dir * self._joy[1][3],
                V * dir * self._joy[1][5],
                V * dir * self._joy[1][6],
                V * dir * self._joy[1][7],
            ]
            
            for _ in range(7):
                qdot.append(0) 

            qdot = trigger * qdot
            self._pub.publish(self._command_to_mg(qdot) )
            self.rate.sleep()


if __name__ == '__main__':
    rospy.init_node('pr2_vel_control')
    pr2 = PR2VelControl()
    pr2.joy_to_qdot()