# import time
import numpy as np

import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState, Joy

class PR2VelControl():

    def __init__(self):

        self._pub = rospy.Publisher('pr2_joint_group_vel_controller/command', Float64MultiArray, queue_size=1)  
        
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