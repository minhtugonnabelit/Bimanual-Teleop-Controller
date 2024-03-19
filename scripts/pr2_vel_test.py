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
    
    def command_to_mg(self,  values: list,):

        msg = Float64MultiArray()
        msg.data = values 

        return msg
    
    def _joy_to_joint_velocities(self):
        
        rospy.wait_for_message('/joy', Joy)

        V = 0.6
        qdot = np.zeros(14)

        while not rospy.is_shutdown():
            
                
            dir = self._joy[0][1] / np.abs(self._joy[0][1]) if np.abs(self._joy[0][1]) > 0.1 else 0
            if self._joy[1][4]:

                for i in range(7):
                    qdot[i+7] = V * dir * self._joy[1][i]

                # qdot[7] = V*dir*self._joy[1][0]

            if self._joy[1][5]:


                for i in range(7):
                    qdot[i] = V * dir * self._joy[1][i]

            if not (self._joy[1][4] + self._joy[1][5]): qdot = np.zeros(14)

            msg = self.command_to_mg(qdot) 
            self._pub.publish(msg)

            self.rate.sleep()


if __name__ == '__main__':
    rospy.init_node('pr2_vel_control')
    pr2 = PR2VelControl()
    pr2._joy_to_joint_velocities()