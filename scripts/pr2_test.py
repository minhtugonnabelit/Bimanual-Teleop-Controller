import numpy as np
import roboticstoolbox as rtb
import matplotlib.pyplot as plt
import spatialmath as sm
import spatialmath.base as smb
from scipy import linalg

import rospy
from geometry_msgs.msg import Pose, Twist, TwistStamped
from sensor_msgs.msg import Joy, JointState
from tf2_geometry_msgs import PoseStamped

from utility import *

class PR2():

    def __init__(self) -> None:
        
        passcd 

    def _joy_callback(self, msg):

        pass

    def _handle_arm_switch(self):

        pass

    def _joint_state_callback(self, msg):

        pass

    def _joint_vel_pub(self, vel):

        pass

smb.trexp()

if __name__ == "__main__":
    pr2 = PR2()
    rospy.init_node('pr2_test', anonymous=True)
    rospy.spin()