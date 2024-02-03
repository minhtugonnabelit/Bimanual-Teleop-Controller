import rospy
import tf
import spatialmath.base as smb
import numpy as np
from geometry_msgs.msg import PoseStamped, TwistStamped

class HydraReader:

    def __init__(self):
        rospy.init_node('hydra_reader', anonymous=True)
        self.listener = tf.TransformListener()
        self.pose_pub = rospy.Publisher('/hydra_pose', PoseStamped, queue_size=1)
        self.twist_pub = rospy.Publisher('/hydra_twist', TwistStamped, queue_size=1)
        self.rate = rospy.Rate(100)

    def pose_callback(self):

        pass


    def twist_conversion(self):
        
        pass

