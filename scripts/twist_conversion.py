import rospy
import tf
from geometry_msgs.msg import PoseStamped, TwistStamped
# from razer_hydra import hydra_raw, hydra_calib

class TwistConversion:
    
    """
    This class is used to perform conversion from pose in discrete time to twist in continuous time.\n
    The pose is obtained from the hydra_reader node, and the twist is published to the servo_server topic for twist control.\n
    """
    
    def __init__(self):
        self.listener = tf.TransformListener()
        self.raw_sub = rospy.Subscriber('/hydra_raw', hydra_raw, self.raw_callback)
        # self.calib_sub = rospy.Subscriber('/hydra_calib', hydra_calib, self.calib_callback)


        self.pose_pub = rospy.Publisher('/hydra_pose', PoseStamped, queue_size=1)
        self.twist_pub = rospy.Publisher('/hydra_twist', TwistStamped, queue_size=1)
        self.rate = rospy.Rate(100)

    def raw_callback(self, msg):
        """
        This function is used to get the raw pose from the hydra_driver node.\n
        """
        rospy.loginfo("Raw Pose: %s", msg)
        pass
    
    def calib_callback(self, msg):
        """
        This function is used to get the calibrated pose from the hydra_driver node.\n
        """
        rospy.loginfo("Calibrated Pose: %s", msg)
        pass    

    def twist_conversion(self, raw : bool):
        """
        This function is used to convert the pose to twist.\n
        - raw: If True, the raw pose is used. If False, the pose is filtered.\n
        """

        pass



def main (args=None):
    """
    This function is used to run the twist_conversion node.\n
    """
    rospy.init_node('twist_conversion', anonymous=True)
    twist_conversion = TwistConversion()
    while not rospy.is_shutdown():
        twist_conversion.twist_conversion(raw=True)
        twist_conversion.rate.sleep()

    pass

if __name__ == '__main__':
    main(args=None)