import rospy
import tf
import spatialmath.base as smb
import numpy as np
from geometry_msgs.msg import TwistStamped

class HydraReader():

    def __init__(self):

        self._tf_listener = tf.TransformListener()
        self._right_twist_pub = rospy.Publisher("/r_arm_servo_server/delta_twist_cmds", TwistStamped, queue_size=1)
        self._first_time = True
        self._twiststamped_msg = TwistStamped()


    def twist_to_twist_stamped_msg(self, frame_id="/hydra_base"):
        """
        Get the twist of the hydra right pivot in the hydra base frame.\n
        The twist is computed as the difference between the current pose and the previous pose divided by the time difference.\n
        
        This is done by a powerful method of lookupTwist from tf.TransformListener.\n
        The method lookupTwist returns a tuple of 6 elements, which are the linear and angular velocities.\n
        
        
        """

         # @TODO figuring out how to track the time difference between the current pose and the previous pose that will be used as buffer time for the lookupTwist method.
        
        twist = self._tf_listener.lookupTwist("/hydra_right_pivot", frame_id, rospy.Time() , rospy.Duration(0.1))
        self._twiststamped_msg.header.stamp = rospy.Time.now()
        self._twiststamped_msg.header.frame_id = frame_id
        self._twiststamped_msg.twist.linear.x = twist[0][0]
        self._twiststamped_msg.twist.linear.y = twist[0][1]
        self._twiststamped_msg.twist.linear.z = twist[0][2]

        self._twiststamped_msg.twist.angular.x = twist[1][0]
        self._twiststamped_msg.twist.angular.y = twist[1][1]
        self._twiststamped_msg.twist.angular.z = twist[1][2]

        return self._twiststamped_msg

    def run(self):

        rate = rospy.Rate(5)

        while not rospy.is_shutdown():


            self._tf_listener.waitForTransform("/hydra_right_pivot", "/hydra_base", rospy.Time(), rospy.Duration(12))
            twist = self.twist_to_twist_stamped_msg()

            print(twist)

            self._right_twist_pub.publish(twist)

            rate.sleep()
    
if __name__ == "__main__":
        
    rospy.init_node("hydra_reader")

    hydra_reader = HydraReader()
    hydra_reader.run()
        



