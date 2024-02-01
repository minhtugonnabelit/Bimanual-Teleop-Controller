import rospy
import tf
import time
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Joy

import copy 

class HydraReader():

    def __init__(self):

        self._last_switch_time = 0
        self._switch_debounce_interval = 1 # 300ms delay between switching arms

        self._tf_listener = tf.TransformListener()
        self._joysub = rospy.Subscriber("/hydra_joy", Joy, self._joy_callback)

        self._right_twist_pub = rospy.Publisher("/r_arm_servo_server/delta_twist_cmds", TwistStamped, queue_size=1)
        self._left_twist_pub = rospy.Publisher("/l_arm_servo_server/delta_twist_cmds", TwistStamped, queue_size=1)

        self._current_twist_pub = self._right_twist_pub
        self._switched = False
        self._twiststamped_msg = TwistStamped()
        self._linear_gain = [0, 0]
        self._angular_gain = [0, 0]

    def _joy_callback(self, msg):
        """
        Callback function for the joy topic.\n
        The callback function is called everytime a message is published to the joy topic.\n
        The callback function is used to set the gain of the linear and angular velocities of the twist message.\n
        The gain is set by the right trigger of the hydra joystick.\n
        The gain is set to the value of the right trigger divided by 10.\n
        The gain is set to 0 if the right trigger is not pressed.\n
        """

        if msg.buttons[3] == 1:
            self._handle_arm_switch()

        if msg.buttons[-7] == 1:
            
            self._linear_gain[0] = msg.axes[-7] * 2
            self._angular_gain[0] = msg.axes[-7] * 5

            self._linear_gain[1] = msg.axes[-7] * 2
            self._angular_gain[1] = msg.axes[-7] * 5


        else:
            self._linear_gain = [0, 0]
            self._angular_gain = [0, 0]

    def _handle_arm_switch(self):
        """
        Handles the arm switching logic with debouncing.
        """
        current_time = time.time()
        if current_time - self._last_switch_time >= self._switch_debounce_interval:
            self._last_switch_time = current_time
            self._switch_arms()

    def _switch_arms(self):
        """
        Switch the arms that will be controlled by the hydra.\n
        The arms are switched by pressing the left trigger of the hydra joystick.\n
        The arms are switched by changing the topic of the current twist publisher.\n
        """

        if self._switched:
            self._current_twist_pub = self._right_twist_pub
            self._switched = False
        else:
            self._current_twist_pub = self._left_twist_pub
            self._switched = True


    def _twist_to_twist_stamped_msg(self, controller_frame_id, base_frame_id, lin_gain, ang_gain):
        """
        Get the twist of the hydra right pivot in the hydra base frame.\n
        The twist is computed as the difference between the current pose and the previous pose divided by the time difference.\n
        
        This is done by a powerful method of lookupTwist from tf.TransformListener.\n
        The method lookupTwist returns a tuple of 6 elements, which are the linear and angular velocities.\n
        
        
        """

         # @TODO figuring out how to track the time difference between the current pose and the previous pose that will be used as buffer time for the lookupTwist method.
        
        twist = self._tf_listener.lookupTwist(controller_frame_id, base_frame_id, rospy.Time() , rospy.Duration(0.1))
        self._twiststamped_msg.header.stamp = rospy.Time.now()
        self._twiststamped_msg.header.frame_id = "torso_lift_link"
        self._twiststamped_msg.twist.linear.x = twist[0][0] * lin_gain
        self._twiststamped_msg.twist.linear.y = twist[0][1] * lin_gain
        self._twiststamped_msg.twist.linear.z = twist[0][2] * lin_gain

        self._twiststamped_msg.twist.angular.x = twist[1][0] * ang_gain
        self._twiststamped_msg.twist.angular.y = twist[1][1] * ang_gain
        self._twiststamped_msg.twist.angular.z = twist[1][2] * ang_gain
        return self._twiststamped_msg

    def run(self):

        rate = rospy.Rate(5)
        synced = False

        while not rospy.is_shutdown():

            if not synced:
                try:
                    self._tf_listener.waitForTransform("/hydra_right_grab", "/hydra_base", rospy.Time(), rospy.Duration(20))
                    synced = True
                except:
                    rospy.logwarn("Waiting for tf")
                    continue
            
            right_twist = self._twist_to_twist_stamped_msg("/hydra_right_grab", "/hydra_base", self._linear_gain[0], self._angular_gain[0])
            self._current_twist_pub.publish(right_twist)

            rate.sleep()
    
if __name__ == "__main__":
        
    rospy.init_node("hydra_reader")

    hydra_reader = HydraReader()
    hydra_reader.run()
        



