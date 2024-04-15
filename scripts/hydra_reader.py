import rospy
import tf
import time
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Joy

class HydraTwist():

    _MAX_LIN_GAIN = 5
    _MAX_ANG_GAIN = 10
    _SWITCH_DEBOUCE_INTERVAL = 1

    def __init__(self, joy_topic, twist_topic, controller_frame_id, base_frame_id, timeout):

        # Data members to load from the parameter server
        self._controller_frame_id = controller_frame_id
        self._base_frame_id = base_frame_id

        # Data members to be initialized
        self._linear_gain = 0
        self._angular_gain = 0
        self._trigger_index = -2
        self._bumper_index = -4
        self._last_switch_time = 0
        self._switched = False
        self._timeout = timeout

        # Initialize ROS components
        self._tf_listener = tf.TransformListener()
        self._joysub = rospy.Subscriber(joy_topic, Joy, self._joy_callback)
        self._twist_pub = rospy.Publisher(twist_topic, TwistStamped, queue_size=1)
        


        # self._left_twist_pub = rospy.Publisher("/l_arm_servo_server/delta_twist_cmds", TwistStamped, queue_size=1)

        # self._current_twist_pub = self._right_twist_pub

        # self._twist_pub = self._right_twist_pub


    def _joy_callback(self, msg):
        r"""
        Callback function for the joy topic.\n
        The callback function is called everytime a message is published to the joy topic.\n
        The callback function is used to set the gain of the linear and angular velocities of the twist message.\n
        The gain is set by the right trigger of the hydra joystick.\n
        The gain is set to the value of the right trigger divided by 10.\n
        The gain is set to 0 if the right trigger is not pressed.\n
        """

        if msg.buttons[3] == 1:
            self._handle_arm_switch()

        self._linear_gain = msg.axes[self._trigger_index] * self._MAX_LIN_GAIN if msg.buttons[self._trigger_index] == 1 else 0
        self._angular_gain = msg.axes[self._trigger_index] * self._MAX_ANG_GAIN if msg.buttons[self._trigger_index] == 1 else 0


    def _handle_arm_switch(self):
        """
        Handles the arm switching logic with debouncing.
        """
        current_time = time.time()
        if current_time - self._last_switch_time >= self._SWITCH_DEBOUCE_INTERVAL:
            self._last_switch_time = current_time
            self._switch_arms()

    def _switch_arms(self):
        r"""
        Switch the arms that will be controlled by the hydra.\n
        The arms are switched by pressing the left trigger of the hydra joystick.\n
        The arms are switched by changing the topic of the current twist publisher.\n
        """
        # pass

        if self._switched:
            self._current_twist_pub = self._right_twist_pub
            self._switched = False
        else:
            self._current_twist_pub = self._left_twist_pub
            self._switched = True


    def _twist_to_twist_stamped_msg(self, controller_frame_id, base_frame_id, lin_gain, ang_gain, time_diff=0.1):
        r"""
        Get the twist of the hydra right pivot in the hydra base frame.\n
        The twist is computed as the difference between the current pose and the previous pose divided by the time difference.\n
        
        This is done by a powerful method of lookupTwist from tf.TransformListener.\n
        The method lookupTwist returns a tuple of 6 elements, which are the linear and angular velocities.\n
        
        
        """

         # @TODO figuring out how to track the time difference between the current pose and the previous pose that will be used as buffer time for the lookupTwist method.
        
        twist = self._tf_listener.lookupTwist(controller_frame_id, base_frame_id, rospy.Time() , rospy.Duration(time_diff))
        twiststamped_msg = TwistStamped()

        twiststamped_msg.header.stamp = rospy.Time.now()

        # Frame ID for each arm is set to the torso_lift_link
        twiststamped_msg.header.frame_id = "torso_lift_link"
        twiststamped_msg.twist.linear.x = twist[0][0] * lin_gain
        twiststamped_msg.twist.linear.y = twist[0][1] * lin_gain
        twiststamped_msg.twist.linear.z = twist[0][2] * lin_gain

        twiststamped_msg.twist.angular.x = twist[1][0] * ang_gain
        twiststamped_msg.twist.angular.y = twist[1][1] * ang_gain
        twiststamped_msg.twist.angular.z = twist[1][2] * ang_gain

        return twiststamped_msg

    def run(self):

        rate = rospy.Rate(5)
        synced = False

        while not rospy.is_shutdown():

            if not synced:
                try:
                    self._tf_listener.waitForTransform(self._controller_frame_id, self._base_frame_id, rospy.Time(), rospy.Duration(20))
                    synced = True
                except:
                    rospy.logwarn("Waiting for tf")
                    continue
                
            twist_msg = self._twist_to_twist_stamped_msg(self._controller_frame_id, self._base_frame_id, self._linear_gain, self._angular_gain)
            self._twist_pub.publish(twist_msg)

            rate.sleep()
            
    
if __name__ == "__main__":
        
    rospy.init_node("hydra_reader")

    # Retrieve parameters
    timeout = rospy.get_param('timeout')
    joy_topic = rospy.get_param('joy_topic')
    twist_topic = rospy.get_param('twist_topic')
    base_frame_id = rospy.get_param('base_frame_id')
    controller_frame_id = rospy.get_param('controller_frame_id')

    # Create and run the hydra reader
    hydra_reader = HydraTwist(joy_topic, twist_topic, controller_frame_id, base_frame_id, timeout)
    hydra_reader.run()

        



