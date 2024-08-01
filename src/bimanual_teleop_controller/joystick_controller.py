import numpy as np
import pygame
import sys

import rospy, tf
from sensor_msgs.msg import Joy
from geometry_msgs.msg import TwistStamped


class LowPassFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.prev_value = None

    def filter(self, value):
        if self.prev_value is None:
            self.prev_value = value
        filtered_value = self.alpha * value + \
            (1 - self.alpha) * self.prev_value
        self.prev_value = filtered_value
        return filtered_value


class JoystickController():
    def __init__(self, motion_tracker=False):

        self._joy_pygame = JoystickController._joy_init()
        self._is_rumbled = False
        rospy.logdebug('Initiating joystick driver')
        
        joy_topic = "/joy"

        self._dead_switch_index = 5
        self._system_halt_index = -3
        self._right_arm_index = 4
        self._left_arm_index = 5
        self._gripper_open_index = -1
        self._gripper_close_index = -1
        self._trigger_constraint_index = [6,7]

        self._up = 3
        self._down = 0
        self._yaw_left = 2
        self._yaw_right = 1
        self._x_ax = 1
        self._y_ax = 0
        self._roll_ax = 3
        self._pitch_ax = 4

        controller_name = self._joy_pygame.get_name()
        if controller_name == "Sony PLAYSTATION(R)3 Controller":
            # pass
            self._dead_switch_index = 5
            self._system_halt_index = -7
            self._right_arm_index = 4
            self._left_arm_index = 5
            self._gripper_open_index = -4
            self._gripper_close_index = -3
            self._trigger_constraint_index = [8, 9]
            
            self._up = 2
            self._down = 0
            self._yaw_left = 3
            self._yaw_right = 1
            self._x_ax = 1
            self._y_ax = 0
            self._roll_ax = 3
            self._pitch_ax = 4

        self._motion_tracker = motion_tracker
        if self._motion_tracker:
            joy_topic = "/hydra_right_joy"

            self._dead_switch_index = 7
            self._system_halt_index = 0
            self._right_arm_index = 5
            self._left_arm_index = 3
            self._gripper_open_index = 4
            self._gripper_close_index = 2
            self._trigger_constraint_index = [1,6]

            self._tf_listener = tf.TransformListener()
            self._controller_frame_id = "hydra_right_pivot"
            self._base_frame_id = "hydra_base"

        self._joy_msg = rospy.wait_for_message(joy_topic, Joy)
        self._subscriber = rospy.Subscriber(joy_topic, Joy, self._joy_callback)

        # Initialize low-pass filters for each axis
        alpha = 0.3  # Smoothing factor for the low-pass filter
        self.lpf_vx = LowPassFilter(alpha)
        self.lpf_vy = LowPassFilter(alpha)
        self.lpf_vz = LowPassFilter(alpha)
        self.lpf_r = LowPassFilter(alpha)
        self.lpf_p = LowPassFilter(alpha)
        self.lpf_y = LowPassFilter(alpha)

    def _joy_callback(self, joy_msg: Joy):
        self._joy_msg = (joy_msg.axes, joy_msg.buttons)

    def motion_to_twist(self, gain, base=False, pygame_joy=False):
        return self.joy_to_twist(gain, base, pygame_joy)

    def get_joy_msg(self):
        return self._joy_msg

    def joy_to_twist(self, gain, base=False, pygame_joy=False):

        vx, vy, vz, r, p, y = 0, 0, 0, 0, 0, 0
        done = False
        aggressive = 0
        motion_amp = 15

        if self._motion_tracker:
            aggressive = self._joy_msg[0][-1]
            twist_stamped = self._twist_to_twist_stamped_msg(
                self._controller_frame_id,
                self._base_frame_id,
                gain[0]*motion_amp,
                gain[1]*motion_amp
            )
            vx, vy, vz = - twist_stamped.twist.linear.x, - twist_stamped.twist.linear.y, twist_stamped.twist.linear.z
            r, p, y = - twist_stamped.twist.angular.x, - twist_stamped.twist.angular.y, twist_stamped.twist.angular.z
        elif pygame_joy:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            if self._joy_pygame.get_button(0):
                print("Button 0 is pressed")
                done = True

            vz = (self._joy_pygame.get_axis(2) + 1) / 2 - \
                (self._joy_pygame.get_axis(5) + 1) / 2
            y = self._joy_pygame.get_button(
                1) * 0.1 - self._joy_pygame.get_button(3) * 0.1

            # Apply low-pass filter
            vy = self.lpf_vy.filter(self._joy_pygame.get_axis(1))
            vx = self.lpf_vx.filter(self._joy_pygame.get_axis(1))
            r = self.lpf_r.filter(self._joy_pygame.get_axis(3))
            p = self.lpf_p.filter(self._joy_pygame.get_axis(4))

        else:
            trigger_side = 5 if not base else 2
            aggressive = (-self._joy_msg[0][trigger_side] + 1) / 2

            # Apply low-pass filter
            vy = self.lpf_vy.filter(-self._joy_msg[0][self._y_ax] / np.abs(
                self._joy_msg[0][self._y_ax]) if self._joy_msg[0][self._y_ax] != 0 else 0)
            vx = self.lpf_vx.filter(-self._joy_msg[0][self._x_ax] / np.abs(
                self._joy_msg[0][self._x_ax]) if self._joy_msg[0][self._x_ax] != 0 else 0)
            y = self._joy_msg[1][self._yaw_left] - self._joy_msg[1][self._yaw_right]  # button X and B

            if not base:
                vz = self.lpf_vz.filter(
                    self._joy_msg[1][self._up] - self._joy_msg[1][self._down])  # button Y and A
                r = self.lpf_r.filter(
                    self._joy_msg[0][self._roll_ax] / np.abs(self._joy_msg[0][self._roll_ax]) if self._joy_msg[0][self._roll_ax] != 0 else 0)
                p = self.lpf_p.filter(-self._joy_msg[0][self._pitch_ax] / np.abs(
                    self._joy_msg[0][self._pitch_ax]) if self._joy_msg[0][self._pitch_ax] != 0 else 0)
                
            # trigger_side = 5 if not base else 2
            # aggressive = (-self._joy_msg[0][trigger_side] + 1) / 2

            # # Apply low-pass filter
            # vy = self.lpf_vy.filter(-self._joy_msg[0][0] / np.abs(
            #     self._joy_msg[0][0]) if self._joy_msg[0][0] != 0 else 0)
            # vx = self.lpf_vx.filter(-self._joy_msg[0][1] / np.abs(
            #     self._joy_msg[0][1]) if self._joy_msg[0][1] != 0 else 0)
            # y = self._joy_msg[1][2] - self._joy_msg[1][1]  # button X and B

            # if not base:
            #     vz = self.lpf_vz.filter(
            #         self._joy_msg[1][3] - self._joy_msg[1][0])  # button Y and A
            #     r = self.lpf_r.filter(
            #         self._joy_msg[0][3] / np.abs(self._joy_msg[0][3]) if self._joy_msg[0][3] != 0 else 0)
            #     p = self.lpf_p.filter(-self._joy_msg[0][4] / np.abs(
            #         self._joy_msg[0][4]) if self._joy_msg[0][4] != 0 else 0)

        twist = np.zeros(6)
        twist[:3] = np.array([vx, vy, vz]) * gain[0] * aggressive
        twist[3:] = np.array([r, p, y]) * gain[1] * aggressive
        return twist, done

    def _twist_to_twist_stamped_msg(self, controller_frame_id, base_frame_id, lin_gain, ang_gain, time_diff=0.1):
        twist = self._tf_listener.lookupTwist(
            controller_frame_id, base_frame_id, rospy.Time(), rospy.Duration(time_diff))
        twiststamped_msg = TwistStamped()
        twiststamped_msg.header.stamp = rospy.Time.now()
        twiststamped_msg.header.frame_id = "torso_lift_link"

        # Apply low-pass filters
        twiststamped_msg.twist.linear.x = self.lpf_vx.filter(
            twist[0][0] * lin_gain)
        twiststamped_msg.twist.linear.y = self.lpf_vy.filter(
            twist[0][1] * lin_gain)
        twiststamped_msg.twist.linear.z = self.lpf_vz.filter(
            twist[0][2] * lin_gain)

        twiststamped_msg.twist.angular.x = self.lpf_r.filter(
            twist[1][0] * ang_gain)
        twiststamped_msg.twist.angular.y = self.lpf_p.filter(
            twist[1][1] * ang_gain)
        twiststamped_msg.twist.angular.z = self.lpf_y.filter(
            twist[1][2] * ang_gain)

        return twiststamped_msg

    def start_rumble(self, low_freq=0.5, high_freq=0.5, duration=1):
        if not self._motion_tracker:
            self._joy_pygame.rumble(low_frequency=low_freq,
                                    high_frequency=high_freq, duration=duration)
            self._is_rumbled = True

    def stop_rumble(self):
        if not self._motion_tracker:
            self._joy_pygame.stop_rumble()
            self._is_rumbled = False

    def is_rumbled(self):
        return self._is_rumbled

    @staticmethod
    def _joy_init():
        pygame.init()
        joystick_count = pygame.joystick.get_count()
        if joystick_count == 0:
            raise Exception('No joystick found')
        else:
            joystick = pygame.joystick.Joystick(0)
            joystick.init()

        return joystick
    
    @property
    def dead_switch_index(self):
        return self._dead_switch_index
    
    @property
    def system_halt_index(self):
        return self._system_halt_index
    
    @property
    def right_arm_index(self):
        return self._right_arm_index
    
    @property
    def left_arm_index(self):
        return self._left_arm_index
    
    @property
    def gripper_open_index(self):
        return self._gripper_open_index
    
    @property
    def gripper_close_index(self):
        return self._gripper_close_index
    
    @property
    def trigger_constraint_index(self):
        return self._trigger_constraint_index
    
    @property
    def controller_name(self):
        return self._joy_pygame.get_name()