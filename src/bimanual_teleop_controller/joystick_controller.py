import numpy as np
import pygame
import sys

import rospy
import rospkg
from sensor_msgs.msg import Joy, JoyFeedback, JoyFeedbackArray
from bimanual_teleop_controller.math_utils import LowPassFilter
from bimanual_teleop_controller.utility import load_config


class JoystickController():
    def __init__(self, motion_tracker=False):

        self._joy_pygame = JoystickController._joy_pygame_init()
        self._is_rumbled = False
        rospy.logdebug('Initiating joystick driver')

        # Load the joystick mapping configuration
        controller_name = self._joy_pygame.get_name()
        rospy.logdebug(f'Controller name: {controller_name}')

        joy_mapping_cfg = load_config(rospkg.RosPack().get_path('bimanual_teleop_controller') /
                                      + '/config/joy_mapping.yaml')
        
        self._left_arm_index = joy_mapping_cfg[controller_name]['left_arm_index']
        self._right_arm_index = joy_mapping_cfg[controller_name]['right_arm_index']
        self._dead_switch_index = joy_mapping_cfg[controller_name]['dead_switch_index']
        self._system_halt_index = joy_mapping_cfg[controller_name]['system_halt_index']
        self._gripper_open_index = joy_mapping_cfg[controller_name]['gripper_open_index']
        self._gripper_close_index = joy_mapping_cfg[controller_name]['gripper_close_index']
        self._trigger_constraint_index = joy_mapping_cfg[controller_name]['trigger_constraint_index']

        self._control_mapping = joy_mapping_cfg[controller_name]['controls']


        # Initialize the joystick message
        joy_topic = "/joy"
        self._joy_msg = rospy.wait_for_message(joy_topic, Joy)
        self._subscriber = rospy.Subscriber(joy_topic, Joy, self._joy_callback)
        self._feedback_pub = rospy.Publisher(
            "/joy/set_feedback", JoyFeedbackArray, queue_size=10)

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

        if pygame_joy:
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
            joy_msg = self.get_joy_msg()
            trigger_side = 5 if not base else 2
            aggressive = (-joy_msg[0][trigger_side] + 1) / 2

            # Apply low-pass filter
            vy = self.lpf_vy.filter(-joy_msg[0][self._control_mapping['y_ax']] / np.abs(joy_msg[0][self._control_mapping['y_ax']])
                                    if joy_msg[0][self._control_mapping['y_ax']] != 0 else 0)
            vx = self.lpf_vx.filter(-joy_msg[0][self._control_mapping['x_ax']] / np.abs(joy_msg[0][self._control_mapping['x_ax']])
                                    if joy_msg[0][self._control_mapping['x_ax']] != 0 else 0)
            y = joy_msg[1][self._control_mapping['yaw_left']] - \
                joy_msg[1][self._control_mapping['yaw_right']]  # button X and B

            if not base:
                vz = self.lpf_vz.filter(joy_msg[1][self._control_mapping['up']] - joy_msg[1][self._control_mapping['down']])  # button Y and A
                r = self.lpf_r.filter(joy_msg[0][self._control_mapping['roll_ax']] / np.abs(joy_msg[0][self._control_mapping['roll_ax']]) \
                                      if joy_msg[0][self._control_mapping['roll_ax']] != 0 else 0)
                p = self.lpf_p.filter(-joy_msg[0][self._control_mapping['pitch_ax']] / np.abs(joy_msg[0][self._control_mapping['pitch_ax']]) \
                                      if joy_msg[0][self._control_mapping['pitch_ax']] != 0 else 0)

            # vy = self.lpf_vy.filter(-self._joy_msg[0][self._y_ax] / np.abs(
            #     self._joy_msg[0][self._y_ax]) if self._joy_msg[0][self._y_ax] != 0 else 0)
            # vx = self.lpf_vx.filter(-self._joy_msg[0][self._x_ax] / np.abs(
            #     self._joy_msg[0][self._x_ax]) if self._joy_msg[0][self._x_ax] != 0 else 0)
            # y = self._joy_msg[1][self._yaw_left] - self._joy_msg[1][self._yaw_right]  # button X and B

            # if not base:
            #     vz = self.lpf_vz.filter(
            #         self._joy_msg[1][self._up] - self._joy_msg[1][self._down])  # button Y and A
            #     r = self.lpf_r.filter(
            #         self._joy_msg[0][self._roll_ax] / np.abs(self._joy_msg[0][self._roll_ax]) if self._joy_msg[0][self._roll_ax] != 0 else 0)
            #     p = self.lpf_p.filter(-self._joy_msg[0][self._pitch_ax] / np.abs(
            #         self._joy_msg[0][self._pitch_ax]) if self._joy_msg[0][self._pitch_ax] != 0 else 0)


        twist = np.zeros(6)
        twist[:3] = np.array([vx, vy, vz]) * gain[0] * aggressive
        twist[3:] = np.array([r, p, y]) * gain[1] * aggressive
        return twist, done

    def controller_LED_on(self, LED_id):
        joy_fbs_msg = self.joyfb_init(JoyFeedback.TYPE_LED, LED_id, 1)
        self._feedback_pub.publish(joy_fbs_msg)

    def rumble(self, strength):
        joy_fbs_msg = self.joyfb_init(JoyFeedback.TYPE_RUMBLE, 0, strength)
        self._feedback_pub.publish(joy_fbs_msg)

    def start_rumble(self, low_freq=0.5, high_freq=0.5, duration=1):
        self._joy_pygame.rumble(low_frequency=low_freq,
                                high_frequency=high_freq, duration=duration)
        self._is_rumbled = True

    def stop_rumble(self):
        self._joy_pygame.stop_rumble()
        self._is_rumbled = False

    def is_rumbled(self):
        return self._is_rumbled
    
    @staticmethod
    def joyfb_init(type, id, intensity):
        joy_fbs = JoyFeedbackArray()
        joy_fb = JoyFeedback()
        joy_fb.type = type
        joy_fb.id = id
        joy_fb.intensity = intensity
        joy_fbs.array.append(joy_fb)

        return joy_fbs

    @staticmethod
    def _joy_pygame_init():
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
