import numpy as np
import pygame
import sys
import rospy
from sensor_msgs.msg import Joy

class LowPassFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.prev_value = None

    def filter(self, value):
        if self.prev_value is None:
            self.prev_value = value
        filtered_value = self.alpha * value + (1 - self.alpha) * self.prev_value
        self.prev_value = filtered_value
        return filtered_value

class JoystickController():
    def __init__(self):

        self._joy_msg = rospy.wait_for_message("/joy", Joy)
        self._subscriber = rospy.Subscriber("/joy", Joy, self._joy_callback)
        self._joy_pygame = JoystickController._joy_init()
        self._is_rumbled = False

        # Initialize low-pass filters for each axis
        alpha = 0.5  # Smoothing factor
        self.lpf_vx = LowPassFilter(alpha)
        self.lpf_vy = LowPassFilter(alpha)
        self.lpf_vz = LowPassFilter(alpha)
        self.lpf_r = LowPassFilter(alpha)
        self.lpf_p = LowPassFilter(alpha)
        self.lpf_y = LowPassFilter(alpha)

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

    def _joy_callback(self, joy_msg : Joy):
        self._joy_msg = (joy_msg.axes, joy_msg.buttons)

    def get_joy_msg(self):
        return self._joy_msg

    def joy_to_twist(self, gain, base=False, pygame_joy=False):

        vx, vy, vz, r, p, y = 0, 0, 0, 0, 0, 0
        done = False
        aggressive = 0
        # def lpf(value, threshold=0.1):
        #     return value if abs(value) > threshold else 0
        
        if pygame_joy:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            if self._joy_pygame.get_button(0):
                print("Button 0 is pressed")
                done = True

            vz = (self._joy_pygame.get_axis(2) + 1) / 2 - (self._joy_pygame.get_axis(5) + 1) / 2
            y = self._joy_pygame.get_button(1) * 0.1 - self._joy_pygame.get_button(3) * 0.1

            # Apply low-pass filter
            vy = self.lpf_vy.filter(self._joy_pygame.get_axis(1))
            vx = self.lpf_vx.filter(self._joy_pygame.get_axis(1))
            r = self.lpf_r.filter(self._joy_pygame.get_axis(3))
            p = self.lpf_p.filter(self._joy_pygame.get_axis(4))

        else:
            trigger_side = 5 if not base else 2
            aggressive = (-self._joy_msg[0][trigger_side] + 1) / 2

            # Apply low-pass filter
            vy = self.lpf_vy.filter(-self._joy_msg[0][0] / np.abs(self._joy_msg[0][0]) if self._joy_msg[0][0] != 0 else 0)
            vx = self.lpf_vx.filter(-self._joy_msg[0][1] / np.abs(self._joy_msg[0][1]) if self._joy_msg[0][1] != 0 else 0)
            y = self._joy_msg[1][2] - self._joy_msg[1][1]  # button X and B

            if not base:
                vz = self.lpf_vz.filter(self._joy_msg[1][3] - self._joy_msg[1][0])  # button Y and A
                r = self.lpf_r.filter(self._joy_msg[0][3] / np.abs(self._joy_msg[0][3]) if self._joy_msg[0][3] != 0 else 0)
                p = self.lpf_p.filter(-self._joy_msg[0][4] / np.abs(self._joy_msg[0][4]) if self._joy_msg[0][4] != 0 else 0)

        twist = np.zeros(6)
        twist[:3] = np.array([vx, vy, vz]) * gain[0] * aggressive
        twist[3:] = np.array([r, p, y]) * gain[1] * aggressive
        return twist, done
        
        # if pygame_joy:
        #     for event in pygame.event.get():
        #         if event.type == pygame.QUIT:
        #             pygame.quit()
        #             sys.exit()
        #     if self._joy_pygame.get_button(0):
        #         print("Button 0 is pressed")
        #         done = True

        #     vz = (lpf(self._joy_pygame.get_axis(2) + 1,))/2 - (lpf(self._joy_pygame.get_axis(5) + 1,))/2
        #     y = self._joy_pygame.get_button(1)*0.1 - self._joy_pygame.get_button(3)*0.1

        #     # Low pass filter
        #     vy = lpf(self._joy_pygame.get_axis(1)) 
        #     vx = lpf(self._joy_pygame.get_axis(1)) 
        #     r = lpf(self._joy_pygame.get_axis(3))  
        #     p = lpf(self._joy_pygame.get_axis(4)) 

        # else:

        #     trigger_side = 5 if not base else 2
        #     agressive =  (-lpf(self._joy_msg[0][trigger_side]) + 1)/2 

        #     # Low pass filter
        #     vy = - lpf(self._joy_msg[0][0]) / np.abs(lpf(self._joy_msg[0][0])) if lpf(self._joy_msg[0][0]) != 0 else 0
        #     vx = - lpf(self._joy_msg[0][1]) / np.abs(lpf(self._joy_msg[0][1])) if lpf(self._joy_msg[0][1]) != 0 else 0
        #     y = self._joy_msg[1][2] - self._joy_msg[1][1] # button X and B

        #     if not base:
        #         vz = self._joy_msg[1][3] - self._joy_msg[1][0] # button Y and A
        #         r = lpf(self._joy_msg[0][3]) / np.abs(lpf(self._joy_msg[0][3])) if lpf(self._joy_msg[0][3]) != 0 else 0
        #         p = - lpf(self._joy_msg[0][4]) / np.abs(lpf(self._joy_msg[0][4])) if lpf(self._joy_msg[0][4]) != 0 else 0

            
        # # ---------------------------------------------------------------------------#
        # twist = np.zeros(6)
        # twist[:3] = np.array([vx, vy, vz]) * gain[0] * agressive
        # twist[3:] = np.array([r, p, y]) * gain[1] * agressive
        # return twist, done
    
    def start_rumble(self, low_freq=0.5, high_freq=0.5, duration=1):
        self._joy_pygame.rumble(low_frequency=low_freq, high_frequency=high_freq, duration=duration)
        self._is_rumbled = True

    def stop_rumble(self):
        self._joy_pygame.stop_rumble()
        self._is_rumbled = False

    def is_rumbled(self):
        return self._is_rumbled



