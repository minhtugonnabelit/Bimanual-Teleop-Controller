#!/usr/bin/env python3

import numpy as np
import threading
import time
import os

import rospy
from visualization_msgs.msg import MarkerArray

from bimanual_teleop_controller.utility import *
from bimanual_teleop_controller.math_utils import CalcFuncs
from bimanual_teleop_controller.pr2_controller import PR2Controller
from bimanual_teleop_controller.joystick_controller import JoystickController as jsk


class BMCP:

    # TODO Inspect issue on hand tracking node crashing

    def __init__(self, config, data_plot, motion_tracker, dominant_hand) -> None:
        
        self._DAMPER_STEEPNESS = config['DAMPER_STEEPNESS']
        self._MANIP_THRESH = config['MANIPULABILITY_THRESHOLD']
        self._CONTROL_RATE = config['CONTROL_RATE']
        self._TWIST_GAIN = config['TWIST_GAIN']
        self._DRIFT_GAIN = config['DRIFT_GAIN']
        self._HOLD_DURATION = config['HOLD_DURATION']

        self._data_plot = data_plot
        self._dominant_hand = dominant_hand
        self._motion_tracker = motion_tracker

        # State variables
        self._state = 'individual'
        self._mode_switched = False
        self._hold_start_time = None
        self._constraint_is_set = False


        if self._motion_tracker:
            self._hand_gesture = {
                'Left': None,
                'Right': None
            }
            self._hand_markers_sub = rospy.Subscriber(
                '/hand_markers', MarkerArray, self._hand_markers_callback)

            self._left_twist = np.zeros(6)
            self._left_twist_sub = rospy.Subscriber(
                '/Left_hand_twist', TwistStamped, self._left_twist_callback)

            self._right_twist = np.zeros(6)
            self._right_twist_sub = rospy.Subscriber(
                '/Right_hand_twist', TwistStamped, self._right_twist_callback)
        else:
            self.joystick = jsk(motion_tracker=self._motion_tracker)
            self.joystick.set_LED_green() # LED indicating initialisation


        self.controller = PR2Controller(rate=self._CONTROL_RATE,
                                        config=config,
                                        data_plotter=self._data_plot)

        self.controller.set_manip_thresh(self._MANIP_THRESH)
        self.controller.move_to_neutral()
        self.controller.move_head_to([0.0, 0.2])
        self.joystick.set_LED_blue() # LED indicating controller ready and in indiv mode
        self._right_arm = self.controller.get_arm_controller('r')
        self._left_arm = self.controller.get_arm_controller('l')
        rospy.loginfo('Robot is in neutral position')
        rospy.sleep(1)


        # Control signal variables
        self._qdot_right = np.zeros(7)
        self._qdot_left = np.zeros(7)

        # Threads and thread control variables
        self._control_signal_ready = threading.Condition()
        self._control_signal_thread = threading.Thread(
            target=self.control_signal_handler)
        self._base_controller_thread = threading.Thread(
            target=self.base_controller_handler)
        self._data_recording_thread = threading.Thread(
            target=self.data_recording_handler)

    # State control functions

    def switch_to_individual_control(self):
        self._state = 'individual'

    def switch_to_central_control(self):
        self._state = 'central'

    def stop(self):
        self._state = 'Done'

    def handle_mode_change(self):

        if self._hold_start_time is None:
            self._hold_start_time = time.time()
            self._mode_switched = False
       
        elif time.time() - self._hold_start_time > self._HOLD_DURATION:
            if not self._mode_switched:
                if not self._constraint_is_set:
                    self._constraint_is_set, _, _ = self.controller.set_kinematics_constraints()
                    self.joystick.set_rumble_strength(0)
                    self.joystick.set_LED_red()
                    rospy.loginfo(
                        'Constraint is set, switching controllers, started velocity controller thread')

                else:
                    self._constraint_is_set = self.controller.reset_constraints()
                    self.joystick.set_rumble_strength(0)
                    self.joystick.set_LED_blue()
                    # self.joystick.rumble(0)
                    rospy.loginfo(
                        'Constraint is unset, switching back to individual control')
                
                self._mode_switched = True

    def reset_hold_timer(self):
        self._hold_start_time = None
        self._mode_switched = False

    def stop_teleop(self):
        self.stop()
        rospy.loginfo('Done teleoperation.')
        rospy.signal_shutdown('Done')

        self._control_signal_thread.join()
        rospy.loginfo('Control signal thread joined.')
        if not self._motion_tracker:
            self._base_controller_thread.join()
            rospy.loginfo('Base controller thread joined.')

        if self._data_plot:
            self._data_recording_thread.join()
            rospy.loginfo('Data recording thread joined.')

        os.system(
            'rosnode kill /hand_tracker') if self._motion_tracker else os.system('rosnode kill /joy')

    # Callback functions

    def _hand_markers_callback(self, msg: MarkerArray):
        for marker in msg.markers:
            self._hand_gesture[marker.ns] = marker.text

    def _left_twist_callback(self, msg: TwistStamped):
        self._left_twist[:3] = np.array([msg.twist.linear.x,
                                         msg.twist.linear.y,
                                         msg.twist.linear.z]) 

    def _right_twist_callback(self, msg: TwistStamped):
        self._right_twist[:3] = np.array([msg.twist.linear.x,
                                          msg.twist.linear.y,
                                          msg.twist.linear.z])

    def _get_twist_from_hand(self, side):
        return self._right_twist if side == 'Right' else self._left_twist

    # Initial teleoperation function
    def run(self):
        r"""
        Initial teleoperation function
        """
        rospy.loginfo('Start teleop using joystick')

        if not self._motion_tracker:
            self._left_arm_index = self.joystick.left_arm_index
            self._right_arm_index = self.joystick.right_arm_index
            self._dead_switch_index = self.joystick.dead_switch_index
            self._system_halt_index = self.joystick.system_halt_index
            self._gripper_open_index = self.joystick.gripper_open_index
            self._gripper_close_index = self.joystick.gripper_close_index
            self._trigger_constraint_index = self.joystick.trigger_constraint_index

        # self.switch_to_central_control()
        self.controller.start_jg_vel_controller()
        self._control_signal_thread.start()
        self._base_controller_thread.start() if not self._motion_tracker else None

        rospy.sleep(1)

        while not rospy.is_shutdown():
            qdot = np.zeros(14)
            qdot = self.teleop_gesture(qdot) if self._motion_tracker \
                else self.teleop_joystick(qdot)

            self._qdot_right = qdot[7:]
            self._qdot_left = qdot[:7]

    def teleop_joystick(self, qdot):

        joy_msg = self.joystick.get_joy_msg()

        if self.joystick.using_ds4:
            if joy_msg.button_ps:
                self.stop_teleop()

            self.handle_constrained_twist(qdot, joy_msg.twist) if (joy_msg.button_share * joy_msg.button_options)else self.reset_hold_timer()

            twist, _ = self.joystick.joy_to_twist(self._TWIST_GAIN)
            if self._constraint_is_set:
                if joy_msg.button_r1 and joy_msg.button_l1:
                    qdot = self.handle_constrained_twist(qdot, twist)
            else:
                qdot = self.handle_indiv_arm_joy(qdot, twist, joy_msg)

        else:
            if joy_msg[1][self._system_halt_index]:
                self.stop_teleop()

            self.handle_mode_change() if (joy_msg[1][self._trigger_constraint_index[0]] * joy_msg[1][self._trigger_constraint_index[1]])\
                else self.reset_hold_timer()
            
            if joy_msg[0][self._dead_switch_index] != 1:
                twist, _ = self.joystick.joy_to_twist(self._TWIST_GAIN)
                qdot = self.handle_constrained_twist(qdot, twist) if self._constraint_is_set \
                    else self.handle_indiv_arm_joy(qdot, twist, joy_msg)

        return qdot

    def handle_indiv_arm_joy(self, qdot, twist, joy_msg):
        if self.joystick.using_ds4:
            if joy_msg.button_l1:
                qdot[7:] = self.controller.process_arm_movement(
                    side='r', twist=twist, manip_thresh=self._MANIP_THRESH, damper_steepness=self._DAMPER_STEEPNESS)
            if joy_msg.button_r1:
                qdot[:7] = self.controller.process_arm_movement(
                    side='l', twist=twist, manip_thresh=self._MANIP_THRESH, damper_steepness=self._DAMPER_STEEPNESS)
        else:
            if joy_msg[1][self._right_arm_index]:  # left bumper
                qdot[7:] = self.controller.process_arm_movement(
                    side='r', twist=twist, manip_thresh=self._MANIP_THRESH, damper_steepness=self._DAMPER_STEEPNESS)
            if joy_msg[1][self._left_arm_index]:  # right bumper
                qdot[:7] = self.controller.process_arm_movement(
                    side='l', twist=twist, manip_thresh=self._MANIP_THRESH, damper_steepness=self._DAMPER_STEEPNESS)

        return qdot

    def teleop_gesture(self, qdot):

        if self._hand_gesture['Left'] == 'Pointing_Up' and self._hand_gesture['Right'] == 'Pointing_Up':
            self.stop_teleop()

        self.handle_mode_change() if (self._hand_gesture['Left'] == 'Thumb_Up' and self._hand_gesture['Right'] == 'Thumb_Up')\
            else self.reset_hold_timer()

        if self._hand_gesture[self._dominant_hand] == 'Closed_Fist':
            twist_left = self._get_twist_from_hand('Left')
            twist_right = self._get_twist_from_hand('Right')
            if self._constraint_is_set:
                object_twist = twist_right if self._dominant_hand == 'Right' else twist_left
                qdot = self.handle_constrained_twist(qdot, object_twist)
            else:
                qdot = self.handle_indiv_arm_ges(
                    qdot, twist_left, twist_right)

        return qdot

    def handle_indiv_arm_ges(self, qdot, twist_left, twist_right):
        if self._hand_gesture['Left'] == 'Closed_Fist':
            qdot[:7] = self.controller.process_arm_movement(
                side='l', twist=twist_left, manip_thresh=self._MANIP_THRESH, damper_steepness=self._DAMPER_STEEPNESS)
        if self._hand_gesture['Right'] == 'Closed_Fist':
            qdot[7:] = self.controller.process_arm_movement(
                side='r', twist=twist_right, manip_thresh=self._MANIP_THRESH, damper_steepness=self._DAMPER_STEEPNESS)

        return qdot

    # Constraint handling functions

    def handle_constrained_twist(self, qdot, object_twist):

        twist_left = self.controller.get_twist_in_tool_frame(
            side='l', twist=object_twist)
        twist_right = self.controller.get_twist_in_tool_frame(
            side='r', twist=object_twist)

        # Extract the Jacobians in the middle frame using the virtual robot with joint states data from the real robot
        jacob_right = self.controller.get_jacobian(side='r')
        jacob_left = self.controller.get_jacobian(side='l')
        jacob_constraint = np.c_[jacob_left, -jacob_right]

        # Calculate the joint velocities using RMRC
        qdot_right = CalcFuncs.rmrc(
            jacob_right, twist_right, w_thresh=self._MANIP_THRESH)
        qdot_left = CalcFuncs.rmrc(
            jacob_left, twist_left,  w_thresh=self._MANIP_THRESH)
        qdot_combined = np.r_[qdot_left, qdot_right]

        # Perform nullspace projection for qdot_combined on constraint Jacobian to ensure the twist synchronisatio
        taskspace_drift_compensation = self.controller.task_drift_compensation(gain_p=self._DRIFT_GAIN['p'],
                                                                               gain_d=self._DRIFT_GAIN['d'],
                                                                               on_taskspace=True) * 2

        # Combine the primary and secondary tasks velocities
        primary_tasks_vel = np.linalg.pinv(
            jacob_constraint) @ taskspace_drift_compensation
        secondary_tasks_vel = CalcFuncs.nullspace_projector(
            jacob_constraint) @ qdot_combined
        qdot = primary_tasks_vel + secondary_tasks_vel

        # Add a joint limits damper to the joint velocities
        qdot_damped, max_weights_scaled = self.controller.joint_limit_damper(
            qdot, steepness=self._DAMPER_STEEPNESS)
        
        qdot += qdot_damped
        self.joystick.set_rumble_strength(strength=max_weights_scaled)
        # self.joystick.rumble(strength=max_weights_scaled)

        return qdot

    # Thread handlers functions

    def handle_gripper(self, arm, joy_msg):

        if self._motion_tracker:
            if joy_msg[1][self._gripper_open_index] == 1:
                arm.open_gripper()
            elif joy_msg[1][self._gripper_close_index] == 1:
                arm.close_gripper()
        
        elif self.joystick.using_ds4:
            if joy_msg.button_dpad_right:
                arm.open_gripper()
            if joy_msg.button_dpad_left:
                arm.close_gripper()
        else:
            if self.joystick.controller_name == "Xbox 360 Controller":
                if joy_msg[0][self._gripper_close_index] == 1:  # up
                    arm.open_gripper()
                elif - joy_msg[0][self._gripper_close_index] == 1:  # down
                    arm.close_gripper()

            elif self.joystick.controller_name == "Sony PLAYSTATION(R)3 Controller":
                if joy_msg[1][self._gripper_open_index]:
                    arm.open_gripper()
                elif joy_msg[1][self._gripper_close_index]:
                    arm.close_gripper()

            else :
                if joy_msg[1][self._gripper_open_index] == 1:
                    arm.open_gripper()
                elif - joy_msg[1][self._gripper_close_index] == 1:
                    arm.close_gripper()

    def base_controller_handler(self):

        while self._state != 'Done':

            joy_msg = self.joystick.get_joy_msg()
            twist = np.zeros(6)

            if not self.joystick.using_ds4:

                if joy_msg[1][self._right_arm_index]:  # left bumper
                    self.handle_gripper(self._right_arm, joy_msg)

                if joy_msg[1][self._left_arm_index]:  # right bumper
                    self.handle_gripper(self._left_arm, joy_msg)

                if joy_msg[0][2] != 1:  # left trigger for base controller
                    twist, _ = self.joystick.joy_to_twist(
                        self._TWIST_GAIN, base=True)
            
            else:

                if joy_msg.button_l1:
                    self.handle_gripper(self._right_arm, joy_msg)

                if joy_msg.button_r1:
                    self.handle_gripper(self._left_arm, joy_msg)

                if not joy_msg.button_l2:
                    twist, _ = self.joystick.joy_to_twist(
                        self._TWIST_GAIN, base=True)

            self.controller.move_base(twist)
            self.controller.sleep()

    def control_signal_handler(self):

        while self._state != 'Done':

            with self._control_signal_ready:
                self._control_signal_ready.notify()

            self._right_arm.send_joint_command(
                joint_command=self._qdot_right)
            self._left_arm.send_joint_command(
                joint_command=self._qdot_left)
            self.controller.sleep()

    def data_recording_handler(self):

        while not rospy.is_shutdown() and self._state != 'Done':

            with self._control_signal_ready:
                self._control_signal_ready.wait(timeout=2.0)

            if self._state != 'individual' and self._state != 'Done':

                self.controller.store_joint_positions()
                self.controller.store_joint_efforts()
                self.controller.store_joint_velocities('right', self._qdot_right)
                self.controller.store_joint_velocities('left', self._qdot_left)

                self.controller.store_manipulability()
                self.controller.store_drift()


if __name__ == "__main__":
    try:
        rospy.init_node('bimanual_controller', log_level=1, anonymous=True)
        data_plot = rospy.get_param('~data_plot', False)
        motion_tracker = rospy.get_param('~motion_tracker', False)
        dominant_hand = rospy.get_param('~dominant_hand', 'Right')
        controller_cfg = load_config(rospkg.RosPack().get_path('bimanual_teleop_controller') + '/config/bmcp_cfg.yaml')

        b = BMCP(config=controller_cfg,
                 data_plot=data_plot,
                 motion_tracker=motion_tracker,
                 dominant_hand=dominant_hand)
        b.run()

    except rospy.ROSInterruptException as e:
        rospy.logerr(e)
