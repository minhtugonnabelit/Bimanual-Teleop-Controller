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

    _DAMPER_STEEPNESS = config['DAMPER_STEEPNESS']
    _MANIP_THRESH = config['MANIPULABILITY_THRESHOLD']
    _CONTROL_RATE = config['CONTROL_RATE']
    _TWIST_GAIN = config['TWIST_GAIN']
    _DRIFT_GAIN = config['DRIFT_GAIN']

    # TODO Implement hand tracker thread and subscriber for hand motion tracker lookup

    def __init__(self, config, data_plot, motion_tracker, dominant_hand) -> None:

        self._data_plot = data_plot
        self._dominant_hand = dominant_hand
        self._motion_tracker = motion_tracker

        if self._motion_tracker:
            self._hand_gesture = {
                'Left': None,
                'Right': None
            }
            self._hand_markers_sub = rospy.Subscriber(
                '/hand_markers', MarkerArray, self._hand_markers_callback)
            self._left_twist_sub = rospy.Subscriber(
                '/Left_hand_twist', TwistStamped, self._left_twist_callback)
            self._right_twist_sub = rospy.Subscriber(
                '/Right_hand_twist', TwistStamped, self._right_twist_callback)
        else:
            self.joystick = jsk(motion_tracker=self._motion_tracker)

        self.controller = PR2Controller(rate=BMCP._CONTROL_RATE,
                                        config=config,
                                        data_plotter=self._data_plot)

        self.controller.set_manip_thresh(BMCP._MANIP_THRESH)
        self.controller.move_to_neutral()
        self.controller.move_head_to([0.0, 0.2])

        self._right_arm = self.controller.get_arm_controller('r')
        self._left_arm = self.controller.get_arm_controller('l')
        rospy.loginfo('Robot is in neutral position')
        rospy.sleep(1)

        # State variables
        self._constraint_is_set = False
        self._state = 'individual'

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

        # self._prev_e = np.asarray([0,0])
        # self._hand_tracker_thread = threading.Thread(
        #     target=self.hand_tracker_handler)

    def switch_to_individual_control(self):
        self._state = 'individual'

    def switch_to_central_control(self):
        self._state = 'central'

    def stop(self):
        self._state = 'Done'

    def hand_tracker_handler(self):
        normalized_vec = True
        kp = 0.5
        kd = 0.2
        while self._state != 'Done':
            # x, y, z = self.camera.get_wrist_point(side='Right', normalized=normalized_vec)
            # if normalized_vec:
            #     gesture = self.camera.get_gesture()
            #     if gesture != []:
            #         if gesture[0][0].category_name == 'Closed_Fist':
            #             cur_e = np.asarray([-x, y])
            #             cmd_vel = kp*cur_e + kd*(cur_e - self._prev_e)
            #             self._prev_e = cur_e
            #             self.controller.move_head(cmd_vel)
            # else:
            #     # print([x, y, z])
            #     pass

            pass

    def _hand_markers_callback(self, msg : MarkerArray):
        # self._hand_markers = msg.markers
        for marker in msg.markers:
            # if marker.ns == 'Left':
            self._hand_gesture[marker.ns] = marker.text
            # elif marker.ns == 'Right':
            #     self._right_hand_gesture = marker.text

    def _left_twist_callback(self, msg: TwistStamped):
        self._left_twist = msg.twist

    def _right_twist_callback(self, msg: TwistStamped):
        self._right_twist = msg.twist

    # Initial teleoperation function with XBOX joystick
    def teleop_test(self):
        r"""
        Initial teleoperation function with XBOX joystick
        """
        rospy.loginfo('Start teleop using joystick')
        # constraint_is_set = False

        # TODO: Swap mapping for gesture to be used
        if not self._motion_tracker:
            self._dead_switch_index = self.joystick.dead_switch_index
            self._system_halt_index = self.joystick.system_halt_index
            self._right_arm_index = self.joystick.right_arm_index
            self._left_arm_index = self.joystick.left_arm_index
            self._gripper_open_index = self.joystick.gripper_open_index
            self._gripper_close_index = self.joystick.gripper_close_index
            self._trigger_constraint_index = self.joystick.trigger_constraint_index

        self.switch_to_central_control()
        self.controller.start_jg_vel_controller()
        self._control_signal_thread.start()
        # self._base_controller_thread.start()

        rospy.sleep(1)

        while not rospy.is_shutdown():

            start_time = time.perf_counter()

            qdot = np.zeros(14)

            if self._motion_tracker:
                # self.teleop_hand_gestures()
                qdot = self.process_hand_gesture_commands(qdot)
            else:
                qdot = self.teleop_joystick(qdot)

            self._qdot_right = qdot[7:]
            self._qdot_left = qdot[:7]

            exec_time = time.perf_counter() - start_time
            if exec_time > 1/BMCP._CONTROL_RATE:
                rospy.logwarn( f'Calculation time exceeds control rate: {exec_time:.4f}')
            rospy.logdebug( f'Calculation time: {exec_time:.4f}')

            # joy_msg = self.joystick.get_joy_msg()

            # if joy_msg[1][self._system_halt_index]:

            #     self.stop()
            #     rospy.loginfo('Done teleoperation.')
            #     rospy.signal_shutdown('Done')

            #     self._control_signal_thread.join()
            #     rospy.loginfo('Control signal thread joined.')
            #     self._base_controller_thread.join()
            #     rospy.loginfo('Base controller thread joined.')

            #     if self._data_plot:
            #         self._data_recording_thread.join()
            #         rospy.loginfo('Data recording thread joined.')

            #     os.system(
            #         'rosnode kill /hand_tracker') if self._motion_tracker else os.system('rosnode kill /joy')

            # # Set the kinematic constraints for BMCP and start joint group velocity controller
            # # DO NOT combine this with the below if statement for not setting the constraint
            # if (joy_msg[1][self._trigger_constraint_index[0]] * joy_msg[1][self._trigger_constraint_index[1]]) and not constraint_is_set:

            #     constraint_is_set, _, constraint_distance = self.controller.set_kinematics_constraints()
            #     if self._data_plot:
            #         self.controller.store_constraint_distance(
            #             constraint_distance)
            #         self._data_recording_thread.start()
            #     rospy.loginfo(
            #         'Constraint is set, switching controllers, started velocity controller thread')

            # # TODO: Add twist extraction method for handtracker
            # # Once constraint is set, start the teleoperation using
            # twist, _ = self.joystick.motion_to_twist(
            #     BMCP._TWIST_GAIN) if self._motion_tracker else self.joystick.joy_to_twist(BMCP._TWIST_GAIN)

            # if constraint_is_set:

            #     # Exrtact the twist from the joystick message
            #     twist_left = self.controller.get_twist_in_tool_frame(
            #         side='l', twist=twist)
            #     twist_right = self.controller.get_twist_in_tool_frame(
            #         side='r', twist=twist)

            #     # RT trigger to allow control signal to be sent
            #     if joy_msg[0][self._dead_switch_index] != 1:

            #         # Extract the Jacobians in the middle frame using the virtual robot with joint states data from the real robot
            #         jacob_right = self.controller.get_jacobian(side='r')
            #         jacob_left = self.controller.get_jacobian(side='l')
            #         jacob_constraint = np.c_[jacob_left, -jacob_right]

            #         # Calculate the joint velocities using RMRC
            #         qdot_right = CalcFuncs.rmrc(
            #             jacob_right, twist_right, w_thresh=BMCP._MANIP_THRESH)
            #         qdot_left = CalcFuncs.rmrc(
            #             jacob_left, twist_left,  w_thresh=BMCP._MANIP_THRESH)
            #         qdot_combined = np.r_[qdot_left, qdot_right]

            #         # Perform nullspace projection for qdot_combined on constraint Jacobian to ensure the twist synchronisatio
            #         taskspace_drift_compensation = self.controller.task_drift_compensation(gain_p=BMCP._DRIFT_GAIN['p'],
            #                                                                                gain_d=BMCP._DRIFT_GAIN['d'],
            #                                                                                on_taskspace=True) * 2

            #         # Combine the primary and secondary tasks velocities
            #         primary_tasks_vel = np.linalg.pinv(
            #             jacob_constraint) @ taskspace_drift_compensation
            #         secondary_tasks_vel = CalcFuncs.nullspace_projector(
            #             jacob_constraint) @ qdot_combined
            #         qdot = primary_tasks_vel + secondary_tasks_vel

            #         # Add a joint limits damper to the joint velocities
            #         qdot += self.controller.joint_limit_damper(
            #             qdot, steepness=BMCP._DAMPER_STEEPNESS)

            # else:
            #     # TODO: consider setting up the separated controller thread for each arm in this mode
            #     if joy_msg[1][self._right_arm_index]:  # left bumper
            #         if joy_msg[0][self._dead_switch_index] != 1:
            #             qdot[7:] = self.controller.process_arm_movement(side='r',
            #                                                             twist=twist,
            #                                                             manip_thresh=BMCP._MANIP_THRESH,
            #                                                             damper_steepness=BMCP._DAMPER_STEEPNESS)
            #     if joy_msg[1][self._left_arm_index]:  # right bumper
            #         if joy_msg[0][self._dead_switch_index] != 1:
            #             qdot[:7] = self.controller.process_arm_movement(side='l',
            #                                                             twist=twist,
            #                                                             manip_thresh=BMCP._MANIP_THRESH,
            #                                                             damper_steepness=BMCP._DAMPER_STEEPNESS)

            # self._qdot_right = qdot[7:]
            # self._qdot_left = qdot[:7]

            # exec_time = time.perf_counter() - start_time
            # if exec_time > 1/BMCP._CONTROL_RATE:
            #     rospy.logwarn( f'Calculation time exceeds control rate: {exec_time:.4f}')
            # rospy.logdebug( f'Calculation time: {exec_time:.4f}')

    def teleop_joystick(self, qdot):

        joy_msg = self.joystick.get_joy_msg()

        if joy_msg[1][self._system_halt_index]:
            self.stop_teleop()

        if (joy_msg[1][self._trigger_constraint_index[0]] * joy_msg[1][self._trigger_constraint_index[1]]) and not self.constraint_is_set:
            self.constraint_is_set, _, constraint_distance = self.controller.set_kinematics_constraints()
            if self._data_plot:
                self.controller.store_constraint_distance(constraint_distance)
                self._data_recording_thread.start()
            rospy.loginfo('Constraint is set, switching controllers, started velocity controller thread')

        twist, _ = self.joystick.joy_to_twist(BMCP._TWIST_GAIN)
        if joy_msg[0][self._dead_switch_index] != 1:
            if self.constraint_is_set:
                qdot = self.handle_constrained_twist(qdot, twist)
            else:
                qdot = self.handle_individual_arm_control(qdot, twist, joy_msg) 
        else:
            return np.zeros(14)

        return qdot

    def teleop_hand_gestures(self):
        # Handle state transitions based on gestures
        if self._hand_gesture['Left'] == 'Pointing_Up' and self._hand_gesture['Right'] == 'Pointing_Up' :
            self.stop_teleop()

        if self._hand_gesture['Left'] == 'Thumb_Up' and self._hand_gesture['Right'] == 'Thumb_Up' and not self.constraint_is_set:
            self.constraint_is_set, _, constraint_distance = self.controller.set_kinematics_constraints()
            if self._data_plot:
                self.controller.store_constraint_distance(constraint_distance)
                self._data_recording_thread.start()
            rospy.loginfo('Switching to coordination mode')


    def process_hand_gesture_commands(self, qdot):
        self.teleop_hand_gestures()
        twist_left = self.get_twist_from_hand('Left')
        twist_right = self.get_twist_from_hand('Right')

        if self._constraint_is_set:
            object_twist = twist_right if self._dominant_hand == 'Right' else twist_left
            if self._hand_gesture[self._dominant_hand] == 'Closed_Palm':
                qdot = self.handle_constrained_twist(qdot, object_twist)
            else:
                return np.zeros(14)
        
        else:
            qdot = self.handle_indiv_arm_control_ges(qdot, twist_left, twist_right)

        return qdot

    def handle_constrained_twist(self, qdot, object_twist):

        twist_left = self.controller.get_twist_in_tool_frame(side='l', twist=object_twist)
        twist_right = self.controller.get_twist_in_tool_frame(side='r', twist=object_twist)

        # Extract the Jacobians in the middle frame using the virtual robot with joint states data from the real robot
        jacob_right = self.controller.get_jacobian(side='r')
        jacob_left = self.controller.get_jacobian(side='l')
        jacob_constraint = np.c_[jacob_left, -jacob_right]

        # Calculate the joint velocities using RMRC
        qdot_right = CalcFuncs.rmrc(jacob_right, twist_right, w_thresh=BMCP._MANIP_THRESH)
        qdot_left = CalcFuncs.rmrc(jacob_left, twist_left,  w_thresh=BMCP._MANIP_THRESH)
        qdot_combined = np.r_[qdot_left, qdot_right]

        # Perform nullspace projection for qdot_combined on constraint Jacobian to ensure the twist synchronisatio
        taskspace_drift_compensation = self.controller.task_drift_compensation(gain_p=BMCP._DRIFT_GAIN['p'],
                                                                               gain_d=BMCP._DRIFT_GAIN['d'],
                                                                               on_taskspace=True) * 2

        # Combine the primary and secondary tasks velocities
        primary_tasks_vel = np.linalg.pinv(jacob_constraint) @ taskspace_drift_compensation
        secondary_tasks_vel = CalcFuncs.nullspace_projector(jacob_constraint) @ qdot_combined
        qdot = primary_tasks_vel + secondary_tasks_vel

        # Add a joint limits damper to the joint velocities
        qdot += self.controller.joint_limit_damper(qdot, steepness=BMCP._DAMPER_STEEPNESS)

        return qdot


    def handle_individual_arm_control(self, qdot, twist, joy_msg):
        if joy_msg[1][self._right_arm_index]:  # left bumper
            qdot[7:] = self.controller.process_arm_movement(
                side='r', twist=twist, manip_thresh=BMCP._MANIP_THRESH, damper_steepness=BMCP._DAMPER_STEEPNESS)
        if joy_msg[1][self._left_arm_index]:  # right bumper
            qdot[:7] = self.controller.process_arm_movement(
                side='l', twist=twist, manip_thresh=BMCP._MANIP_THRESH, damper_steepness=BMCP._DAMPER_STEEPNESS)

        return qdot

    def handle_indiv_arm_control_ges(self, qdot, twist_left, twist_right):
        if self._hand_gesture['Left'] == 'closed_palm':
            qdot[:7] = self.controller.process_arm_movement(
                side='l', twist=twist_left, manip_thresh=BMCP._MANIP_THRESH, damper_steepness=BMCP._DAMPER_STEEPNESS)
        if self._hand_gesture['Right'] == 'closed_palm':
            qdot[7:] = self.controller.process_arm_movement(
                side='r', twist=twist_right, manip_thresh=BMCP._MANIP_THRESH, damper_steepness=BMCP._DAMPER_STEEPNESS)

        return qdot

    # TODO: Add method for gesture handling in gripper control
    def handle_gripper(self, arm, joy_msg):

        if self._motion_tracker:
            if joy_msg[1][self._gripper_open_index] == 1:
                arm.open_gripper()
            elif joy_msg[1][self._gripper_close_index] == 1:
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

    def base_controller_handler(self):

        while self._state != 'Done':

            joy_msg = self.joystick.get_joy_msg()

            if joy_msg[1][self._right_arm_index]:  # left bumper
                self.handle_gripper(self._right_arm, joy_msg)

            if joy_msg[1][self._left_arm_index]:  # right bumper
                self.handle_gripper(self._left_arm, joy_msg)

            twist = np.zeros(6)
            if joy_msg[0][2] != 1:  # left trigger for base controller
                twist, _ = self.joystick.joy_to_twist(
                    BMCP._TWIST_GAIN, base=True)

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

    def get_twist_from_hand(self, side):
        return self._right_twist if side == 'Right' else self._left_twist
    
    def stop_teleop(self):
        self.stop()
        rospy.loginfo('Done teleoperation.')
        rospy.signal_shutdown('Done')

        self._control_signal_thread.join()
        rospy.loginfo('Control signal thread joined.')
        # self._base_controller_thread.join()
        # rospy.loginfo('Base controller thread joined.')

        if self._data_plot:
            self._data_recording_thread.join()
            rospy.loginfo('Data recording thread joined.')

        os.system('rosnode kill /hand_tracker') if self._motion_tracker else os.system('rosnode kill /joy')


if __name__ == "__main__":
    try:
        rospy.init_node('bimanual_controller', log_level=2, anonymous=True)
        data_plot = rospy.get_param('~data_plot', True)
        motion_tracker = rospy.get_param('~motion_tracker', False)
        dominant_hand = rospy.get_param('~dominant_hand', 'Right')

        b = BMCP(config=config, data_plot=data_plot)
        b.teleop_test()
    except rospy.ROSInterruptException:
        pass
