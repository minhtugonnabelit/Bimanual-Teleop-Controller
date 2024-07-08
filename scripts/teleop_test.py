#!/usr/bin/env python3

import numpy as np
import threading
import time
import os

import rospy
from bimanual_controller.utility import *
from bimanual_controller.math_utils import CalcFuncs
from bimanual_controller.pr2_controller import PR2Controller
from bimanual_controller.joystick_controller import JoystickController as jsk


class BMCP:

    _DAMPER_STEEPNESS = config['DAMPER_STEEPNESS']
    _MANIP_THRESH = config['MANIPULABILITY_THRESHOLD']
    _CONTROL_RATE = config['CONTROL_RATE']
    _TWIST_GAIN = config['TWIST_GAIN']
    _DRIFT_GAIN = config['DRIFT_GAIN']

    def __init__(self, config, data_plot) -> None:

        self._data_plot = data_plot
        self._motion_tracker = rospy.get_param('~motion_tracker', False)
        self.joystick = jsk(motion_tracker=self._motion_tracker)
        self.controller = PR2Controller(
            rate=BMCP._CONTROL_RATE, joystick=self.joystick, config=config, data_plotter=self._data_plot)
        self.controller.set_manip_thresh(BMCP._MANIP_THRESH)
        self.controller.move_to_neutral()
        self._right_arm = self.controller.get_arm_controller('r')
        self._left_arm = self.controller.get_arm_controller('l')
        rospy.loginfo('Robot is in neutral position')
        rospy.sleep(3)

        # State variables
        self._constraint_is_set = False
        self._state = 'individual'

        self._qdot_right = np.zeros(7)
        self._qdot_left = np.zeros(7)

        self._control_signal_ready = threading.Condition()
        self._control_signal_thread = threading.Thread(
            target=self.control_signal_handler)
        self._base_controller_thread = threading.Thread(
            target=self.base_controller_handler)
        self._data_recording_thread = threading.Thread(
            target=self.data_recording_handler)

    def switch_to_individual_control(self):
        self._state = 'individual'

    def switch_to_central_control(self):
        self._state = 'central'

    def stop(self):
        self._state = 'Done'

    # Initial teleoperation function with XBOX joystick
    def teleop_test(self):
        r"""
        Initial teleoperation function with XBOX joystick
        """
        rospy.loginfo('Start teleop using joystick')
        constraint_is_set = False
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
        self._base_controller_thread.start()

        rospy.sleep(1)

        while not rospy.is_shutdown():

            start_time = time.perf_counter()

            qdot = np.zeros(14)
            joy_msg = self.joystick.get_joy_msg()

            if joy_msg[1][self._system_halt_index]:

                self.stop()
                rospy.loginfo('Done teleoperation.')
                rospy.signal_shutdown('Done')

                self._control_signal_thread.join()
                rospy.loginfo('Control signal thread joined.')
                self._base_controller_thread.join()
                rospy.loginfo('Base controller thread joined.')
                if self._data_plot:
                    self._data_recording_thread.join()
                    rospy.loginfo('Data recording thread joined.')

                os.system(
                    'rosnode kill /razer_hydra_driver') if self._motion_tracker else os.system('rosnode kill /joy_node')

            # Set the kinematic constraints for BMCP and start joint group velocity controller
            # DO NOT combine this with the below if statement for not setting the constraint
            if (joy_msg[1][self._trigger_constraint_index[0]] * joy_msg[1][self._trigger_constraint_index[1]]) and not constraint_is_set:

                constraint_is_set, _, constraint_distance = self.controller.set_kinematics_constraints()
                if self._data_plot:
                    self.controller.store_constraint_distance(
                        constraint_distance)
                    self._data_recording_thread.start()
                rospy.loginfo(
                    'Constraint is set, switching controllers, started velocity controller thread')

            # Once constraint is set, start the teleoperation using
            twist, _ = self.joystick.motion_to_twist(
                BMCP._TWIST_GAIN) if self._motion_tracker else self.joystick.joy_to_twist(BMCP._TWIST_GAIN)

            if constraint_is_set:

                # Exrtact the twist from the joystick message
                twist_left = self.controller.get_twist_in_tool_frame(
                    side='l', twist=twist)
                twist_right = self.controller.get_twist_in_tool_frame(
                    side='r', twist=twist)

                # RT trigger to allow control signal to be sent
                if joy_msg[0][self._dead_switch_index] != 1:

                    # Extract the Jacobians in the middle frame using the virtual robot with joint states data from the real robot
                    jacob_right = self.controller.get_jacobian(side='r')
                    jacob_left = self.controller.get_jacobian(side='l')
                    jacob_constraint = np.c_[jacob_left, -jacob_right]

                    # Calculate the joint velocities using RMRC
                    qdot_right = CalcFuncs.rmrc(
                        jacob_right, twist_right, w_thresh=BMCP._MANIP_THRESH)
                    qdot_left = CalcFuncs.rmrc(
                        jacob_left, twist_left,  w_thresh=BMCP._MANIP_THRESH)
                    qdot_combined = np.r_[qdot_left, qdot_right]

                    # Perform nullspace projection for qdot_combined on constraint Jacobian to ensure the twist synchronisatio
                    taskspace_drift_compensation = self.controller.task_drift_compensation(gain_p=BMCP._DRIFT_GAIN['p'],
                                                                                           gain_d=BMCP._DRIFT_GAIN['d'],
                                                                                           on_taskspace=True) * 2

                    # Combine the primary and secondary tasks velocities
                    primary_tasks_vel = np.linalg.pinv(
                        jacob_constraint) @ taskspace_drift_compensation
                    secondary_tasks_vel = CalcFuncs.nullspace_projector(
                        jacob_constraint) @ qdot_combined
                    qdot = primary_tasks_vel + secondary_tasks_vel

                    # Add a joint limits damper to the joint velocities
                    qdot += self.controller.joint_limit_damper(
                        qdot, steepness=BMCP._DAMPER_STEEPNESS)

            else:

                if joy_msg[1][self._right_arm_index]:  # left bumper
                    if joy_msg[0][self._dead_switch_index] != 1:
                        qdot[7:] = self.controller.process_arm_movement(side='r',
                                                                        twist=twist,
                                                                        manip_thresh=BMCP._MANIP_THRESH,
                                                                        damper_steepness=BMCP._DAMPER_STEEPNESS)
                if joy_msg[1][self._left_arm_index]:  # right bumper
                    if joy_msg[0][self._dead_switch_index] != 1:
                        qdot[:7] = self.controller.process_arm_movement(side='l',
                                                                        twist=twist,
                                                                        manip_thresh=BMCP._MANIP_THRESH,
                                                                        damper_steepness=BMCP._DAMPER_STEEPNESS)

            self._qdot_right = qdot[7:]
            self._qdot_left = qdot[:7]

            exec_time = time.perf_counter() - start_time
            if exec_time > 1/BMCP._CONTROL_RATE:
                rospy.logwarn(
                    f'Calculation time exceeds control rate: {exec_time:.4f}')
            rospy.logdebug(
                f'Calculation time: {exec_time:.4f}')

    def handle_gripper(self, arm, joy_msg):
        if self._motion_tracker:
            if joy_msg[1][self._gripper_open_index] == 1:
                arm.open_gripper()
            elif joy_msg[1][self._gripper_close_index] == 1:
                arm.close_gripper()
        else:
            if joy_msg[0][-1] == 1:  # up
                arm.open_gripper()
            elif - joy_msg[0][-1] == 1:  # down
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
                self.controller.store_joint_velocities(
                    'right', self._qdot_right)
                self.controller.store_joint_velocities('left', self._qdot_left)

                self.controller.store_manipulability()
                self.controller.store_drift()


if __name__ == "__main__":
    try:
        rospy.init_node('bimanual_controller', log_level=2, anonymous=True)
        data_plot = rospy.get_param('~data_plot', True)

        b = BMCP(config=config, data_plot=data_plot)
        b.teleop_test()
    except rospy.ROSInterruptException:
        pass
