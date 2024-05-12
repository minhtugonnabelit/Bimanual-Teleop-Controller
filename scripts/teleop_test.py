#!/usr/bin/env python3

import numpy as np
import time

import rospy
from sensor_msgs.msg import Joy

from bimanual_controller.utility import (
    CalcFuncs, CONTROL_RATE, TWIST_GAIN, joy_to_twist)
from bimanual_controller.pr2_controller import PR2Controller
import threading
import copy


class BMCP:

    def __init__(self) -> None:

        self.controller = PR2Controller(
            name='teleop_test', log_level=2, rate=CONTROL_RATE)
        
        self.controller.set_manip_thresh(0.1)
        self.controller.move_to_neutral()
        rospy.loginfo('Neutral position reached.')
        rospy.sleep(5)

        # State variables

        self._constraint_is_set = False
        self._state = 'individual'

        # Control signals and locks

        self._qdot_right = np.zeros(7)
        self._qdot_right_lock = threading.Lock()

        self._qdot_left = np.zeros(7)
        self._qdot_left_lock = threading.Lock()

        self.control_signal_thread = threading.Thread(target=self.control_signal_handler)
        
    # State machine functions

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

        # State variables
        constraint_is_set = False
        done = False
        self.switch_to_central_control()
        rospy.loginfo('Start teleop using joystick')
        rospy.wait_for_message('/joy', Joy)

        while not done:

            start_time = time.perf_counter()
            # start_time = rospy.Time.now()

            # ---------------------- #
            # ---------------------- #

            qdot = np.zeros(14)

            joy_msg = self.controller.get_joy_msg()

            # Trigger the controller to move to neutral position
            if joy_msg[1][-3]:
                rospy.loginfo('Moving to neutral position')
                self.controller.move_to_neutral()

            if joy_msg[1][4]:
                if joy_msg[1][6]:
                    self.controller.left_arm.open_gripper()
                    # self.controller.open_gripper(side='left')
                elif joy_msg[1][7]:
                    self.controller.left_arm.close_gripper()
                    # self.controller.close_gripper(side='left')

            if joy_msg[1][5]:
                if joy_msg[1][6]:
                    self.controller.right_arm.open_gripper()
                    # self.controller.open_gripper(side='right')
                elif joy_msg[1][7]:
                    self.controller.right_arm.close_gripper()
                    # self.controller.close_gripper(side='right')

            # Set the kinematic constraints for BMCP and start joint group velocity controller
            if (joy_msg[1][-1] * joy_msg[1][-2]) and not constraint_is_set:

                self.controller.start_jg_vel_controller()
                rospy.sleep(1)
                constraint_is_set, _, constraint_distance = self.controller.set_kinematics_constraints()
                self.controller.store_constraint_distance(constraint_distance)
                self.control_signal_thread.start()
                rospy.loginfo(
                    'Constraint is set, switching controllers, started velocity controller thread')

            # Once constraint is set, start the teleoperation using
            if constraint_is_set:

                # Exrtact the twist from the joystick message
                twist, done = joy_to_twist(joy_msg, TWIST_GAIN)
                rospy.logdebug(f'Twist: {twist}')

                if joy_msg[1][5]:  # Safety trigger to allow control signal to be sent

                    # Extract the Jacobians in the middle frame using the virtual robot with joint states data from the real robot
                    jacob_right = self.controller.get_jacobian(side='r')
                    jacob_left = self.controller.get_jacobian(side='l')
                    jacob_constraint = np.c_[jacob_left, -jacob_right]

                    # Calculate the joint velocities using RMRC
                    qdot_right = CalcFuncs.rmrc(
                        jacob_right, twist, w_thresh=0.07)
                    qdot_left = CalcFuncs.rmrc(
                        jacob_left, twist,  w_thresh=0.07)
                    qdot_combined = np.r_[qdot_left, qdot_right]

                    qdot = qdot_combined

                    # Perform nullspace projection for qdot_combined on constraint Jacobian to ensure the twist synchronisation
                    qdot = CalcFuncs.nullspace_projector(
                        jacob_constraint) @ qdot_combined

                    # @TODO: Joint limits avoidance to be added here through nullspace filtering

                    # Drift compensation for the joint group velocity controller
                    drift = self.controller.get_drift_compensation()
                    qdot += drift

                # # Control signal send from this block
                self._qdot_right = qdot[7:]
                self._qdot_left = qdot[:7]

            # ---------------------- #

            if done:
                rospy.loginfo('Done teleoperation.')
                rospy.signal_shutdown('Done')
                self.control_signal_thread.join()
                rospy.loginfo('Control signal thread joined.')

            # ---------------------- #
            # ---------------------- #

            exec_time = time.perf_counter() - start_time
            rospy.logdebug(
                f'Calculation time: {exec_time:.4f}')

            time.sleep(1/(CONTROL_RATE*10))

    def teleop_with_hand_motion(self):

        r"""
        Similar system to the coordination part of teleop_test, except for the beginning already start with
        joint_group_vel_controller vel cmd input from montion controller interface instead of 
        letting trajectory-liked motion to set the initial state for setting the constraint
        """

        # State variables
        last_switch_time = 0

        rospy.loginfo('Start teleop using Razer Hydra Controller')

        jgvc_started = self.controller.start_jg_vel_controller()
        if not jgvc_started:
            rospy.logerr('Failed to start joint group velocity controller')
            return
        
        def check_state(last_switch_time):

            left_joy_msg = self.controller.get_hydra_joy_msg(side='l')
            right_joy_msg = self.controller.get_hydra_joy_msg(side='r')

            debounced, last_switch_time = BMCP.debounce(last_switch_time)
            if debounced:

                if left_joy_msg[1][0] and right_joy_msg[1][0]:
                    if self._state == 'central':
                        self._constraint_is_set = False  # Reset the constraint condition
                        self.switch_to_individual_control()
                    else:
                        self.switch_to_central_control()

                elif left_joy_msg[1][1] and right_joy_msg[1][1]: 
                    self.stop()  

            return last_switch_time
        
        # Wait for the controller to start
        rospy.sleep(1)  
        self.control_signal_thread.start()

        while not rospy.is_shutdown() and self._state != 'Done':

            if self._state == 'individual':

                rospy.loginfo('Switching to individual control')

                # Start individual control threads
                left_thread = threading.Thread(target=self.hand_controller, args=('l'))
                right_thread = threading.Thread(target=self.hand_controller, args=('r'))
                left_thread.start()
                right_thread.start()

                # Wait for state change
                while self._state == 'individual':

                    last_switch_time = check_state(last_switch_time)
                    rospy.sleep(0.1)

                # Stop individual control threads
                left_thread.join()
                right_thread.join()

            elif self._state == 'central':

                rospy.loginfo('Switching to central control')

                # Start central control thread
                central_thread = threading.Thread(target=self.central_controller)
                central_thread.start()

                # Wait for state change
                while self._state == 'central':

                    last_switch_time = check_state(last_switch_time)
                    rospy.sleep(0.1)

                # Stop central control thread
                central_thread.join()

            elif self._state == "Done":
                rospy.loginfo('Done teleoperation.')
                rospy.signal_shutdown('Done')
                self.control_signal_thread.join()
                rospy.loginfo('Control signal thread joined.')

            else:
                raise ValueError('Invalid state')

    def hand_controller(self, side):

        synced = False
        arm = self.controller.right_arm if side == 'r' else self.controller.left_arm
        qdot_lock = self._qdot_right_lock if side == 'r' else self._qdot_left_lock

        while self._state == 'individual':

            qd = np.zeros(7)
            joy_msg = self.controller.get_hydra_joy_msg(side=side)
            twist_msg, synced = self.controller.get_twist(side=side, synced=synced,  gain=TWIST_GAIN)

            if not synced:
                continue

            if joy_msg[1][-2]:  # Safety trigger to allow control signal to be sent
                jacob = self.controller.get_jacobian(side=side)
                qd = CalcFuncs.rmrc(
                    jacob, twist_msg, w_thresh=0.1)

            with qdot_lock: 
                if side == 'r': self._qdot_right = copy.deepcopy(qd)
                else: self._qdot_left = copy.deepcopy(qd)

            if joy_msg[1][3]: arm.open_gripper()
            if joy_msg[1][2]: arm.close_gripper()

            rospy.sleep(0.02)

    def central_controller(self):

        synced = False

        if not self._constraint_is_set:

            self._constraint_is_set, _, constraint_distance = self.controller.set_kinematics_constraints()
            self.controller.store_constraint_distance(constraint_distance)
            rospy.loginfo(
                'Constraint is set, switching controllers, started velocity controller thread')

        while self._state == 'central':

            qdot = np.zeros(14)

            joy_msg = self.controller.get_hydra_joy_msg(side='r')
            twist_msg, synced = self.controller.get_twist(side='r', synced=synced, gain=TWIST_GAIN)

            if joy_msg[1][-2]:

                # Extract the Jacobians in the middle frame using the virtual robot with joint states data from the real robot
                jacob_right = self.controller.get_jacobian(side='r')
                jacob_left = self.controller.get_jacobian(side='l')
                jacob_constraint = np.c_[jacob_left, -jacob_right]

                # Calculate the joint velocities using RMRC
                qdot_right = CalcFuncs.rmrc(jacob_right, twist_msg, w_thresh=0.07)
                qdot_left = CalcFuncs.rmrc(jacob_left, twist_msg,  w_thresh=0.07)
                qdot_combined = np.r_[qdot_left, qdot_right]

                # Perform nullspace projection for qdot_combined on constraint Jacobian to ensure the twist synchronisation
                qdot = CalcFuncs.nullspace_projector(jacob_constraint) @ qdot_combined

                # @TODO: Joint limits avoidance to be added here through nullspace filtering

                # Drift compensation for the joint group velocity controller
                drift = self.controller.get_drift_compensation()
                qdot += drift

            # # Control signal send from this block
            with self._qdot_right_lock: self._qdot_right = qdot[7:]
            with self._qdot_left_lock: self._qdot_left = qdot[:7]

            if joy_msg[1][3]:
                self.controller.left_arm.open_gripper()
            elif joy_msg[1][2]:
                self.controller.left_arm.close_gripper()

            rospy.sleep(1/(CONTROL_RATE*10))

    def control_signal_handler(self):

        while self._state != 'Done':

            if self._state != 'individual':

                self.controller.store_manipulability()
                self.controller.store_drift()

            self.controller.right_arm.send_joint_command(joint_command=self._qdot_right)
            self.controller.left_arm.send_joint_command(joint_command=self._qdot_left)
            self.controller.sleep()

    @staticmethod
    def debounce(last_switch_time, debounce_interval=1):
        current_time = time.time()
        if current_time - last_switch_time >= debounce_interval:
            last_switch_time = current_time
            return True, last_switch_time
        else:
            return False, last_switch_time


    # def right_hand_controller(self):

    #     synced = False
    #     while self._state == 'individual':

    #         qdot = np.zeros(7)
    #         joy_msg = self.controller.get_hydra_joy_msg(side='r')
    #         twist_msg, synced = self.controller.get_twist(side='r', synced=synced,  gain=TWIST_GAIN)

    #         if not synced:
    #             continue

    #         if joy_msg[1][-2]:  # Safety trigger to allow control signal to be sent
    #             jacob = self.controller.get_jacobian(side='r')
    #             qdot = CalcFuncs.rmrc(
    #                 jacob, twist_msg, w_thresh=0.1)

    #         with self._qdot_right_lock: self._qdot_right = copy.deepcopy(qdot)

    #         if joy_msg[1][3]: self.controller.right_arm.open_gripper()
    #         if joy_msg[1][2]: self.controller.right_arm.close_gripper()

    #         rospy.sleep(0.02)

    # def left_hand_controller(self):

    #     synced = False

    #     while self._state == 'individual':

    #         qdot_left = np.zeros(7)
    #         joy_msg = self.controller.get_hydra_joy_msg(side='l')
    #         twist_msg, synced = self.controller.get_twist(side='l', synced=synced,  gain=TWIST_GAIN)
    #         if not synced:
    #             continue

    #         if joy_msg[1][-2]:
    #             jacob_left = self.controller.get_jacobian(side='l')
    #             qdot_left = CalcFuncs.rmrc(
    #                 jacob_left, twist_msg, w_thresh=0.1)

    #         with self._qdot_left_lock: self._qdot_left = copy.deepcopy(qdot_left)

    #         if joy_msg[1][3]: self.controller.left_arm.open_gripper()
    #         if joy_msg[1][2]: self.controller.left_arm.close_gripper()

    #         rospy.sleep(0.02)   



# def main():

    # controller = PR2Controller(
    #     name='teleop_test', log_level=1, rate=CONTROL_RATE)

    # rospy.loginfo('Start teleop using joystick')
    # rospy.wait_for_message('/joy', Joy)

    # # State variables
    # constraint_is_set = False
    # done = False

    # while not done:

    #     start_time = time.perf_counter()
    #     # start_time = rospy.Time.now()

    #     # ---------------------- #
    #     # ---------------------- #

    #     qdot = np.zeros(14)

    #     joy_msg = controller.get_joy_msg()

    #     # Trigger the controller to move to neutral position
    #     if joy_msg[1][-3]:
    #         controller.move_to_neutral()

    #     if joy_msg[1][4]:
    #         if joy_msg[1][6]:
    #             controller.open_gripper(side='left')
    #         elif joy_msg[1][7]:
    #             controller.close_gripper(side='left')

    #     if joy_msg[1][5]:
    #         if joy_msg[1][6]:
    #             controller.open_gripper(side='right')
    #         elif joy_msg[1][7]:
    #             controller.close_gripper(side='right')

    #     # Set the kinematic constraints for BMCP and start joint group velocity controller
    #     if (joy_msg[1][-1] * joy_msg[1][-2]) and not constraint_is_set:

    #         controller.start_jg_vel_controller()
    #         rospy.sleep(1)
    #         constraint_is_set, _, constraint_distance = controller.set_kinematics_constraints()
    #         controller.store_constraint_distance(constraint_distance)
    #         rospy.loginfo('Constraint is set, switching controllers')

    #     # Once constraint is set, start the teleoperation using
    #     if constraint_is_set:

    #         # Exrtact the twist from the joystick message
    #         twist, done = joy_to_twist(joy_msg, TWIST_GAIN)
    #         rospy.logdebug(f'Twist: {twist}')

    #         if joy_msg[1][5]:  # Safety trigger to allow control signal to be sent

    #             # Extract the Jacobians in the middle frame using the virtual robot with joint states data from the real robot
    #             jacob_right = controller.get_jacobian(side='right')
    #             jacob_left = controller.get_jacobian(side='left')
    #             jacob_constraint = np.c_[jacob_left, -jacob_right]

    #             qdot_right = CalcFuncs.rmrc(jacob_right, twist, w_thresh=0.1)
    #             qdot_left = CalcFuncs.rmrc(jacob_left, twist,  w_thresh=0.1)
    #             qdot_combined = np.r_[qdot_left, qdot_right]

    #             # Perform nullspace projection for qdot_combined on constraint Jacobian to ensure the twist synchronisation
    #             qdot = CalcFuncs.nullspace_projector(
    #                 jacob_constraint) @ qdot_combined

    #             # Drift compensation for the joint group velocity controller
    #             drift = controller.get_drift_compensation()
    #             qdot += drift

    #         # Control signal send from this block
    #         controller.send_joint_velocities('right', qdot[7:])
    #         controller.send_joint_velocities('left', qdot[:7])

    #     # ---------------------- #s
    #     # Record the joints data

    #     controller.store_joint_velocities('right', qdot[7:])
    #     controller.store_joint_velocities('left', qdot[:7])
    #     controller.store_drift()

    #     # ---------------------- #

    #     if done:
    #         rospy.loginfo('Done teleoperation.')
    #         rospy.signal_shutdown('Done')

    #     # ---------------------- #
    #     # ---------------------- #

    #     exec_time = time.perf_counter() - start_time
    #     rospy.logdebug(
    #         f'Calculation time: {exec_time:.4f}')

    #     controller.sleep()

    #     # if exec_time < 1 / CONTROL_RATE:
    #     #     rospy.sleep(1/CONTROL_RATE - exec_time)

    #     total_time = time.perf_counter() - start_time
    #     rospy.logdebug(
    #         f'Total time: {total_time:.4f}')

    #     # exec_time = rospy.Time.now() - start_time
    #     # rospy.logdebug(
    #     #     f'Calculation time: {exec_time.to_sec():.4f}')

    #     # # Sleep to control the rate of the loop execution based on the control rate
    #     # if exec_time.to_sec() < 1 / CONTROL_RATE:
    #     #     rospy.sleep(1/CONTROL_RATE - exec_time.to_sec())

    #     # #total_time = time.time() - start_time
    #     # total_time = rospy.Time.now() - start_time
    #     # rospy.logdebug(
    #     #     f'Total time: {total_time.to_sec():.4f}')


if __name__ == "__main__":
    try:
        b = BMCP()
        # b.teleop_test()
        b.teleop_with_hand_motion()
    except rospy.ROSInterruptException:
        pass
