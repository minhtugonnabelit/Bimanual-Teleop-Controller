#!/usr/bin/env python3

import numpy as np
import threading, time
import os

import rospy
from bimanual_controller.utility import *
from bimanual_controller.math_utils import CalcFuncs
from bimanual_controller.pr2_controller import PR2Controller
from bimanual_controller.joystick_controller import JoystickController

class BMCP:

    # DAMPER_STEEPNESS = 5
    # MANIP_THRESH = 0.07
    # CONTROL_RATE = 50
    # TWIST_GAIN = [0.1, 0.1]
    # DRIFT_GAIN = {
    #     'p': [2,2,2,8,8,8],
    #     'd': [1,1,1,2,2,2]
    # }

    _DAMPER_STEEPNESS = config['DAMPER_STEEPNESS']
    _MANIP_THRESH = config['MANIPULABILITY_THRESHOLD']
    _CONTROL_RATE = config['CONTROL_RATE']
    _TWIST_GAIN = config['TWIST_GAIN']
    _DRIFT_GAIN = config['DRIFT_GAIN']

    def __init__(self, config) -> None:
        
        rospy.init_node('bimanual_controller', log_level=2, anonymous=True)

        self.joystick = JoystickController()
        self.controller = PR2Controller(rate=BMCP._CONTROL_RATE, joystick=self.joystick, config=config)
        self.controller.set_manip_thresh(BMCP._MANIP_THRESH)
        self.controller.move_to_neutral()
        rospy.loginfo('Robot is in neutral position')
        rospy.sleep(3)

        # State variables
        self._constraint_is_set = False
        self._state = 'individual'

        # Control signals and locks
        self._right_arm = self.controller.get_arm_controller('r')
        self._qdot_right = np.zeros(7)
        # self._qdot_right_lock = threading.Lock()

        self._left_arm = self.controller.get_arm_controller('l')
        self._qdot_left = np.zeros(7)
        # self._qdot_left_lock = threading.Lock()

        self._control_signal_ready = threading.Condition()
        self._control_signal_thread = threading.Thread(target=self.control_signal_handler)
        self._base_controller_thread = threading.Thread(target=self.base_controller_handler)
        self._data_recording_thread = threading.Thread(target=self.data_recording_handler)

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
        self.switch_to_central_control()
        self.controller.start_jg_vel_controller()
        self._control_signal_thread.start()
        self._base_controller_thread.start()

        rospy.sleep(1)

        while not rospy.is_shutdown():

            start_time = time.perf_counter()

            qdot = np.zeros(14)
            joy_msg = self.joystick.get_joy_msg()

            if joy_msg[1][-3]:

                self.stop()
                rospy.loginfo('Done teleoperation.')
                rospy.signal_shutdown('Done')

                self._control_signal_thread.join()
                rospy.loginfo('Control signal thread joined.')
                self._base_controller_thread.join()
                rospy.loginfo('Base controller thread joined.')
                self._data_recording_thread.join()
                rospy.loginfo('Data recording thread joined.')

                os.system('rosnode kill /joy')

            # Set the kinematic constraints for BMCP and start joint group velocity controller
            # Do not combine this with the below if statement for not setting the constraint
            if (joy_msg[1][6] * joy_msg[1][7]) and not constraint_is_set:

                constraint_is_set, _, constraint_distance = self.controller.set_kinematics_constraints()
                self.controller.store_constraint_distance(constraint_distance)
                self._data_recording_thread.start()
                rospy.loginfo(
                    'Constraint is set, switching controllers, started velocity controller thread')

            # Once constraint is set, start the teleoperation using
            twist, _ = self.joystick.joy_to_twist(BMCP._TWIST_GAIN)
            if constraint_is_set:

                # Exrtact the twist from the joystick message
                twist_left = self.controller.get_twist_in_tool_frame(side='l', twist=twist)
                twist_right = self.controller.get_twist_in_tool_frame(side='r', twist=twist)

                
                if joy_msg[0][5] != 1:  # RT trigger to allow control signal to be sent

                    # Extract the Jacobians in the middle frame using the virtual robot with joint states data from the real robot
                    jacob_right = self.controller.get_jacobian(side='r')
                    jacob_left = self.controller.get_jacobian(side='l')
                    jacob_constraint = np.c_[jacob_left, -jacob_right]

                    # Calculate the joint velocities using RMRC
                    qdot_right = CalcFuncs.rmrc(jacob_right, twist_right, w_thresh=BMCP._MANIP_THRESH)
                    qdot_left = CalcFuncs.rmrc(jacob_left, twist_left,  w_thresh=BMCP._MANIP_THRESH)
                    # qdot_right = self.controller.process_arm_movement(side='r', 
                    #                                                   twist=twist, 
                    #                                                   manip_thresh=BMCP._MANIP_THRESH, 
                    #                                                   joint_limit_damper=False,
                    #                                                   damper_steepness=BMCP._DAMPER_STEEPNESS, 
                    #                                                   twist_in_ee=True)
                    # qdot_left = self.controller.process_arm_movement(side='l', 
                    #                                                  twist=twist, 
                    #                                                  manip_thresh=BMCP._MANIP_THRESH, 
                    #                                                  joint_limit_damper=False,
                    #                                                  damper_steepness=BMCP._DAMPER_STEEPNESS, 
                    #                                                  twist_in_ee=True)
                    qdot_combined = np.r_[qdot_left, qdot_right]

                    # Perform nullspace projection for qdot_combined on constraint Jacobian to ensure the twist synchronisatio
                    taskspace_drift_compensation = self.controller.task_drift_compensation(gain_p=BMCP._DRIFT_GAIN['p'], 
                                                                                           gain_d=BMCP._DRIFT_GAIN['d'], 
                                                                                           on_taskspace=True) * 2
                    
                    primary_tasks_vel = np.linalg.pinv(jacob_constraint) @ taskspace_drift_compensation
                    secondary_tasks_vel = CalcFuncs.nullspace_projector(jacob_constraint) @ qdot_combined

                    qdot = primary_tasks_vel + secondary_tasks_vel

                    # Add a joint limits damper to the joint velocities
                    qdot += self.controller.joint_limit_damper(qdot, steepness=BMCP._DAMPER_STEEPNESS)
                    
            else:
                
                if joy_msg[1][4]:  # left bumper
                    if joy_msg[0][5] != 1:  
                        qdot[7:] = self.controller.process_arm_movement(side='r', 
                                                                        twist=twist, 
                                                                        manip_thresh=BMCP._MANIP_THRESH, 
                                                                        damper_steepness=BMCP._DAMPER_STEEPNESS)
                if joy_msg[1][5]:  # right bumper
                    if joy_msg[0][5] != 1:
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

            # time.sleep(1/(BMCP.CONTROL_RATE*10))

    # def teleop_with_hand_motion(self):
    #     r"""
    #     Similar system to the coordination part of teleop_test, except for the beginning already start with
    #     joint_group_vel_controller vel cmd input from montion controller interface instead of 
    #     letting trajectory-liked motion to set the initial state for setting the constraint
    #     """

    #     # State variables
    #     last_switch_time = 0

    #     rospy.loginfo('Start teleop using Razer Hydra Controller')

    #     jgvc_started = self.controller.start_jg_vel_controller()
    #     if not jgvc_started:
    #         rospy.logerr('Failed to start joint group velocity controller')
    #         return

    #     def check_state(last_switch_time):

    #         # left_joy_msg = self.controller.get_hydra_joy_msg(side='l')
    #         right_joy_msg = self.controller.get_hydra_joy_msg(side='r')

    #         debounced, last_switch_time = CalcFuncs.debounce(last_switch_time)
    #         if debounced:

    #             if right_joy_msg[1][0]:
    #                 if self._state == 'central':
    #                     self._constraint_is_set = False  # Reset the constraint condition
    #                     self.switch_to_individual_control()
    #                 else:
    #                     self.switch_to_central_control()

    #             elif right_joy_msg[1][1]:
    #                 self.stop()

    #         return last_switch_time

    #     # Wait for the controller to start
    #     rospy.sleep(1)
    #     self._control_signal_thread.start()

    #     while not rospy.is_shutdown() and self._state != 'Done':

    #         if self._state == 'individual':

    #             rospy.loginfo('Switching to individual control')

    #             # Start individual control threads
    #             left_thread = threading.Thread(
    #                 target=self.hand_controller, args=('l'))
    #             right_thread = threading.Thread(
    #                 target=self.hand_controller, args=('r'))
    #             left_thread.start()
    #             right_thread.start()

    #             # Wait for state change
    #             while self._state == 'individual':

    #                 last_switch_time = check_state(last_switch_time)
    #                 rospy.sleep(0.1)

    #             # Stop individual control threads
    #             left_thread.join()
    #             right_thread.join()

    #         elif self._state == 'central':

    #             rospy.loginfo('Switching to central control')

    #             # Start central control thread
    #             central_thread = threading.Thread(
    #                 target=self.central_controller)
    #             central_thread.start()

    #             # Wait for state change
    #             while self._state == 'central':

    #                 last_switch_time = check_state(last_switch_time)
    #                 rospy.sleep(0.1)

    #             # Stop central control thread
    #             central_thread.join()

    #         elif self._state == "Done":
    #             rospy.loginfo('Done teleoperation.')
    #             rospy.signal_shutdown('Done')
    #             self._control_signal_thread.join()
    #             rospy.loginfo('Control signal thread joined.')

    #         else:
    #             raise ValueError('Invalid state')

    # def hand_controller(self, side):

    #     synced = False
    #     arm = self._right_arm if side == 'r' else self._left_arm
    #     qdot_lock = self._qdot_right_lock if side == 'r' else self._qdot_left_lock

    #     while self._state == 'individual':

    #         qd = np.zeros(7)
    #         joy_msg = self.controller.get_hydra_joy_msg(side=side)
    #         twist_msg, synced = self.controller.get_twist(
    #             side=side, synced=synced,  gain=BMCP.TWIST_GAIN)

    #         if not synced:
    #             continue

    #         if joy_msg[1][-2]:  # Safety trigger to allow control signal to be sent
    #             jacob = self.controller.get_jacobian(side=side)
    #             qd = CalcFuncs.rmrc(
    #                 jacob, twist_msg, w_thresh=BMCP.MANIP_THRESH)

    #         with qdot_lock:
    #             if side == 'r':
    #                 self._qdot_right = copy.deepcopy(qd)
    #             else:
    #                 self._qdot_left = copy.deepcopy(qd)

    #         if joy_msg[1][3]:
    #             arm.open_gripper()
    #         if joy_msg[1][2]:
    #             arm.close_gripper()

    #         rospy.sleep(0.02)

    # def central_controller(self):

    #     synced = False

    #     if not self._constraint_is_set:

    #         self._constraint_is_set, _, constraint_distance = self.controller.set_kinematics_constraints()
    #         self.controller.store_constraint_distance(constraint_distance)
    #         rospy.loginfo(
    #             'Constraint is set, switching controllers, started velocity controller thread')

    #     while self._state == 'central':

    #         qdot = np.zeros(14)

    #         joy_msg = self.controller.get_hydra_joy_msg(side='r')
    #         twist_msg, synced = self.controller.get_twist(
    #             side='r', synced=synced, gain=BMCP.TWIST_GAIN)

    #         if joy_msg[1][-2]:

    #             # Extract the Jacobians in the middle frame using the virtual robot with joint states data from the real robot
    #             jacob_right = self.controller.get_jacobian(side='r')
    #             jacob_left = self.controller.get_jacobian(side='l')
    #             jacob_constraint = np.c_[jacob_left, -jacob_right]

    #             # Calculate the joint velocities using RMRC
    #             qdot_right = CalcFuncs.rmrc(
    #                 jacob_right, twist_msg, w_thresh=BMCP.MANIP_THRESH)
    #             qdot_left = CalcFuncs.rmrc(
    #                 jacob_left, twist_msg,  w_thresh=BMCP.MANIP_THRESH)
    #             qdot_combined = np.r_[qdot_left, qdot_right]

    #             # @TODO: Joint limits avoidance to be added here through nullspace filtering
    #             joint_limits_damper = self.controller.joint_limit_damper(
    #                 qdot_combined)
    #             qdot_combined += joint_limits_damper

    #             # Perform nullspace projection for qdot_combined on constraint Jacobian to ensure the twist synchronisatio
    #             taskspace_drift_compensation = self.controller.task_drift_compensation(
    #                 gain=BMCP.DRIFT_GAIN, taskspace_compensation=True) * 2
    #             qdot = np.linalg.pinv(jacob_constraint) @ taskspace_drift_compensation + CalcFuncs.nullspace_projector(
    #                 jacob_constraint) @ qdot_combined

    #         # # Control signal send from this block
    #         with self._qdot_right_lock:
    #             self._qdot_right = qdot[7:]
    #         with self._qdot_left_lock:
    #             self._qdot_left = qdot[:7]

    #         if joy_msg[1][3]:
    #             self._left_arm.open_gripper()
    #         elif joy_msg[1][2]:
    #             self._left_arm.close_gripper()

    #         rospy.sleep(1/(self.CONTROL_RATE*10))

    @staticmethod
    def handle_gripper(arm, joy_msg):   
        if joy_msg[0][-1] == 1:  # up
            arm.open_gripper()
        elif - joy_msg[0][-1] == 1:  # down
            arm.close_gripper()

    def base_controller_handler(self):

        while self._state != 'Done':

            twist = np.zeros(6)
            joy_msg = self.joystick.get_joy_msg()
            if joy_msg[0][2] != 1:  # left trigger for base controller
                twist, _ = self.joystick.joy_to_twist(BMCP._TWIST_GAIN, base=True)

            if joy_msg[1][4]:  # left bumper
                BMCP.handle_gripper(self._right_arm, joy_msg)

            if joy_msg[1][5]:  # right bumper
                BMCP.handle_gripper(self._left_arm, joy_msg)

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

                self.controller.store_joint_velocities(
                    'right', self._qdot_right)
                self.controller.store_joint_velocities('left', self._qdot_left)

                self.controller.store_joint_positions()
                self.controller.store_manipulability()
                self.controller.store_drift()


if __name__ == "__main__":
    try:
        # cfg_path = rospkg.RosPack().get_path('bimanual_teleop_controller') + '/config/bmcp_cfg.yaml'
        # config = Config.load_config(cfg_path)
        b = BMCP(config=config)
        b.teleop_test()
    except rospy.ROSInterruptException:
        pass
