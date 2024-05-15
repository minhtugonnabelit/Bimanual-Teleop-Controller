# /usr/bin/env python3

# This file contains the PR2Controller class that is used to control the PR2 robot
#
# The PR2Controller class is used to control the PR2 robot.
# It is responsible for setting up the robot model, initializing the arms, and handling the joint states and joystick messages.
# It also provides functions to move the robot to a neutral position, open and close the grippers, and send joint velocities to the robot.

import rospy
import tf
from tf import TransformerROS as tfROS

from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState, Joy
from pr2_mechanism_msgs.srv import SwitchController, UnloadController
from pr2_controllers_msgs.msg import Pr2GripperCommand

import numpy as np
from bimanual_controller.utility import *
from bimanual_controller.arm_controller import ArmController


class PR2Controller:
    r"""
    Class to control the PR2 robot
    """

    def __init__(self, name, log_level, rate):

        # Initialize the robot model to handle constraints and calculate Jacobians
        rospy.init_node(name, log_level=log_level, anonymous=True)
        self._virtual_robot = FakePR2(launch_visualizer=False)
        self._rate = rospy.Rate(rate)
        self._dt = 1/rate
        self._robot_base_frame = 'base_footprint'
        self.right_arm = ArmController(arm='r',
                                       arm_group_joint_names=JOINT_NAMES['right'],
                                       arm_group_controller_name="/r_arm_joint_group_velocity_controller",
                                       controller_cmd_type=Float64MultiArray,
                                       gripper_cmd_type=Pr2GripperCommand,
                                       robot_base_frame=self._robot_base_frame)

        self.left_arm = ArmController(arm='l',
                                      arm_group_joint_names=JOINT_NAMES['left'],
                                      arm_group_controller_name="/l_arm_joint_group_velocity_controller",
                                      controller_cmd_type=Float64MultiArray,
                                      gripper_cmd_type=Pr2GripperCommand,
                                      robot_base_frame=self._robot_base_frame)

        # Initialize the joint states subscriber
        self._joint_states = None
        self._joint_state_sub = rospy.Subscriber(
            '/joint_states', JointState, self.__joint_state_callback)

        # Initialize the joystick subscriber
        self._joy_msg = None
        self._joy_pygame = joy_init()
        self._rumbled = False
        self._joystick_sub = rospy.Subscriber(
            '/joy', Joy, self.__joystick_callback)

        # self._hydra_joy_msg = {
        #     'l': None,
        #     'r': None
        # }
        # self._hydra_joy_sub = {
        #     'l': rospy.Subscriber('/hydra_left_joy', Joy, self.__hydra_joystick_callback, callback_args='l'),
        #     'r': rospy.Subscriber('/hydra_right_joy', Joy, self.__hydra_joystick_callback, callback_args='r')
        # }
        # self._hydra_base_frame_id = 'hydra_base'
        # self._controller_frame_id = {
        #     'l': 'hydra_left_grab',
        #     'r': 'hydra_right_grab'
        # }

        # Initialize the buffer for the joint velocities recording
        self._constraint_distance = 0
        self._constraint_is_set = False
        self._offset_distance = []
        self._manipulability = [[], []]
        self._q_record = [[], []]
        self._qdot_record = {
            'left': [],
            'right': []
        }
        self._qdot_record_PID = {
            'left': {'desired': [],  'actual': []},
            'right': {'desired': [],  'actual': []}
        }
        
        rospy.loginfo('Controller ready to go')
        rospy.on_shutdown(self.__clean)

    def __clean(self):
        self._virtual_robot.shutdown()
        rospy.loginfo('Shutting down the virtual robot')
        PR2Controller.kill_jg_vel_controller()
        self.move_to_neutral()

        joint_limits = self._virtual_robot.get_joint_limits_all()
        fig, ax = plot_manip_and_drift(
            self._constraint_distance,
            self._manip_thresh,
            joint_limits,
            self._q_record,
            self._qdot_record,
            self._offset_distance,
            self._manipulability,
            dt=self._dt)

        plt.show()

    def sleep(self):
        self._rate.sleep()

    def set_kinematics_constraints(self):
        
        left_pose = self.left_arm.get_gripper_transform()
        right_pose = self.right_arm.get_gripper_transform()

        virtual_pose = np.eye(4)
        virtual_pose[:3, -1] = (left_pose[:3, -1] + right_pose[:3, -1]) / 2
        constraint_distance = np.linalg.norm(
            left_pose[:3, -1] - right_pose[:3, -1])

        self._virtual_robot.set_constraints(virtual_pose)
        return True, virtual_pose, constraint_distance

    def set_manip_thresh(self, manip_thresh):
        self._manip_thresh = manip_thresh

    def move_to_neutral(self):

        result_r = self.right_arm.move_to_neutral()
        result_l = self.left_arm.move_to_neutral()
        return result_l

    # Callback functions

    def __joint_state_callback(self, msg: JointState):
        self._joint_states = msg
        self._virtual_robot.set_states(self._joint_states.position)

    def __joystick_callback(self, msg: Joy):
        self._joy_msg = (msg.axes, msg.buttons)

    # def __hydra_joystick_callback(self, msg: Joy, side: str):
    #     self._hydra_joy_msg[side] = (msg.axes, msg.buttons)

    # Getters

    def get_joy_msg(self):
        return self._joy_msg

    # def get_hydra_joy_msg(self, side: str):
    #     return self._hydra_joy_msg[side]

    # def get_twist(self, side: str, synced=False, gain=[1, 1]):

    #     try:
    #         self._tf_listener.waitForTransform(
    #             self._controller_frame_id[side], self._hydra_base_frame_id, rospy.Time(), rospy.Duration(20))
    #         synced = True
    #     except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
    #         rospy.logwarn("Waiting for tf")
    #         synced = False

    #     if not synced:
    #         return None, synced

    #     twist = self._tf_listener.lookupTwist(
    #         self._controller_frame_id[side], self._hydra_base_frame_id, rospy.Time(), rospy.Duration(1/CONTROL_RATE)) if synced else None

    #     xdot = np.zeros(6)
    #     xdot[:3] = np.array(twist[0]) * gain[0]
    #     xdot[3:] = np.array(twist[1]) * gain[1]

    #     return xdot, synced

    def get_joint_states(self):
        return self._joint_states

    def get_jacobian(self, side: str):
        return self._virtual_robot.get_jacobian(side)

    def joint_limit_damper(self, qdot, steepness=10) -> list:
        r"""
        joint limit avoidance mechanism with speed scaling factor 

        Args:
            qdot (list): Joint velocities

        Returns:
            list: Joint velocities with joint limit avoidance mechanism applied
        """
        joint_limits_damper, max_weights = self._virtual_robot.joint_limits_damper(
            qdot, self._dt, steepness)

        if max_weights > 0.75:
            rumble_freq = (max_weights - 0.75)*3
            self._rumbled = self._joy_pygame.rumble(rumble_freq, 2*rumble_freq, 0)
            rospy.logwarn(
                f"\nJoint limit avoidance mechanism is applied with max weight: {max_weights:.2f}")
        else :
            self._joy_pygame.stop_rumble() if self._rumbled else None
            self._rumbled = False
        return joint_limits_damper

    def task_drift_compensation(self, gain=5, taskspace_compensation=True):
        r"""
        Task drift compensator mechanism 

        Args:
            gain (int, optional): Gain of the RMRC. Defaults to 5.
            taskspace_compensation (bool, optional): Flag to indicate if the compensation is in task space. Defaults to True.

        Returns:
            list: Joint velocities with task drift compensation mechanism applied
        """

        return self._virtual_robot.task_drift_compensation(gain, taskspace_compensation)
    
    @ staticmethod
    def start_jg_vel_controller():

        switched = ROSUtils.call_service('pr2_controller_manager/switch_controller',
                                         SwitchController,
                                         start_controllers=[
                                             'r_arm_joint_group_velocity_controller',
                                             'l_arm_joint_group_velocity_controller'],
                                         stop_controllers=[
                                             'r_arm_controller',
                                             'l_arm_controller'],
                                         strictness=1)

        return switched

    @ staticmethod
    def kill_jg_vel_controller():
        rospy.loginfo(
            'Switching controllers and unloading velocity controllers')

        switched = ROSUtils.call_service('pr2_controller_manager/switch_controller',
                                         SwitchController,
                                         start_controllers=[
                                             'r_arm_controller',
                                             'l_arm_controller'],
                                         stop_controllers=[
                                             'r_arm_joint_group_velocity_controller',
                                             'l_arm_joint_group_velocity_controller'],
                                         strictness=1)

        ROSUtils.call_service('pr2_controller_manager/unload_controller',
                              UnloadController, name='l_arm_joint_group_velocity_controller')
        ROSUtils.call_service('pr2_controller_manager/unload_controller',
                              UnloadController, name='r_arm_joint_group_velocity_controller')

        rospy.loginfo('Controllers switched and unloaded')

        return switched

    # Record function

    def store_constraint_distance(self, distance: float):
        self._constraint_distance = distance

    def store_drift(self):
        self._offset_distance.append(
            np.linalg.norm(
                self._virtual_robot.get_tool_pose(side=self.left_arm.get_arm_name(), offset=False)[:3, -1] -
                self._virtual_robot.get_tool_pose(side=self.right_arm.get_arm_name(), offset=False)[:3, -1]))

    def store_manipulability(self):
        self._manipulability[0].append(CalcFuncs.manipulability(
            self.get_jacobian(self.left_arm.get_arm_name())))
        self._manipulability[1].append(CalcFuncs.manipulability(
            self.get_jacobian(self.right_arm.get_arm_name())))

    def store_joint_velocities(self, side: str, qdot: list):
        self._qdot_record[side].append(qdot)

    def store_joint_positions(self):
        self._q_record[0].append(CalcFuncs.reorder_values(
            self._joint_states.position[31:38]))
        self._q_record[1].append(CalcFuncs.reorder_values(
            self._joint_states.position[17:24]))
        
    def store_joint_velocities_for_PID_tuner(self, side: str, qdot: list):
        self._qdot_record_PID[side]['desired'].append(qdot)

        if side == 'left':
            self._qdot_record_PID[side]['actual'].append(
                CalcFuncs.reorder_values(self._joint_states.velocity[31:38]))
        else:
            self._qdot_record_PID[side]['actual'].append(
                CalcFuncs.reorder_values(self._joint_states.velocity[17:24]))
