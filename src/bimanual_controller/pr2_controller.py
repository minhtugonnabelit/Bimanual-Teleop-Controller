# /usr/bin/env python3

import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState, Joy
from geometry_msgs.msg import Twist
from pr2_mechanism_msgs.srv import SwitchController, UnloadController
from pr2_controllers_msgs.msg import Pr2GripperCommand

import numpy as np
from bimanual_controller.utility import *
from bimanual_controller.arm_controller import ArmController


class PR2Controller:

    def __init__(self, name, log_level, rate):

        rospy.init_node(name, log_level=log_level, anonymous=True)
        self._virtual_robot = FakePR2(control_rate=rate,launch_visualizer=False)
        self._rate = rospy.Rate(rate)
        self._dt = 1/rate
        self._robot_base_frame = 'base_footprint'
        self._right_arm = ArmController(arm='r',
                                       arm_group_joint_names=JOINT_NAMES['right'],
                                       arm_group_controller_name="/r_arm_joint_group_velocity_controller",
                                       controller_cmd_type=Float64MultiArray,
                                       gripper_cmd_type=Pr2GripperCommand,
                                       robot_base_frame=self._robot_base_frame)

        self._left_arm = ArmController(arm='l',
                                      arm_group_joint_names=JOINT_NAMES['left'],
                                      arm_group_controller_name="/l_arm_joint_group_velocity_controller",
                                      controller_cmd_type=Float64MultiArray,
                                      gripper_cmd_type=Pr2GripperCommand,
                                      robot_base_frame=self._robot_base_frame)

        self._base_controller_pub = rospy.Publisher(
            '/base_controller/command', Twist, queue_size=10)
        
        self._joint_states = None
        self._joint_state_sub = rospy.Subscriber(
            '/joint_states', JointState, self.__joint_state_callback)

        joy = rospy.wait_for_message('/joy', Joy)
        self._joy_msg = (joy.axes, joy.buttons)
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
        if self._rumbled:
            self._joy_pygame.stop_rumble()
        PR2Controller.kill_jg_vel_controller()

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
        left_pose = self._left_arm.get_gripper_transform()
        right_pose = self._right_arm.get_gripper_transform()

        virtual_pose = np.eye(4)
        virtual_pose[:3, -1] = (left_pose[:3, -1] + right_pose[:3, -1]) / 2
        constraint_distance = np.linalg.norm(
            left_pose[:3, -1] - right_pose[:3, -1])

        self._virtual_robot.set_constraints(virtual_pose)
        return True, virtual_pose, constraint_distance

    def set_manip_thresh(self, manip_thresh):
        self._manip_thresh = manip_thresh


    # Getters

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

    def get_arm_controller(self, side: str):
        return self._right_arm if side == 'r' else self._left_arm

    def get_joint_states(self):
        return self._joint_states

    def get_jacobian(self, side: str):
        return self._virtual_robot.get_jacobian(side)

    def get_tool_pose(self, side: str, isOffset=True):
        return self._virtual_robot.get_tool_pose(side, isOffset)
    
    def get_twist_in_tool_frame(self, side: str, twist):
        return self._virtual_robot.get_twist_in_tool_frame(side, twist)


    def joint_limit_damper(self, qdot, steepness=10) -> list:
        r"""
        joint limit avoidance mechanism with speed scaling factor 

        Args:
            qdot (list): Joint velocities

        Returns:
            list: Joint velocities with joint limit avoidance mechanism applied
        """
        joint_limits_damper, max_weights, joint_on_max_limit = self._virtual_robot.joint_limits_damper(
            qdot, self._dt, steepness)

        if max_weights > 0.8:

            side = 'right'
            if joint_on_max_limit > 6:
                joint_on_max_limit -= 7
                side = 'left'

            rumble_freq = (max_weights - 0.8)*3
            self._rumbled = self._joy_pygame.rumble(rumble_freq, 2*rumble_freq, 0)
            rospy.logwarn(
                f"\nJoint limit avoidance mechanism is applied with max weight: {max_weights:.2f} at joint {JOINT_NAMES[side][joint_on_max_limit[0]]}")
        else :
            self._joy_pygame.stop_rumble() if self._rumbled else None
            self._rumbled = False
        return joint_limits_damper

    def joint_limit_damper_side(self, side: str, qdot, steepness=10) -> list:
            
        joint_limits_damper, max_weights, joint_on_max_limit = self._virtual_robot.joint_limits_damper_side(
            side, qdot, self._dt, steepness)

        if max_weights > 0.8:
            side = 'left' if side == 'l' else 'right'
            rumble_freq = (max_weights - 0.8)*3
            self._rumbled = self._joy_pygame.rumble(rumble_freq, 2*rumble_freq, 0)
            rospy.logwarn(
                f"\nJoint limit avoidance mechanism is applied with max weight: {max_weights:.2f} at joint {JOINT_NAMES[side][joint_on_max_limit[0]]}")
        else:
            self._joy_pygame.stop_rumble() if self._rumbled else None
            self._rumbled = False

        return joint_limits_damper

    def joint_limit_damper_right(self, qdot, steepness=10) -> list:

        joint_limits_damper, _ = self._virtual_robot.joint_limits_damper_right(
            qdot, self._dt, steepness)

        return joint_limits_damper 
    
    def joint_limit_damper_left(self, qdot, steepness=10) -> list:

        joint_limits_damper, _ = self._virtual_robot.joint_limits_damper_left(
            qdot, self._dt, steepness)

        return joint_limits_damper

    def task_drift_compensation(self, gain_p=5, gain_d=0.5, on_taskspace=True):
        r"""
        Task drift compensator mechanism 

        Args:
            gain (int, optional): Gain of the RMRC. Defaults to 5.
            taskspace_compensation (bool, optional): Flag to indicate if the compensation is in task space. Defaults to True.

        Returns:
            list: Joint velocities with task drift compensation mechanism applied
        """

        return self._virtual_robot.task_drift_compensation(gain_p, gain_d, on_taskspace)


    def move_to_neutral(self):
        result_r = self._right_arm.move_to_neutral()
        result_l = self._left_arm.move_to_neutral()
        return result_l

    def move_base(self, twist):
        twist_msg = Twist()
        twist_msg.linear.x = twist[0]
        twist_msg.linear.y = twist[1]
        twist_msg.angular.z = twist[2]
        twist_msg.angular.x = twist[3]
        twist_msg.angular.y = twist[4]
        twist_msg.angular.z = twist[5]
        self._base_controller_pub.publish(twist_msg)

    def process_arm_movement(self, side: str, twist, manip_thresh, joint_limit_damper=True, damper_steepness=10):
        ee_pose = np.round(self._right_arm.get_gripper_transform(), 4) if side == 'r' else np.round(self._left_arm.get_gripper_transform(), 4)
        
        qdot_indiv, twist_converted, jacob = self._world_twist_to_qdot(ee_pose, twist, side=side, manip_thresh=manip_thresh)
        
        if joint_limit_damper:
            qdot_indiv += self.joint_limit_damper_side(side=side, qdot=qdot_indiv, steepness=damper_steepness)
        
        qdot = np.linalg.pinv(jacob) @ twist_converted + CalcFuncs.nullspace_projector(jacob) @ qdot_indiv
        return qdot

    # TODO: Put this function in a utility class as a calculation function
    def _world_twist_to_qdot(self, ee_pose : np.ndarray, twist : list, side, manip_thresh) -> list:
        adjoint = CalcFuncs.adjoint(np.linalg.inv(ee_pose))
        twist = adjoint @ twist
        jacob = self.get_jacobian(side=side)
        qdot = CalcFuncs.rmrc(jacob, twist, w_thresh=manip_thresh)

        return qdot, twist, jacob


    #  TODO: Implement the following methods as separate class for the joystick
    def __joystick_callback(self, msg: Joy):
        self._joy_msg = (msg.axes, msg.buttons)

    def get_joy_msg(self):
        return self._joy_msg

    def rumble_joy(self, freq, duration):
        self._rumbled = self._joy_pygame.rumble(freq, duration, 0)
    
    def rumble_stop(self):
        self._joy_pygame.stop_rumble() if self._rumbled else None
        self._rumbled = False
    

    @ staticmethod
    def start_jg_vel_controller():
        rospy.loginfo('Loading and starting velocity controllers')
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
        rospy.loginfo('Switching controllers and unloading velocity controllers')
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


    def __joint_state_callback(self, msg: JointState):
        self._joint_states = msg
        self._virtual_robot.set_states(self._joint_states.position)

    # def __hydra_joystick_callback(self, msg: Joy, side: str):
    #     self._hydra_joy_msg[side] = (msg.axes, msg.buttons)

    def store_constraint_distance(self, distance: float):
        self._constraint_distance = distance

    def store_drift(self):
        self._offset_distance.append(
            np.linalg.norm(
                self._virtual_robot.get_tool_pose(side=self._left_arm.get_arm_name(), offset=False)[:3, -1] -
                self._virtual_robot.get_tool_pose(side=self._right_arm.get_arm_name(), offset=False)[:3, -1]))

    def store_manipulability(self):
        self._manipulability[0].append(CalcFuncs.manipulability(
            self.get_jacobian(self._left_arm.get_arm_name())))
        self._manipulability[1].append(CalcFuncs.manipulability(
            self.get_jacobian(self._right_arm.get_arm_name())))

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
