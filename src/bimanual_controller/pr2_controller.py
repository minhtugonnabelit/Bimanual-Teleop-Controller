# /usr/bin/env python3

import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState, Joy
from geometry_msgs.msg import Twist
from pr2_mechanism_msgs.srv import SwitchController, UnloadController
from pr2_controllers_msgs.msg import Pr2GripperCommand

import numpy as np
from bimanual_controller.utility import *
from bimanual_controller.fake_pr2 import FakePR2
from bimanual_controller.arm_controller import ArmController
from bimanual_controller.joystick_controller import JoystickController


class PR2Controller:

    def __init__(self, rate, joystick : JoystickController, config):

        self._cfg = config
        self._JOINT_NAMES = self._cfg['JOINT_NAMES']

        self._virtual_robot = FakePR2(control_rate=rate,launch_visualizer=False)
        self._rate = rospy.Rate(rate)
        self._dt = 1/rate

        self._robot_base_frame = 'base_footprint'
        self._right_arm = ArmController(arm='r',
                                       arm_group_joint_names=self._JOINT_NAMES['right'],
                                       arm_group_controller_name="/r_arm_joint_group_velocity_controller",
                                       controller_cmd_type=Float64MultiArray,
                                       gripper_cmd_type=Pr2GripperCommand,
                                       robot_base_frame=self._robot_base_frame)

        self._left_arm = ArmController(arm='l',
                                      arm_group_joint_names=self._JOINT_NAMES['left'],
                                      arm_group_controller_name="/l_arm_joint_group_velocity_controller",
                                      controller_cmd_type=Float64MultiArray,
                                      gripper_cmd_type=Pr2GripperCommand,
                                      robot_base_frame=self._robot_base_frame)
        
        self._joystick = joystick

        self._base_controller_pub = rospy.Publisher(
            '/base_controller/command', Twist, queue_size=10)
        
        self._joint_states = None
        self._joint_state_sub = rospy.Subscriber(
            '/joint_states', JointState, self._joint_state_callback)

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
        if self._joystick.is_rumbled():
            self._joystick.stop_rumble()
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

    def _joint_state_callback(self, msg: JointState):
        self._joint_states = msg
        self._virtual_robot.set_states(self._joint_states.position)

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
    

    def process_arm_movement(self, side: str, twist, manip_thresh, joint_limit_damper=True, damper_steepness=10, twist_in_ee=False):
        if not twist_in_ee:
            ee_pose = np.round(self._right_arm.get_gripper_transform(), 4) if side == 'r' else np.round(self._left_arm.get_gripper_transform(), 4)
            twist_w_in_ee = CalcFuncs.adjoint(np.linalg.inv(ee_pose)) @ twist
        else:
            twist_w_in_ee = twist   
        jacob = self.get_jacobian(side=side)

        alpha = 0.1 # gain value for manipulability gradient
        qdot_sec = alpha * self._virtual_robot.manipulability_gradient(side=side)

        qdot = np.linalg.pinv(jacob) @ twist_w_in_ee + \
                CalcFuncs.nullspace_projector(jacob) @ qdot_sec
        
        if joint_limit_damper:
            qdot += self.joint_limit_damper_side(side=side, qdot=qdot, steepness=damper_steepness)
        
        # qdot_indiv = CalcFuncs.rmrc(jacob, twist_w_in_ee, w_thresh=manip_thresh)
        
        # if joint_limit_damper:
        #     qdot_indiv += self.joint_limit_damper_side(side=side, qdot=qdot_indiv, steepness=damper_steepness)
        
        # qdot = np.linalg.pinv(jacob) @ twist_w_in_ee + CalcFuncs.nullspace_projector(jacob) @ qdot_indiv

        return qdot

    def joint_limit_damper_side(self, side: str, qdot, steepness=10) -> list:
            
        joint_limits_damper, max_weights, joint_on_max_limit = self._virtual_robot.joint_limits_damper_side(
            side, qdot, self._dt, steepness)

        if max_weights > 0.8:
            side = 'left' if side == 'l' else 'right'
            rumble_freq = (max_weights - 0.8)*3
            self._joystick.start_rumble(rumble_freq, 2*rumble_freq, 0)
            rospy.logwarn(
                f"\nJoint limit avoidance mechanism is applied with max weight: {max_weights:.2f} at joint {self._JOINT_NAMES[side][joint_on_max_limit[0]]}")
        else:
            self._joystick.stop_rumble() if self._joystick.is_rumbled() else None

        return joint_limits_damper

    def joint_limit_damper(self, qdot, steepness=10) -> list:
        joint_limits_damper, max_weights, joint_on_max_limit = self._virtual_robot.joint_limits_damper(
            qdot, self._dt, steepness)

        if max_weights > 0.8:

            side = 'left'
            if joint_on_max_limit > 6:
                joint_on_max_limit -= 7
                side = 'right'

            rumble_freq = (max_weights - 0.8)*3
            self._joystick.start_rumble(rumble_freq, 2*rumble_freq, 0)
            rospy.logwarn(
                f"\nJoint limit avoidance mechanism is applied with max weight: {max_weights:.2f} at joint {self._JOINT_NAMES[side][joint_on_max_limit[0]]}")
        else :
            self._joystick.stop_rumble() if self._joystick.is_rumbled() else None

        return joint_limits_damper

    def task_drift_compensation(self, gain_p=5, gain_d=0.5, on_taskspace=True):
        return self._virtual_robot.task_drift_compensation(gain_p, gain_d, on_taskspace)

    def move_to_neutral(self):
        result_r = self._right_arm.move_to_neutral(self._cfg['SAMPLE_STATES'][self._right_arm.get_arm_name()])
        result_l = self._left_arm.move_to_neutral(self._cfg['SAMPLE_STATES'][self._left_arm.get_arm_name()])
        return result_l

    def move_base(self, twist):
        twist_msg = Twist()
        twist_msg.linear.x = twist[0]
        twist_msg.linear.y = twist[1]
        twist_msg.linear.z = twist[2]
        twist_msg.angular.x = twist[3]
        twist_msg.angular.y = twist[4]
        twist_msg.angular.z = twist[5]
        self._base_controller_pub.publish(twist_msg)


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
