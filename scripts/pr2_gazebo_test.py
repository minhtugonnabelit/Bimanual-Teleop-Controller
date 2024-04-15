import tf
import rospy
from std_msgs.msg import Float64MultiArray, Float64
from sensor_msgs.msg import JointState, Joy
from geometry_msgs.msg import TwistStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from pr2_mechanism_msgs.srv import SwitchController, UnloadController
from pr2_controllers_msgs.msg import Pr2GripperCommand

import numpy as np
import spatialmath as sm
import roboticstoolbox as rtb
import matplotlib.pyplot as plt
import time
from scipy import linalg, optimize

# Import custom utility functions
from bimanual_controller.utility import *
from bimanual_controller.fakePR2 import FakePR2

class PR2Controller:

    def __init__(self, rate):

        # Initialize the robot model to handle constraints and calculate Jacobians
        self._virtual_robot = FakePR2(launch_visualizer=True)
        self._rate = rospy.Rate(rate)
        self._dt = 1/rate

        self._qdot_msg = {
            'left': np.zeros(7),
            'right': np.zeros(7)
        }

        # Initialize the buffer for the joint velocities recording
        self._qdot_record = {
            'left': {'desired': [],  'actual': []},
            'right': {'desired': [],  'actual': []}
        }

        self._offset_distance = []
        self._constraint_is_set = False

        # Initialize arms trajectory controllers publishers
        self._arm_control_pub = {
            'left': rospy.Publisher('l_arm_controller/command', JointTrajectory, queue_size=1),
            'right': rospy.Publisher('r_arm_controller/command', JointTrajectory, queue_size=1)
        }

        self._arms_vel_controller_pub = {
            'right': rospy.Publisher('/r_arm_joint_group_velocity_controller/command', Float64MultiArray, queue_size=1),
            'left': rospy.Publisher('/l_arm_joint_group_velocity_controller/command', Float64MultiArray, queue_size=1)
        }

        # Initialize the gripper command publishers
        self._gripper_cmd_pub = {
            'right': rospy.Publisher('r_gripper_controller/command', Pr2GripperCommand, queue_size=1),
            'left': rospy.Publisher('l_gripper_controller/command', Pr2GripperCommand, queue_size=1)
        }

        # Initialize the joint states subscriber
        self._joint_states = None
        self._joint_state_sub = rospy.Subscriber(
            '/joint_states', JointState, self._joint_state_callback)

        # Initialize the joystick subscriber
        self._joy_msg = None
        self._joystick_sub = rospy.Subscriber(
            '/joy', Joy, self._joystick_callback)

        # # Initialize the velocities command subscriber
        # self._vel_cmd_sub = {
        #     'right': rospy.Subscriber('/r_arm_joint_group_velocity_controller/command', Float64MultiArray, self._r_vel_cmd_callback),
        #     'left': rospy.Subscriber('/l_arm_joint_group_velocity_controller/command', Float64MultiArray, self._l_vel_cmd_callback),
        # }

        # Initialize the twist subscriber from node Hydra_reader
        # self._twist_msg = {
        #     'left': None,
        #     'right': None
        # }
        # self._twist_sub = {
        #     'left': rospy.Subscriber(
        #         '/l_arm_servo_server/delta_twist_cmds', TwistStamped, self._twist_callback, callback_args='left'),
        #     'right': rospy.Subscriber(
        #         '/r_arm_servo_server/delta_twist_cmds', TwistStamped, self._twist_callback, callback_args='right')
        # }

        # Initialize the transform listener
        self._tf_listener = tf.TransformListener()
        self._tf_broadcaster = tf.TransformBroadcaster()


        rospy.loginfo('Controller ready to go')
        rospy.on_shutdown(self._clean)

    def _clean(self):
        r"""
        Clean up function with dedicated shutdown procedure"""

        self._virtual_robot.shutdown()
        rospy.loginfo('Shutting down the virtual robot')
        PR2Controller._kill_jg_vel_controller()
        fig1, ax1 = plot_joint_velocities(
            self._qdot_record['left']['actual'], self._qdot_record['left']['desired'], distance_data=self._offset_distance, dt=self._dt, title='left')
        fig2, ax2 = plot_joint_velocities(
            self._qdot_record['right']['actual'], self._qdot_record['right']['desired'], distance_data=self._offset_distance, dt=self._dt, title='right')
        plt.show()

    def set_kinematics_constraints(self):
        r"""
        Set the kinematics constraints for the robot
        :return: None
        """
        tf.TransformListener().waitForTransform(
            'base_link', 'l_gripper_tool_frame', rospy.Time(), rospy.Duration(4.0))

        left_pose = self._tf_listener.lookupTransform(
            'base_footprint', 'l_gripper_tool_frame', rospy.Time(0))
        left_pose = tf.TransformerROS.fromTranslationRotation(
            tf.TransformerROS, translation=left_pose[0], rotation=left_pose[1])

        right_pose = self._tf_listener.lookupTransform(
            'base_footprint', 'r_gripper_tool_frame', rospy.Time(0))
        right_pose = tf.TransformerROS.fromTranslationRotation(
            tf.TransformerROS, translation=right_pose[0], rotation=right_pose[1])

        virtual_pose = np.eye(4)
        virtual_pose[:3, -1] = (left_pose[:3, -1] + right_pose[:3, -1]) / 2

        self._virtual_robot.set_constraints(virtual_pose)
        return True, virtual_pose

    def move_to_neutral(self):
        r"""
        Move the robot to neutral position
        :return: None
        """
        self._arm_control_pub['right'].publish(
            PR2Controller._create_joint_traj_msg(JOINT_NAMES['right'], 3, q=SAMPLE_STATES['right']))
        self._arm_control_pub['left'].publish(
            PR2Controller._create_joint_traj_msg(JOINT_NAMES['left'], 3, q=SAMPLE_STATES['left']))

    # Gripper functions

    def open_gripper(self, side: str):
        r"""
        Close the gripper
        :param side: side of the robot
        :return: None
        """

        msg = Pr2GripperCommand()
        msg.position = 0.08
        msg.max_effort = 10.0
        self._gripper_cmd_pub[side].publish(msg)

    def close_gripper(self, side: str):
        r"""
        Close the gripper
        :param side: side of the robot
        :return: None
        """

        msg = Pr2GripperCommand()
        msg.position = 0.0
        msg.max_effort = 10.0
        self._gripper_cmd_pub[side].publish(msg)

    # Test functions

    def joy_to_direct_joint(self):
        r"""
        This test control loop function is used to control PR2 arms directly using PS4 Joystick
        """
        
        V = TWIST_GAIN[0]
        rospy.wait_for_message('/joy', Joy)

        while not rospy.is_shutdown():
            
            qdot = np.zeros(14)

            dir = self._joy_msg[0][1] / np.abs(self._joy_msg[0][1]) if np.abs(self._joy_msg[0][1]) > 0.4 else 0
            if self._joy_msg[1][4]:
                
                for i in range(7):
                    qdot[i+7] = V * dir * self._joy_msg[1][i]


            if self._joy_msg[1][5]:

                for i in range(7):
                    qdot[i] = V * dir * self._joy_msg[1][i]


            self._arms_vel_controller_pub['right'].publish(PR2Controller._joint_group_command_to_msg(qdot[:7])) 
            self._arms_vel_controller_pub['left'].publish(PR2Controller._joint_group_command_to_msg(qdot[7:])) 

            self.rate.sleep()

    def bmcp_teleop(self):
        r"""
        This test control loop function is used to perform coordination control on PR2 arms using single PS4 Joystick
        """

        rospy.loginfo('Start teleop')
        rospy.wait_for_message('/joy', Joy)

        done = False
        while not done:

            if self._joy_msg[1][-3]:

                self.move_to_neutral()

            if (self._joy_msg[1][-1] * self._joy_msg[1][-2]) and not self._constraint_is_set:

                PR2Controller._start_jg_vel_controller()
                rospy.sleep(1)
                self._constraint_is_set, _ = self.set_kinematics_constraints()
                rospy.loginfo('Constraint is set, switching controllers')

            if not self._constraint_is_set:

                gripper_sides = {5: 'right', 4: 'left'}

                for button_index, side in gripper_sides.items():
                    if self._joy_msg[1][button_index]:
                        if self._joy_msg[0][-1] == 1:
                            self.open_gripper(side)
                        elif self._joy_msg[0][-1] == -1:
                            self.close_gripper(side)

            if self._constraint_is_set:

                qdot_r = np.zeros(7)
                qdot_l = np.zeros(7)
                twist, done = joy_to_twist(self._joy_msg, TWIST_GAIN)

                if self._joy_msg[1][5]:

                    start_time = time.time()
                    jacob_right = self._virtual_robot.get_jacobian('right')
                    jacob_left = self._virtual_robot.get_jacobian('left')

                    qdot_l, qdot_r = CalcFuncs.duo_arm_qdot_constraint(jacob_left,
                                                                       jacob_right,
                                                                       twist,
                                                                       activate_nullspace=True)

                    exec_time = time.time() - start_time
                    rospy.loginfo(
                        f'Calculation time: {exec_time:.4f} Twist received: {twist}')

                self._qdot_msg['right'] = qdot_r
                self._qdot_msg['left'] = qdot_l

                self._arms_vel_controller_pub['right'].publish(
                    PR2Controller._joint_group_command_to_msg(qdot_r))

                self._arms_vel_controller_pub['left'].publish(
                    PR2Controller._joint_group_command_to_msg(qdot_l))

            if done:
                rospy.loginfo('Done teleoperation.')
                rospy.signal_shutdown('Done')

            self._rate.sleep()

    def path_trakcing_test(self):

        rospy.loginfo('Start path tracking test')
        rospy.wait_for_message('/joy', Joy)

        updated_joined_left = self._virtual_robot.get_tool_pose('left')
        arrived = False
        qdot = np.zeros(14)
        target = np.eye(4)

        while not rospy.is_shutdown():

            if self._joy_msg[1][-3]:

                self.move_to_neutral()

            if (self._joy_msg[1][4] * self._joy_msg[1][5]) and not self._constraint_is_set:

                PR2Controller._start_jg_vel_controller()
                rospy.sleep(1)
                self._constraint_is_set, pose = self.set_kinematics_constraints()
                target = pose @ sm.SE3(0, 0, 0.1).A
                rospy.loginfo('constraint is set')

            if self._constraint_is_set:

                updated_joined_left = self._virtual_robot.get_tool_pose('left')
                middle_twist, arrived = rtb.p_servo(updated_joined_left,
                                                    target,
                                                    gain=0.4,
                                                    threshold=0.05,
                                                    method='angle-axis')  # Servoing in the virtual middle frame using angle-axis representation for angular error

                jacob_left = self._virtual_robot.get_jacobian('left')
                jacob_right = self._virtual_robot.get_jacobian('right')

                # Calculate the joint velocities using the Resolved Motion Rate Control (RMRC) method with the projection onto nullspace of Constraint Jacobian
                qdot_l, qdot_r = CalcFuncs.duo_arm_qdot_constraint(jacob_left,
                                                                   jacob_right,
                                                                   middle_twist,
                                                                   activate_nullspace=True)

                # Visualization of the frames
                updated_joined_left = self._virtual_robot.get_tool_pose('left')

                qdot = np.concatenate([qdot_r, qdot_l])
                msg = PR2Controller._joint_group_command_to_msg(qdot)
                self._joint_group_vel_pub.publish(msg)

            if arrived:
                print('Arrived')
                break

            self._rate.sleep()

    # Callback functions

    def _joint_state_callback(self, msg: JointState):
        r"""
        Callback function for the joint state subscriber
        :param msg: JointState message
        :return: None
        """

        self._joint_states = msg
        self._virtual_robot.set_states(self._joint_states.position)

        self._qdot_record['right']['desired'].append(self._qdot_msg['right'])
        self._qdot_record['left']['desired'].append(self._qdot_msg['left'])

        self._qdot_record['right']['actual'].append(reorder_values(self._joint_states.effort[17:24]))
        self._qdot_record['left']['actual'].append(reorder_values(self._joint_states.effort[31:38]))
        
        self._offset_distance.append(np.linalg.norm(self._virtual_robot.get_tool_pose(
            'left', offset=False)[:3, -1] - self._virtual_robot.get_tool_pose('right', offset=False)[:3, -1]))

    def _joystick_callback(self, msg: Joy):
        r"""
        Callback function for the joystick subscriber
        :param msg: Joy message
        :return: None
        """

        self._joy_msg = (msg.axes, msg.buttons)

    def _twist_callback(self, msg: TwistStamped, side: str):
        r"""
        Callback function for the twist subscriber
        :param msg: TwistStamped message
        :param side: side of the arm
        :return: None
        """

        self._twist_msg[side] = msg

    def _r_vel_cmd_callback(self, msg: Float64MultiArray):
        r"""
        Callback function for the right velocity command subscriber to record the joint velocities"""

        self._qdot_record['right']['desired'].append(msg.data)
        self._qdot_record['right']['actual'].append(
            reorder_values(self._joint_states.velocity[17:24]))
 
    def _l_vel_cmd_callback(self, msg: Float64MultiArray):
        r"""
        Callback function for the left velocity command subscriber to record the joint velocities"""

        self._qdot_record['left']['desired'].append(msg.data)
        self._qdot_record['left']['actual'].append(
            reorder_values(self._joint_states.velocity[31:38]))

        self._offset_distance.append(np.linalg.norm(self._virtual_robot.get_tool_pose(
            'left')[:3, -1] - self._virtual_robot.get_tool_pose('right')[:3, -1]))

    # Utility functions
    @staticmethod
    def _call(service_name: str, service_type: str, **kwargs):
        r"""
        Call the service
        :param service_name: name of the service
        :param service_type: type of the service
        :param kwargs: additional arguments
        :return: bool value of the service call
        """

        rospy.wait_for_service(service_name)
        try:
            service = rospy.ServiceProxy(service_name, service_type)
            response = service(**kwargs)
            return response.ok
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    @staticmethod
    def _start_jg_vel_controller():
        r"""
        Switch the controllers
        :return: bool value of the service call
        """

        switched = PR2Controller._call('pr2_controller_manager/switch_controller',
                                       SwitchController,
                                       start_controllers=[
                                           'r_arm_joint_group_velocity_controller',
                                           'l_arm_joint_group_velocity_controller'],
                                       stop_controllers=[
                                           'r_arm_controller',
                                           'l_arm_controller'],
                                       strictness=1)

        return switched

    @staticmethod
    def _kill_jg_vel_controller():
        r"""
        Switch the controllers
        :return: bool value of the service call
        """
        rospy.loginfo(
            'Switching controllers and unloading velocity controllers')

        switched = PR2Controller._call('pr2_controller_manager/switch_controller',
                                       SwitchController,
                                       start_controllers=[
                                           'r_arm_controller',
                                           'l_arm_controller'],
                                       stop_controllers=[
                                           'r_arm_joint_group_velocity_controller',
                                           'l_arm_joint_group_velocity_controller'],
                                       strictness=1)

        PR2Controller._call('pr2_controller_manager/unload_controller',
                            UnloadController,
                            name='l_arm_joint_group_velocity_controller')

        PR2Controller._call('pr2_controller_manager/unload_controller',
                            UnloadController,
                            name='r_arm_joint_group_velocity_controller')

        rospy.loginfo('Controllers switched and unloaded')

        return switched

    @staticmethod
    def _joint_group_command_to_msg(values: list):
        r"""
        Convert the joint group command to Float64MultiArray message

        Args:
            values (list): List of joint values

        Returns:
            Float64MultiArray: Float64MultiArray message
        """

        msg = Float64MultiArray()
        msg.data = values
        return msg

    @staticmethod
    def _create_joint_traj_msg(joint_names: list, dt: float, joint_states: list = None, qdot: list = None, q: list = None):
        r"""
        Create a joint trajectory message. 
        If q desired is not provided, the joint velocities are used to calculate the joint positions

        Args:
            joint_names (list): List of joint names.
            dt (float): Time duration.
            joint_states (list, optional): List of joint states. Defaults to None.
            qdot (list, optional): List of joint velocities. Defaults to None.
            q (list, optional): List of joint positions. Defaults to None.

        Returns:
            JointTrajectory: JointTrajectory message.
        """

        joint_traj = JointTrajectory()
        joint_traj.header.frame_id = 'torso_lift_link'
        joint_traj.joint_names = joint_names

        traj_point = JointTrajectoryPoint()
        if q is not None:
            traj_point.positions = q
        else:
            traj_point.positions = reorder_values(joint_states) + qdot * dt
            traj_point.velocities = qdot

        traj_point.time_from_start = rospy.Duration(dt)
        joint_traj.points = [traj_point]

        return joint_traj


if __name__ == "__main__":

    rospy.init_node('bcmp_test', log_level=rospy.INFO, anonymous=True,)
    rospy.logdebug('Command node initialized')
    controller = PR2Controller(rate=CONTROL_RATE)
    controller.bmcp_teleop()
