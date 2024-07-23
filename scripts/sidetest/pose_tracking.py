import rospy

from bimanual_teleop_controller.utility import CONTROL_RATE
from bimanual_teleop_controller.pr2_controller import PR2Controller

def main():

    rospy.init_node('bcmp_test', log_level=rospy.INFO, anonymous=True,)
    rospy.logdebug('Command node initialized')
    controller = PR2Controller(rate=CONTROL_RATE)
    controller.path_trakcing_test()

        # Test functions

    def joy_to_direct_qdot(self):
        r"""
        This test control loop function is used to control PR2 arms directly using PS4 Joystick
        """

        V = TWIST_GAIN[0]
        rospy.wait_for_message('/joy', Joy)

        while not rospy.is_shutdown():

            qdot = np.zeros(14)

            dir = self._joy_msg[0][1] / np.abs(self._joy_msg[0]
                                               [1]) if np.abs(self._joy_msg[0][1]) > 0.4 else 0
            if self._joy_msg[1][4]:

                for i in range(7):
                    qdot[i+7] = V * dir * self._joy_msg[1][i]

            if self._joy_msg[1][5]:

                for i in range(7):
                    qdot[i] = V * dir * self._joy_msg[1][i]

            self._arms_vel_controller_pub['right'].publish(
                PR2Controller._joint_group_command_to_msg(qdot[:7]))
            self._arms_vel_controller_pub['left'].publish(
                PR2Controller._joint_group_command_to_msg(qdot[7:]))

            self._rate.sleep()
            
    def path_trakcing_test(self):

        rospy.loginfo('Start path tracking test')
        rospy.wait_for_message('/joy', Joy)

        updated_joined_left = self._virtual_robot.get_tool_pose('left')
        arrived = False
        target = np.eye(4)

        while not arrived:

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

                # qdot = np.concatenate([qdot_r, qdot_l])
                self._arms_vel_controller_pub['right'].publish(
                    PR2Controller._joint_group_command_to_msg(qdot_r))
                self._arms_vel_controller_pub['left'].publish(
                    PR2Controller._joint_group_command_to_msg(qdot_l))
                
            controller.store_joint_velocities('left', qdot_l)
            controller.store_joint_velocities('right', qdot_r)
            controller.store_drift()

            if arrived:
                rospy.loginfo('Arrived at the target')
                rospy.signal_shutdown('Done')

                
            self._rate.sleep()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
