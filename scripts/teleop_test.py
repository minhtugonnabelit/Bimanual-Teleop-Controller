#!/usr/bin/env python3

import numpy as np
import time

import rospy
from sensor_msgs.msg import Joy

from bimanual_controller.utility import (
    CalcFuncs, CONTROL_RATE, TWIST_GAIN, joy_to_twist)
from bimanual_controller.pr2_controller import PR2Controller


def main():

    controller = PR2Controller(name='teleop_test', log_level=1, rate=CONTROL_RATE)

    rospy.loginfo('Start teleop using joystick')
    rospy.wait_for_message('/joy', Joy)

    done = False
    constraint_is_set = False
    while not done:

        #start_time = time.time()
        statt_time = rospy.Time.now()

        # ---------------------- #
        # ---------------------- #

        qdot_r = np.zeros(7)
        qdot_l = np.zeros(7)

        joy_msg = controller.get_joy_msg()

        if joy_msg[1][-3]:
            controller.move_to_neutral()

        if (joy_msg[1][-1] * joy_msg[1][-2]) and not constraint_is_set:

            PR2Controller._start_jg_vel_controller()
            rospy.sleep(1)
            constraint_is_set, _ = controller.set_kinematics_constraints()
            rospy.loginfo('Constraint is set, switching controllers')

        if not constraint_is_set:

            gripper_sides = {5: 'right', 4: 'left'}
            for button_index, side in gripper_sides.items():
                if joy_msg[1][button_index]:
                    if joy_msg[0][-1] == 1:
                        controller.open_gripper(side)
                    elif joy_msg[0][-1] == -1:
                        controller.close_gripper(side)

        if constraint_is_set:

            # Calculate the twist from the joystick message
            twist, done = joy_to_twist(joy_msg, TWIST_GAIN)
            rospy.logdebug(f'Twist: {twist}')

            if joy_msg[1][5]:  # Safety trigger to allow control signal to be sent

                # Extract the Jacobians in the middle frame using the virtual robot with joint states data from the real robot
                jacob_right = controller.get_jacobian('right')
                jacob_left = controller.get_jacobian('left')

                # Perform RMRC with DLS applied independently on each arm and the projection onto the nullspace of the constraint Jacobian
                qdot_l, qdot_r = CalcFuncs.duo_arm_qdot_constraint(jacob_left,
                                                                   jacob_right,
                                                                   twist,
                                                                   activate_nullspace=True)

            # Control signal send from this block
            controller.send_joint_velocities('right', qdot_r)
            controller.send_joint_velocities('left', qdot_l)

        # ---------------------- #
        # Record the joints data

        controller.store_joint_velocities('right', qdot_r)
        controller.store_joint_velocities('left', qdot_l)
        controller.store_drift()

        # ---------------------- #

        if done:
            rospy.loginfo('Done teleoperation.')
            rospy.signal_shutdown('Done')

        # ---------------------- #
        # ---------------------- #

        #exec_time = time.time() - start_time
        exec_time = rospy.Time.now() - statt_time
        rospy.logdebug(
            f'Calculation time: {exec_time.to_sec():.4f}')

        # Sleep to control the rate of the loop execution based on the control rate
        if exec_time.to_sec() < 1 / CONTROL_RATE:
            rospy.sleep(1/CONTROL_RATE - exec_time.to_sec())

        #total_time = time.time() - start_time
        total_time = rospy.Time.now() - statt_time
        rospy.logdebug(
            f'Total time: {total_time.to_sec():.4f}')


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    
        
