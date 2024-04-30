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

    # State variables
    constraint_is_set = False
    done = False

    while not done:

        start_time = time.perf_counter()
        # start_time = rospy.Time.now()

        # ---------------------- #
        # ---------------------- #

        qdot = np.zeros(14)

        joy_msg = controller.get_joy_msg()

        # Trigger the controller to move to neutral position
        if joy_msg[1][-3]:
            controller.move_to_neutral()

        # Set the kinematic constraints for BMCP and start joint group velocity controller
        if (joy_msg[1][-1] * joy_msg[1][-2]) and not constraint_is_set:

            controller.start_jg_vel_controller()
            rospy.sleep(1)
            constraint_is_set, _ = controller.set_kinematics_constraints()
            rospy.loginfo('Constraint is set, switching controllers')

        # Once constraint is set, start the teleoperation using 
        if constraint_is_set:

            # Exrtact the twist from the joystick message
            twist, done = joy_to_twist(joy_msg, TWIST_GAIN)
            rospy.logdebug(f'Twist: {twist}')

            if joy_msg[1][5]:  # Safety trigger to allow control signal to be sent

                # Extract the Jacobians in the middle frame using the virtual robot with joint states data from the real robot
                jacob_right = controller.get_jacobian(side = 'right')
                jacob_left = controller.get_jacobian(side = 'left')
                jacob_constraint = np.c_[jacob_left, -jacob_right]

                qdot_right = CalcFuncs.rmrc(jacob_right, twist, w_thresh=0.1)
                qdot_left = CalcFuncs.rmrc(jacob_left, twist,  w_thresh=0.1)        
                qdot_combined = np.r_[qdot_left, qdot_right]



                # Perform nullspace projection for qdot_combined on constraint Jacobian to ensure the twist synchronisation
                qdot = CalcFuncs.nullspace_projector(jacob_constraint) @ qdot_combined

            # Control signal send from this block
            controller.send_joint_velocities('right', qdot[7:])
            controller.send_joint_velocities('left', qdot[:7])

        # ---------------------- #s
        # Record the joints data

        controller.store_joint_velocities('right', qdot[7:])
        controller.store_joint_velocities('left', qdot[:7])
        controller.store_drift()

        # ---------------------- #

        if done:
            rospy.loginfo('Done teleoperation.')
            rospy.signal_shutdown('Done')

        # ---------------------- #
        # ---------------------- #

        exec_time = time.perf_counter() - start_time
        rospy.logdebug(
            f'Calculation time: {exec_time:.4f}')
        
        if exec_time < 1 / CONTROL_RATE:
            rospy.sleep(1/CONTROL_RATE - exec_time)

        total_time = time.perf_counter() - start_time
        rospy.logdebug(
            f'Total time: {total_time:.4f}')
        

        # exec_time = rospy.Time.now() - start_time
        # rospy.logdebug(
        #     f'Calculation time: {exec_time.to_sec():.4f}')

        # # Sleep to control the rate of the loop execution based on the control rate
        # if exec_time.to_sec() < 1 / CONTROL_RATE:
        #     rospy.sleep(1/CONTROL_RATE - exec_time.to_sec())

        # #total_time = time.time() - start_time
        # total_time = rospy.Time.now() - start_time
        # rospy.logdebug(
        #     f'Total time: {total_time.to_sec():.4f}')


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    