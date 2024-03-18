import time
import numpy as np

import rospy
from sensor_msgs.msg import JointState, Joy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from pr2_controllers_msgs.msg import JointTrajectoryControllerState


class PR2VelControl():

    V = 0.5

    def __init__(self):

        self._pub = rospy.Publisher('l_arm_controller/command', JointTrajectory, queue_size=1)
        self._joint_states_sub = rospy.Subscriber('l_arm_controller/state', JointTrajectoryControllerState, self._joint_states_callback)
        self._joy_sub = rospy.Subscriber('joy', Joy, self._joy_callback)

        self.rate = rospy.Rate(10)
        pass

    def _joint_states_callback(self, msg: JointTrajectoryControllerState):
        self._joint_states = msg.actual.positions
        print(self._joint_states)

    def _joy_callback(self, msg: Joy):
        self._joy = (msg.axes, msg.buttons)


    def command_to_mg(self, side: str, values: list, duration: float):

        msg = JointTrajectory()
        msg.header.frame_id = 'torso_lift_link'
        msg.header.stamp = rospy.Time.now()
        msg.joint_names = [
            f"{side}_shoulder_pan_joint",
            f"{side}_shoulder_lift_joint",
            f"{side}_upper_arm_roll_joint",
            f"{side}_forearm_roll_joint",
            f"{side}_elbow_flex_joint",
            f"{side}_wrist_flex_joint",
            f"{side}_wrist_roll_joint",
        ]
        point = JointTrajectoryPoint()
        point.positions = values
        point.time_from_start = rospy.Duration(duration)
        msg.points.append(point)

        return msg
    
    def _joy_to_joint_velocities(self):
        
        rospy.wait_for_message('/l_arm_controller/state', JointTrajectoryControllerState)
        rospy.wait_for_message('/joy', Joy)

        while not rospy.is_shutdown():
            
            print('bruh ')
            # print(self._joint_states)
            trigger = self._joy[1][4]
            dir = self._joy[0][1] / np.abs(self._joy[0][1])
            qdot = [
                self._joint_states[0] + self.V * dir * self._joy[1][0],
                self._joint_states[1] + self.V * dir * self._joy[1][1],
                self._joint_states[2] + self.V * dir * self._joy[1][2],
                self._joint_states[3] + self.V * dir * self._joy[1][3],
                self._joint_states[4] + self.V * dir * self._joy[1][5],
                self._joint_states[5] + self.V * dir * self._joy[1][6],
                self._joint_states[6] + self.V * dir * self._joy[1][7],
            ]

            # qdot = [
            #     self.V * dir * self._joy[1][0],
            #     self.V * dir * self._joy[1][1],
            #     self.V * dir * self._joy[1][2],
            #     self.V * dir * self._joy[1][3],
            #     self.V * dir * self._joy[1][5],
            #     self.V * dir * self._joy[1][6],
            #     self.V * dir * self._joy[1][7],
            # ]

            qdot = trigger * qdot
            msg = self.command_to_mg('l', qdot, 0.1)
            self._pub.publish(msg)
            self.rate.sleep()

    
    # def _handle_arm_switch(self):
    #     """
    #     Handles the arm switching logic with debouncing.
    #     """
    #     current_time = time.time()
    #     if current_time - self._last_switch_time >= self._SWITCH_DEBOUCE_INTERVAL:
    #         self._last_switch_time = current_time
    #         self._switch_arms()

    # def _switch_arms(self):
    #     r"""
    #     Switch the arms that will be controlled by the hydra.\n
    #     The arms are switched by pressing the left trigger of the hydra joystick.\n
    #     The arms are switched by changing the topic of the current twist publisher.\n
    #     """
    #     pass

    #     if self._switched:
    #         self._current_twist_pub = self._right_twist_pub
    #         self._switched = False
    #     else:
    #         self._current_twist_pub = self._left_twist_pub
    #         self._switched = True

if __name__ == '__main__':
    rospy.init_node('pr2_vel_control')
    pr2 = PR2VelControl()
    # rospy.spin()
    pr2._joy_to_joint_velocities()