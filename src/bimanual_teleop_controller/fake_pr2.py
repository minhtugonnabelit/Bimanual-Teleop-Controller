import numpy as np
import threading

import roboticstoolbox as rtb
import spatialgeometry as geometry
from swift import Swift
from bimanual_teleop_controller.math_utils import CalcFuncs

from typing import Union
ArrayLike = Union[list, np.ndarray, tuple, set]

class FakePR2:

    r"""
    ### Class to simulate PR2 on Swift environment

    This will initialize the Swift environment with robot model without any ROS components. 
    This model will be fed with joint states provided and update the visualization of the virtual frame . """

    def __init__(self, control_rate, launch_visualizer = False) -> None:

        self._control_rate = control_rate
        self._is_collapsed = False
        self._launch_visualizer = launch_visualizer
        self._robot = rtb.models.PR2()

        self._drift_error = np.zeros(6)
        self._tool_offset = {
            'l': np.eye(4),
            'r': np.eye(4)
        }

        self._arms_frame = {
            'l': {
                'end': self._robot.grippers[1],
                'start': 'l_shoulder_pan_link'
            },
            'r': {
                'end': self._robot.grippers[0],
                'start': 'r_shoulder_pan_link'
            }
        }

        # Get the joint limits and form the symmetric soft limits
        qmin, qmax = self.get_joint_limits_all()
        qmin[1] = qmin[1] * 0.7         # enforce pr2 actual hardware limits for shoulder lift joint at lower bound
        qmin[8] = qmin[8] * 0.7 
        self.qmid = (qmin + qmax) / 2

        # Normal joint with soft limits start at 80% and end at 90% of the joint limits
        self.soft_limit_start = ((qmax - qmin)/2) * 0.7
        self.soft_limit_end = ((qmax - qmin)/2) * 0.9

        self.soft_limit_range = (self.soft_limit_end - self.soft_limit_start)

        if self._launch_visualizer:
            self._env = Swift()
            self._env.launch()
            self.init_visualization()
            self._thread = threading.Thread(target=self.timeline)
            self._thread.start()

    def init_visualization(self):
        r"""
        Initialize the visualization of the robot

        :return: None
        """

        self._left_ax = geometry.Axes(length=0.05, pose=self._robot.fkine(
            self._robot.q, end=self._arms_frame['l']['end'], tool=self._tool_offset['l']).A)

        self._right_ax = geometry.Axes(length=0.05, pose=self._robot.fkine(
            self._robot.q, end=self._arms_frame['r']['end'], tool=self._tool_offset['r']).A)

        self._env.add(self._robot)
        self._env.add(self._left_ax)
        self._env.add(self._right_ax)
    
    def timeline(self):
        r"""
        Timeline function to update the visualization
        :return: None
        """

        while not self._is_collapsed:
            self._env.step(1/self._control_rate)

    def set_constraints(self, virtual_pose: np.ndarray):
        r"""
        Set the kinematics constraints for the robot
        :param virtual_pose: virtual pose on robot base frame
        :return: None
        """

        self._tool_offset['l'] = np.linalg.inv(self._robot.fkine(
            self._robot.q, end=self._arms_frame['l']['end'])) @ virtual_pose
        self._tool_offset['r'] = np.linalg.inv(self._robot.fkine(
            self._robot.q, end=self._arms_frame['r']['end'])) @ virtual_pose

        return True

    def set_states(self, joint_states, real_robot=True):
            r"""
            Set the joint states of the arms only
            :param joint_states: list of joint states
            :return: None
            """

            if real_robot:

                right_js = CalcFuncs.reorder_values(joint_states.position[17:24])
                left_js = CalcFuncs.reorder_values(joint_states.position[31:38])

                right_jv = CalcFuncs.reorder_values(joint_states.velocity[17:24])
                left_jv = CalcFuncs.reorder_values(joint_states.velocity[31:38])

            else:

                right_js = joint_states.position[0:7]
                left_js = joint_states.position[7:14]

                right_jv = joint_states.velocity[0:7]
                left_jv = joint_states.velocity[7:14]

            self._robot.q[16:23] = right_js
            self._robot.q[23:30] = left_js

            self._robot.qd[16:23] = right_jv
            self._robot.qd[23:30] = left_jv

            if self._launch_visualizer:
                self._left_ax.T = self._robot.fkine(
                    self._robot.q, end=self._arms_frame['l']['end'], tool=self._tool_offset['l']).A
                self._right_ax.T = self._robot.fkine(
                    self._robot.q, end=self._arms_frame['r']['end'], tool=self._tool_offset['r']).A

    def get_tool_pose(self, side: str, offset=True):
        r"""
        Get the tool pose of the robot
        :param side: side of the robot
        :param offset: include tool offset or not

        :return: tool pose
        """

        tool = self._tool_offset[side] if offset else np.eye(4)

        return self._robot.fkine(self._robot.q, end=self._arms_frame[side]['end'], tool=tool).A

    def get_joint_positions(self, side: str):
        r"""
        Get the joint states of the robot
        :return: joint states
        """

        if side == 'l':
            return self._robot.q[23:30]
        else:
            return self._robot.q[16:23]
    
    def get_joint_velocities(self, side: str):
        r"""
        Get the joint velocities of the robot
        :return: joint velocities
        """

        if side == 'l':
            return self._robot.qd[23:30]
        else:
            return self._robot.qd[16:23]
        
    def get_joint_limits(self, side : str):
        r"""
        Get the joint limits of the robot
        :param side: side of the robot
        :return: joint limits
        """

        return self._robot.qlim[0][16:23] if side == 'r' else self._robot.qlim[0][23:30], self._robot.qlim[1][16:23] if side == 'r' else self._robot.qlim[1][23:30]
        
    def get_joint_limits_all(self):
        r"""
        Get the joint limits of the robot
    
        First 7 joints are for the left arm and the next 7 joints are for the right arm
        :return: joint limits   
        """

        qmin_l, qmax_l = self.get_joint_limits('l')
        qmin_r, qmax_r = self.get_joint_limits('r')
        qmin = np.r_[qmin_l, qmin_r]
        qmax = np.r_[qmax_l, qmax_r]

        return qmin, qmax

    def get_jacobian(self, side : str, q : Union[list, np.ndarray, tuple, set, None] = None):
        r"""
        Get the Jacobian of the robot on the tool frame
        :param side: side of the robot

        :return: Jacobian
        """

        if q is None:
            q = self._robot.q.copy()
        return self._robot.jacobe(q, end=self._arms_frame[side]['end'], start=self._arms_frame[side]['start'], tool=self._tool_offset[side])
    
    def manipulability_gradient(self, side : str, eps : float = 1e-3):
        r"""
        Get the manipulability gradient of the robot - numerical differentiation.
        Please note that this function is computationally expensive

        :param side: side of the robot
        :param eps: epsilon value for numerical differentiation

        :return: manipulability gradient
        """
        # self._robot.jacobm()
        # start_index = 23 if side == 'l' else 16
        # manip_grad = np.zeros(7)
        # for i in range(len(manip_grad)):

        #     q_up = self._robot.q.copy()
        #     q_up[start_index + i] += eps
        #     j_up = self.get_jacobian(side = side, q = q_up)
        #     m_up = CalcFuncs.manipulability(j_up)

        #     q_low = self._robot.q.copy()
        #     q_low[start_index + i] -= eps
        #     j_low = self.get_jacobian(side = side, q = q_low)
        #     m_low = CalcFuncs.manipulability(j_low) 

        #     manip_grad[i] = (m_up - m_low) / (2 * eps)
            
        manip = self._robot.jacobm(end=self._arms_frame[side]['end'], start=self._arms_frame[side]['start'])
        manip = manip.reshape(7)
        # print(manip)
        return manip

    def get_twist_in_tool_frame(self, side : str, twist : np.ndarray):
        r"""
        Get the twist in the tool frame by 
        converting the twist in the base frame 
        to the tool frame using adjoint matrix
        
        :param side: side of the robot
        :param twist: twist in the base frame

        :return: twist in the tool frame
        """

        tool = self.get_tool_pose(side) 
        twist_converted= CalcFuncs.adjoint(np.linalg.inv(tool)) @ twist

        return twist_converted

    def joint_limits_damper(self, qdot : np.ndarray, dt : float, steepness=10):
        r"""
        Repulsive potential field for joint limits for both arms
        :param qdot: joint velocities
        :param steepness: steepness of the transition
        :return: repulsive velocity potential field 
        """

        # Get the joint positions for next step
        q = np.r_[self.get_joint_positions(
            'l'), self.get_joint_positions('r')] + qdot * dt
        
        x = np.zeros(qdot.shape[0])
        for i in range(len(x)):
            qi, qm, qsls, qslr = q[i], self.qmid[i],  self.soft_limit_start[i], self.soft_limit_range[i]
            a = np.abs(qi - qm)
            x[i] = np.round((a-qsls) / qslr , 4)

        weights = CalcFuncs.weight_vector(x, steepness)
        qdot_repulsive = - weights.max() * qdot

        return qdot_repulsive, weights.max(), np.where(weights == weights.max())[0]
    
    def joint_acceleration_damper(self, qd_command, dt, steepness=10):
        r"""
        velocity damper for joint acceleration limits

        Joint acceleration will be calculated as taking the error 
        between the current joint velocities and the next joint velocity commanded over interval dt

        the limit is set to 0.5 rad/s^2
        """
        qd_cur_l = self.get_joint_velocities('l')
        qd_cur_r = self.get_joint_velocities('r')
        qd_cur = np.r_[qd_cur_l, qd_cur_r]

        qdd = (qd_command - qd_cur) / dt
        

        # setup objective function with the error betwwen the commanding qd and the desired qd as qd_command
        

        # setup inequality constraint with the joint acceleration limits which is differentiated from the joint velocities
        
        

        pass
    
    def joint_limits_damper_side(self, side, qdot, dt, steepness=10):
        r"""
        Repulsive potential field for joint limits for both arms
        :param qdot: joint velocities
        :param steepness: steepness of the transition
        :return: repulsive velocity potential field 
        """
        # Get the joint positions for next step
        q = self.get_joint_positions(side) + qdot * dt
        arm_offset = 0 if side == 'l' else 7
        
        x = np.zeros(qdot.shape[0])
        for i in range(len(x)):
            qi, qm, qsls, qslr = q[i], self.qmid[i+arm_offset],  self.soft_limit_start[i+arm_offset], self.soft_limit_range[i+arm_offset]
            a = np.abs(qi - qm)
            x[i] = np.round((a-qsls) / qslr , 4)

        weights = CalcFuncs.weight_vector(x, steepness)
        qdot_repulsive = - weights.max() * qdot

        return qdot_repulsive, weights.max(), np.where(weights == weights.max())[0]


    def task_drift_compensation(self,  gain_p = 5, gain_d = 0.5, on_taskspace=True) -> np.ndarray:
        r"""
        Get the drift compensation for the robot
        :return: drift compensation velocities as joint velocities
        """
    
        v, _, self._drift_error = CalcFuncs.pd_servo(self.get_tool_pose(side='l', offset=True),
                                                    self.get_tool_pose(side='r', offset=True),
                                                    self._drift_error,
                                                    gain_p=gain_p,
                                                    gain_d=gain_d,
                                                    threshold=0.001,
                                                    method='angle-axis',
                                                    dt=1/self._control_rate)

        if on_taskspace:
            return v
        else:
            qdot_fix_left = CalcFuncs.rmrc(self.get_jacobian('l'), v, w_thresh=0.05)
            qdot_fix_right = CalcFuncs.rmrc(self.get_jacobian('r'), -v, w_thresh=0.05)

            return np.r_[qdot_fix_left, qdot_fix_right]
        
    def shutdown(self):
        r"""
        Get the joint states of the robot
        :return: joint states
        """
        if self._launch_visualizer:
            self._is_collapsed = True
            self._thread.join()
        return True

