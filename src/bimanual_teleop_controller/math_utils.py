
import numpy as np
import spatialmath.base as smb
import spatialmath as sm
import math
import time
from typing import Union
from numpy.typing import ArrayLike


class CalcFuncs:
    r"""
    Class to calculate the necessary functions for the robot control.
    """
    
    @staticmethod    
    def adjoint(T):
        r"""
        Adjoint transformation matrix.

        :param T: A 4x4 homogeneous transformation matrix
        :type T: np.ndarray

        :return: The adjoint transformation matrix
        :rtype: np.ndarray

        """

        R = T[:3, :3]
        p = T[:3, -1]

        ad = np.eye(6, 6)
        ad[:3, :3] = R
        ad[3:, 3:] = R
        ad[3:, :3] = np.cross(p, R)

        return ad

    @staticmethod    
    def manipulability(jacob):
        r"""
        Calculate the manipulability of the robot.
        
        :param jacob: The Jacobian matrix of the robot
        :type jacob: np.ndarray

        :return: The manipulability of the robot
        :rtype: float

        """
        return np.sqrt(np.linalg.det(jacob @ np.transpose(jacob)))
    
    @staticmethod    
    def rmrc(jacob, twist, w_thresh=0.08):
        """
        Description of what the function does.

        Parameters
        ----------
        jacob : np.ndarray
            Jacobian matrix of the robot.
        twist : list
            The twist of the robot.
        
        Returns
        -------
        list 
            The joint velocities of the robot.

        """
        w = CalcFuncs.manipulability(jacob)

        max_damp = 1
        damp = np.power(1 - w/w_thresh, 2) * max_damp if w < w_thresh else 0

        j_dls = np.transpose(jacob) @ np.linalg.inv(jacob @
                                                    np.transpose(jacob) + damp * np.eye(6))

        qdot = j_dls @ np.transpose(twist)
        return qdot
    
    @staticmethod
    def world_twist_to_qdot(ee_pose : np.ndarray, twist : list, jacob : np.ndarray, manip_thresh) -> list:
        adjoint = CalcFuncs.adjoint(np.linalg.inv(ee_pose))
        twist = adjoint @ twist
        qdot = CalcFuncs.rmrc(jacob, twist, w_thresh=manip_thresh)

        return qdot, twist

    @staticmethod
    def nullspace_projector(m):
        r"""
        Calculate the nullspace projector of the matrix.
        
        Parameters
        ----------
        m : np.ndarray
            The matrix to calculate the nullspace projector.

        Returns
        -------
        np.ndarray
            The nullspace projector of the matrix.
        """
        return np.eye(m.shape[1]) - np.linalg.pinv(m) @ m
    
    @staticmethod
    def duo_arm_qdot_constraint(jacob_l, jacob_r, twist, activate_nullspace=True):
        qdot_left = CalcFuncs.rmrc(jacob_l, twist)
        qdot_right = CalcFuncs.rmrc(jacob_r, twist)

        qdotc = np.r_[qdot_left, qdot_right]
        jacob_c = np.c_[jacob_l, -jacob_r]

        if activate_nullspace:
            nullspace_projection_matrix = np.eye(
                jacob_c.shape[1]) - np.linalg.pinv(jacob_c) @ jacob_c
            qdotc = nullspace_projection_matrix @ qdotc

        qdot_l = qdotc[0:jacob_l.shape[1]]
        qdot_r = qdotc[jacob_l.shape[1]:]

        return qdot_l, qdot_r
    
    @staticmethod
    def angle_axis_python(T, Td):
        r"""
        Calculate the error between two poses using the angle-axis representation.
        
        Parameters
        ----------
        T : np.ndarray
            The current pose of the robot.
        Td : np.ndarray
            The desired pose of the robot.

        Returns
        -------
        np.ndarray
            The error between the two poses.

        """
        e = np.empty(6)
        e[:3] = Td[:3, -1] - T[:3, -1]
        R = Td[:3, :3] @ T[:3, :3].T
        li = np.array([R[2, 1] - R[1, 2], R[0, 2] -
                      R[2, 0], R[1, 0] - R[0, 1]])
        ln = smb.norm(li)

        if smb.iszerovec(li):
            # diagonal matrix case
            if np.trace(R) > 0:
                a = np.zeros((3,))
            else:
                a = np.pi / 2 * (np.diag(R) + 1)
        else:
            # non-diagonal matrix case
            a = math.atan2(ln, np.trace(R) - 1) * li / ln


        e[3:] = a

        return e
    
    @staticmethod
    def pd_servo(
        wTe, 
        wTep, 
        prev_error: np.ndarray, 
        gain_p: Union[float, ArrayLike] = 1.0, 
        gain_d: Union[float, ArrayLike] = 0.0,
        threshold=0.1, 
        method='angle-axis',
        dt=0.001
    ):
        r"""
        Position-based servoing.

        Parameters
        ----------
        wTe : np.ndarray
            The current end effector pose of the robot.
        wTep : np.ndarray
            The desired end effector pose of the robot.
        prev_error : np.ndarray
            The previous error of the end effector pose in time interval dt.
        gain_p : Union[float, ArrayLike], optional
            The proportional gain of the robot, by default 1.0.
        gain_d : Union[float, ArrayLike], optional
            The derivative gain of the robot, by default 0.0.
        threshold : float, optional
            The threshold of the robot, by default 0.1.
        method : str, optional
            The method to calculate the error, by default 'angle-axis'.
        dt : float, optional
            The time step of the robot, by default 0.001.

        Returns
        -------
        np.ndarray
            The velocity of the robot.
        bool
            The arrival status of the robot.
        np.ndarray
            The error of the end effector pose.

        """

        if isinstance(wTe, sm.SE3):
            wTe = wTe.A

        if isinstance(wTep, sm.SE3):
            wTep = wTep.A
        
        p = [gain_p['linear']] * 3 + [gain_p['angular']] * 3
        d = [gain_d['linear']] * 3 + [gain_d['angular']] * 3

        kp = np.diag(p)
        kd = np.diag(d)

        if method == "rpy":
            # Pose difference
            eTep = np.linalg.inv(wTe) @ wTep
            e = np.empty(6)

            # Translational error
            e[:3] = eTep[:3, -1]

            # Angular error
            e[3:] = smb.tr2rpy(eTep, unit="rad", order="zyx", check=False)
        else:
            e = CalcFuncs.angle_axis_python(wTe, wTep)

        # Calculate the derivative of the error
        d_error = (e - prev_error) / dt

        v = kp @ e + kd @ d_error

        arrived = True if np.sum(np.abs(e)) < threshold else False

        return v, arrived, e
    
    @staticmethod
    def weight(x, k):
        return 1 / (1 + np.exp(-k * (x - 0.5)))
    
    @staticmethod
    def weight_vector(x, k):
        return np.array([CalcFuncs.weight(xi, k) for xi in x])
    
    @staticmethod
    def reorder_values(data):
        if len(data) != 7:
            raise ValueError("The length of the data should be 7")

        data_array = np.asarray(data)
        data_array[0], data_array[1], data_array[2], data_array[3], data_array[
            4] = data_array[1], data_array[2], data_array[0], data_array[4], data_array[3]

        return data_array.tolist()
    
    @staticmethod
    def debounce(last_switch_time, debounce_interval=1):
        current_time = time.time()
        if current_time - last_switch_time >= debounce_interval:
            last_switch_time = current_time
            return True, last_switch_time
        else:
            return False, last_switch_time
    
    @staticmethod
    def lerp(start, end, t):
        return start + t * (end - start)
    
class LowPassFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.prev_value = None

    def filter(self, value):
        if self.prev_value is None:
            self.prev_value = value
        filtered_value = self.alpha * value + \
            (1 - self.alpha) * self.prev_value
        self.prev_value = filtered_value
        return filtered_value