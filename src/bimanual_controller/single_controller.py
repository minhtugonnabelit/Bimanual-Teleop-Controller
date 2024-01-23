import numpy as numpy
import spatialmath as sm
import spatialmath.base as smb

class SingleController():

    def __init__(self, arm, dt=0.01):

        self.arm = arm
        self.dt = dt
        self.target = None
        self.J = None
        self.Jbar = None
        self.N = None
        self.Jinv = None
        self.Jbarinv = None
        self.Ninv = None
        self.Kp = 1
        self.Kd = 1

    def qdot_solve(self, J, Jbar, N, xdot, qdot_posture):
        """
        Solve the inverse kinematics problem to compute the joint velocity of the robot.\n
        The joint velocity is computed as: qdot = Jbar * xdot + N * qdot_posture.\n
        """
        pass

    def manipulability(self, J):
        """
        Compute the manipulability of the robot.\n
        The manipulability is computed as: m = sqrt(det(J * J')).\n
        """
        pass

    def dls(self, J, lam=0.01):
        """
        Compute the damped least square inverse of the Jacobian matrix.\n
        The damped least square inverse of the Jacobian matrix is computed as: Jbar = J' * inv(J * J' + lam^2 * I).\n
        """
        pass