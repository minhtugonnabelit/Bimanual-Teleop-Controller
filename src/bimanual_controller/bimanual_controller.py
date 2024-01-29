import numpy as numpy
import spatialmath as sm
import spatialmath.base as smb

class BimanualController():

    """
    Class to control the bimanual robot.\n
    Given the velocity of the target pose and the current pose of the object, this controller will compute the velocity of the end-effector of each arm.\n
    The controller is based on the paper: "A Bimanual Coordination Framework for Humanoid Robots" by Mistry et al.\n
    The controller is implemented in the operational space and it is composed by two main components:\n
    - The first component is the primary task controller using the constraint jacobian, which is used to control the end-effector of each arm.\n
    - The first option for 2nd component is the null-space projection to minimize the error when each arm is near the singular condition.\n
    - The second option for 2nd component is picking the velocity with more damped to be the target velocity, which doesnt need to apply projection upon the vel.\n

    """
            
    def __init__(self, l_arm, r_arm, base_tf, dt=0.01):

        self.base_tf = sm.SE3(base_tf)
        self.l_arm = l_arm
        self.r_arm = r_arm



        pass
    
    def set_target_velocity(self, target):
        """
        Set the target velocity for the bimanual controller, which is the pose to be control objective.\n
        This frame is set by the median of the relative TF between two arms end-effector and is used to define the constraint Jacobian matrix J.\n"""

        pass

    def nullspace_projection(self, J, Jbar):
        """
        Compute the nullspace projection matrix N, which is used to control the posture of the robot.\n
        The nullspace projection matrix N is computed as: N = I - J' * J.\n
        """
        pass

    def constraint_jacobian(self, Jl, Jr):
        """
        Compute the constraint Jacobian matrix J.\n
        The constraint Jacobian matrix J is computed as: J = Jr - Jl.\n
        """
        pass

