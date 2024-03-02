# Require libraries
import numpy as np
import matplotlib.pyplot as plt
from spatialmath.base import *
from roboticstoolbox import models, jtraj, trapezoidal

# Useful variables
from math import pi

# -----------------------------------------------------------------------------------#
def lab4_solution_question2_and_3():
    """
    Lab 4 - Question 2 & 3 - Inverse Kinematics & Joint Interpolation
    """

    plt.close('all')

    ## Options
    # interpolation = 2                                                                                       # 1 = Quintic Polynomial, 2 = Trapezoidal Velocity                       
    steps = 50                                                                                              # Specify no. of steps

    ## Load model
    p560 = models.DH.Puma560()
    qlim = np.transpose(p560.qlim)

    ## Define End-Effector transformation, use inverse kinematics to get joint angles
    T1 = transl(0.5,-0.4,0.5)                                                                               # Create translation matrix
    q1 = p560.ikine_LM(T1, q0 = np.zeros([1,6])).q                                                          # Derive joint angles for required end-effector transformation
    T2 = transl(0.5,0.4,0.1)                                                                                # Define a translation matrix            
    q2 = p560.ikine_LM(T2, q0 = np.zeros([1,6])).q      
    
    # q1 = [-0.4382, -0.5942, -0.0178, 0.0000, -0.4550, -0.9113]  
    # q2 = [0.9113, 0.2842, -0.7392, -3.1416, 0.6120, -2.7034]                                                  # Use inverse kinematics to get the joint angles

    ## Interpolate joint angles, also calculate relative velocity, accleration
    def get_traj(interpolation):
        if interpolation == 1:
            q_matrix = jtraj(q1, q2, steps).q
        elif interpolation == 2:
            s = trapezoidal(0, 1, steps).q                                                                      # Create the scalar function
            q_matrix = np.empty((steps, 6))                                                                     # Create memory allocation for variables
            for i in range(steps):
                q_matrix[i, :] = (1 - s[i]) * q1 + s[i] * q2                                                    # Generate interpolated joint angles
        else:
            raise ValueError("interpolation = 1 for Quintic Polynomial, or 2 for Trapezoidal Velocity")
    
        return q_matrix
    
    q_matrix = get_traj(1)
    q_matrix2 = get_traj(2)


    velocity = np.zeros([steps, 6])
    acceleration = np.zeros([steps, 6])
    for i in range(1,steps):
        velocity[i,:] = q_matrix[i,:] - q_matrix[i-1,:]
        acceleration[i,:] = velocity[i,:] - velocity[i-1,:]

    velocity2 = np.zeros([steps, 6])
    acceleration2 = np.zeros([steps, 6])
    for i in range(1,steps):
        velocity2[i,:] = q_matrix2[i,:] - q_matrix2[i-1,:]
        acceleration2[i,:] = velocity2[i,:] - velocity2[i-1,:]   

    vel_diff = abs(velocity -velocity2)
    print('max', np.max(vel_diff))

    ## Plot the results
    fig = plt.figure(1)
    fig = p560.plot(q1, fig = fig)
    ax = plt.gca()
    for q in q_matrix: # Plot the motion between poses, draw a red line of the end-effector path
        p560.q = q
        ee_pos = transl(p560.fkine(q).A) # End-effector position 
        # ax.plot(ee_pos[0],ee_pos[1],ee_pos[2], color = 'red', marker = '.', markersize= 3)
        fig.step(0.05)

    # Plot joint angles
    plt.figure(2)
    for i in range(6):
        plt.subplot(3, 2, i+1)
        plt.plot(q_matrix[:, i], 'k', linewidth=1)
        plt.title('Joint ' + str(i+1))
        plt.xlabel('Step')
        plt.ylabel('Joint Angle (rad)')
        plt.axhline(qlim[i, 0], color='r')  # Plot lower joint limit
        plt.axhline(qlim[i, 1], color='r')  # Plot upper joint limit

    # Plot joint velocities
    plt.figure(3)
    for i in range(6):
        plt.subplot(3, 2, i+1)
        plt.plot(velocity[:, i], 'k', linewidth=1)
        plt.title('Joint ' + str(i+1))
        plt.xlabel('Step')
        plt.ylabel('Joint Velocity')

    # Plot joint accelerations
    plt.figure(4)
    for i in range(6):
        plt.subplot(3, 2, i+1)
        plt.plot(acceleration[:, i], 'k', linewidth=1)
        plt.title('Joint ' + str(i+1))
        plt.xlabel('Step')
        plt.ylabel('Joint Acceleration')

    input('Enter to exit\n')

# ---------------------------------------------------------------------------------------#
if __name__ == "__main__":
    lab4_solution_question2_and_3()