import numpy as np
import sys
from scipy import linalg

import spatialmath as sm
import matplotlib.pyplot as plt
import spatialgeometry as geometry
import roboticstoolbox as rtb
from swift import Swift

# Import custom utility functions
from utility import *

pr2 = rtb.models.PR2()
qtest = np.zeros(31)

# Set the initial joint angles
qtest[16:23] = [-np.pi/6, np.pi/6, -np.pi/3, -np.pi/2, 0, -np.pi/4, np.pi/2]
qtest[23:30] = [np.pi/6, np.pi/6, np.pi/3, -np.pi/2, 0, -np.pi/4, np.pi/2]
pr2.q = qtest

env = Swift()
env.set_camera_pose([1, 0, 1], [0, 0, 1])
env.launch()

# Set the initial pose of the end-effectors
left_tip_pose = pr2.fkine(pr2.q, end=pr2.grippers[1], ).A
right_tip_pose = pr2.fkine(pr2.q, end=pr2.grippers[0], ).A
l2r = linalg.inv(sm.SE3(left_tip_pose)) @ right_tip_pose
left_ax = geometry.Axes(length=0.05, pose=left_tip_pose)
right_ax = geometry.Axes(length=0.05, pose=right_tip_pose)

# Extract the middle point between the two tools
joined = sm.SE3()
joined.A[0:3, 3] = (left_tip_pose[0:3, 3] + right_tip_pose[0:3, 3]) / 2
joined.A[0:3, 0:3] = np.eye(3)
joined_ax = geometry.Axes(length=0.05, pose=joined.A)
joined_in_left = linalg.inv(sm.SE3(left_tip_pose)) @ joined.A
joined_in_right = linalg.inv(sm.SE3(right_tip_pose)) @ joined.A

env.add(pr2)
env.add(left_ax)
env.add(right_ax)

df = list()
dt = 0.015

joy = joy_init()
LIN_G = 0.05
ANG_G = 0.05

while True:

    # ---------------------------------------------------------------------------#
    # SECTION TO HANDLE THE JOYSTICK INPUT
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    if joy.get_button(0):
        break

    twist = joy_to_twist(joy, [LIN_G, ANG_G])


    # ---------------------------------------------------------------------------#
    # SECTION TO PERFORMS TWIST TRANSFORMATION IN A RIGID BODY MOTION
    jacob_l = pr2.jacobe(pr2.q,
                            end=pr2.grippers[1],
                            start="l_shoulder_pan_link",  
                            tool=sm.SE3(joined_in_left))  # Jacobian of the left arm within tool frame
    
    jacob_r = pr2.jacobe(pr2.q, 
                            end=pr2.grippers[0], 
                            start="r_shoulder_pan_link", 
                            tool=sm.SE3(joined_in_right))  # Jacobian of the right arm within tool frame
    
    # Calculate the joint velocities using the Resolved Motion Rate Control (RMRC) method with the projection onto nullspace of Constraint Jacobian
    qdot_l, qdot_r = duo_arm_qdot_constraint(jacob_l, jacob_r, twist,)


    # ---------------------------------------------------------------------------#
    # SECTION TO UPDATE VISUALIZATION AND RECORD THE NECESSARY DATA
    pr2.q[16:23] = pr2.q[16:23] + qdot_r * dt  # right arm
    pr2.q[23:30] = pr2.q[23:30] + qdot_l * dt   # left arm

    # Visualization of the frames
    left_tip_pose = pr2.fkine(pr2.q, end=pr2.grippers[1], ).A
    right_tip_pose = pr2.fkine(pr2.q, end=pr2.grippers[0], ).A
    updated_joined_left = left_tip_pose @ joined_in_left
    updated_joined_right = right_tip_pose @ joined_in_right
    left_ax.T = updated_joined_left
    right_ax.T = updated_joined_right

    # Record the distance between offset frames of each arm to  observe the drift of tracked frame
    dis = np.linalg.norm( updated_joined_left[0:3, 3] - updated_joined_right[0:3, 3])
    df.append(dis)

    env.step()


# Record and plot the distance between offset frames of each arm to  observe the drift of tracked frame
plt.figure(1)
plt.plot(df, 'k', linewidth=1)
plt.title('Drift graph')
plt.xlabel('Time')
plt.ylabel('Distance')

plt.show()
# env.hold()

