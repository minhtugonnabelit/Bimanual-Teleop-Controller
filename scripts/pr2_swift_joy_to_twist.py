import numpy as np
from scipy import linalg

import spatialmath as sm
import matplotlib.pyplot as plt
import spatialgeometry as geometry
import roboticstoolbox as rtb
from swift import Swift

# Import custom utility functions
from bimanual_controller.utility import *

pr2 = rtb.models.PR2()
qtest = np.zeros(31)

# Set the initial joint angles
qtest[16:23] = [-np.pi/6, np.pi/6, -np.pi/3, -np.pi/2, 0, -np.pi/4, np.pi/2]
qtest[23:30] = [np.pi/6, np.pi/6, np.pi/3, -np.pi/2, 0, -np.pi/4, np.pi/2]
pr2.q = qtest

env = Swift()
env.set_camera_pose([1, 0, 1], [0, 0.5, 1])
env.launch()

# Set the initial pose of the end-effectors
left_tip_pose = pr2.fkine(pr2.q, end=pr2.grippers[1], ).A
right_tip_pose = pr2.fkine(pr2.q, end=pr2.grippers[0], ).A
l2r = linalg.inv(sm.SE3(left_tip_pose)) @ right_tip_pose
left_ax = geometry.Axes(length=0.05, pose=left_tip_pose, color=[1, 0, 0])
right_ax = geometry.Axes(length=0.05, pose=right_tip_pose)

# Extract the middle point between the two tools
joined = np.eye(4,4)
joined[0:3, 3] = (left_tip_pose[0:3, 3] + right_tip_pose[0:3, 3]) / 2
joined_ax = geometry.Axes(length=0.05, pose=joined,)
joined_in_left = linalg.inv(sm.SE3(left_tip_pose)) @ joined
joined_in_right = linalg.inv(sm.SE3(right_tip_pose)) @ joined

env.add(pr2)
env.add(left_ax)
env.add(right_ax)

w_l = list()
w_r = list()
df = list()
dt = 0.015

joy = joy_init()
LIN_G = 0.02
ANG_G = 0.05

done  = False

while not done:

    # ---------------------------------------------------------------------------#
    # SECTION TO HANDLE THE JOYSTICK INPUT
    twist, done = joy_to_twist(joy, [LIN_G, ANG_G], done)

    # ---------------------------------------------------------------------------#
    # SECTION TO PERFORMS TWIST TRANSFORMATION IN A RIGID BODY MOTION
    jacob_l = pr2.jacobe(pr2.q, end=pr2.grippers[1], start="l_shoulder_pan_link", tool=sm.SE3(joined_in_left))  # Jacobian of the left arm within tool frame
    jacob_r = pr2.jacobe(pr2.q, end=pr2.grippers[0], start="r_shoulder_pan_link", tool=sm.SE3(joined_in_right))  # Jacobian of the right arm within tool frame

    w_l.append(manipulability(jacob_l))
    w_r.append(manipulability(jacob_r))
    
    # Calculate the joint velocities using the Resolved Motion Rate Control (RMRC) method with the projection onto nullspace of Constraint Jacobian
    qdot_l, qdot_r = duo_arm_qdot_constraint(jacob_l, jacob_r, twist, activate_nullspace=True)

    # ---------------------------------------------------------------------------#
    # SECTION TO UPDATE VISUALIZATION AND RECORD THE NECESSARY DATA
    pr2.q[16:23] = pr2.q[16:23] + qdot_r * dt  # right arm
    pr2.q[23:30] = pr2.q[23:30] + qdot_l * dt   # left arm

    # check if any of the joint angles reach the joint limits

    for i in range(16, 30):
        if pr2.q[i] > pr2.qlim[1, i] or pr2.q[i] < pr2.qlim[0, i]:
            print(f"Joint angle of {i} reach the joint limit")
            # done = True

    # Visualization of the frames
    updated_joined_left = pr2.fkine(pr2.q, end=pr2.grippers[1], ).A @ joined_in_left
    updated_joined_right = pr2.fkine(pr2.q, end=pr2.grippers[0], ).A @ joined_in_right
    left_ax.T = updated_joined_left
    right_ax.T = updated_joined_right

    # Record the distance between offset frames of each arm to  observe the drift of tracked frame
    dis = np.linalg.norm(updated_joined_left[0:3, 3] - updated_joined_right[0:3, 3])
    df.append(dis)

    env.step()


# Record and plot the distance between offset frames of each arm to  observe the drift of tracked frame
fig, ax = plt.subplots(2,2)
ax[0,0].plot(w_l, 'r', linewidth=1)
ax[0,0].plot(w_r, 'b', linewidth=1)
ax[0,0].set_title('Manipulability graph')
ax[0,0].set_xlabel('Time')
ax[0,0].set_ylabel('Manipulability')
ax[0,0].legend(['Left arm', 'Right arm'])

ax[0,1].plot(np.diff(w_l), 'r', linewidth=1)
ax[0,1].plot(np.diff(w_r), 'b', linewidth=1)
ax[0,1].set_title('wdot')
ax[0,1].set_xlabel('Time')
ax[0,1].set_ylabel('Manipulability rate')
ax[0,1].legend(['Left arm', 'Right arm'])

ax[1,1].plot(df, 'k', linewidth=1)
ax[1,1].set_title('Drift graph')
ax[1,1].set_xlabel('Time')
ax[1,1].set_ylabel('Distance')


plt.show()
# env.hold()

