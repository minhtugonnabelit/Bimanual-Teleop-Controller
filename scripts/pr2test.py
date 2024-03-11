
import roboticstoolbox as rtb
from swift import Swift     

import spatialgeometry as geometry
import spatialmath.base as smb
import spatialmath as sm
import matplotlib.pyplot as plt

import numpy as np
from scipy import linalg

from utility import *

pr2 = rtb.models.PR2()
qtest = np.zeros(31)

# Set the initial joint angles
qtest[16:23] = [-pi/6,pi/6,-pi/3,-pi/2,0,-pi/4,pi/2]
qtest[23:30] = [pi/6,pi/6,pi/3,-pi/2,0,-pi/4,pi/2]
pr2.q = qtest

env = Swift()
env.set_camera_pose([1, 0, 1], [0,0,1])
env.launch()

# Set the initial pose of the end-effectors
left_tip_pose = pr2.fkine(pr2.q, end=pr2.grippers[1], ).A 
right_tip_pose = pr2.fkine(pr2.q, end=pr2.grippers[0], ).A
l2r = linalg.inv(sm.SE3(left_tip_pose)) @ right_tip_pose
left_ax = geometry.Axes(length=0.05, pose = left_tip_pose ) 
right_ax = geometry.Axes(length=0.05, pose = right_tip_pose )


# Extract the middle point between the two tools
joined = sm.SE3()
joined.A[0:3,3] = (left_tip_pose[0:3,3] + right_tip_pose[0:3,3] ) /2
joined.A[0:3,0:3] = np.eye(3)
joined_ax = geometry.Axes(length=0.05, pose = joined.A )
joined_in_left = linalg.inv(sm.SE3(left_tip_pose)) @ joined.A
joined_in_right = linalg.inv(sm.SE3(right_tip_pose)) @ joined.A


# Target pose with full twist motion
target = joined.A @ sm.SE3(0.1, 0.1, 0.1).A @ sm.SE3.RPY(0.3,0.1,0.1).A
target1 = joined.A @ sm.SE3(0.0, 0.2, 0.1).A @ sm.SE3.RPY(0.1,0.3,0.1).A
target2 = joined.A @ sm.SE3(0.1, -0.1, 0.1).A @ sm.SE3.RPY(0.0,0.1,0.2).A


traj = list()
traj.append(target)
traj.append(target1)
traj.append(target2)

# Target pose for test case: Only angular motion
# target = joined.A @  sm.SE3.RPY(0.2,0,0).A

# Target pose for the drift test case: Only linear motion
# target = joined.A @ sm.SE3(0.1, -0.2, -0.1).A 

target_ax = geometry.Axes(length=0.05, pose = target )
target1_ax = geometry.Axes(length=0.05, pose = target1 )
target2_ax = geometry.Axes(length=0.05, pose = target2 )

env.add(pr2)
env.add(left_ax )
env.add(right_ax )
env.add(target_ax)
env.add(target1_ax)
env.add(target2_ax)


df = list()
dt_f = list()
dt = 0.015

updated_joined_left = left_tip_pose @ joined_in_left

for target in traj:
    arrived = False

    while not arrived:

        # ---------------------------------------------------------------------------#
        # SECTION TO PERFORMS SERVOING IN THE VIRTUAL MIDDLE FRAME
        
        middle_twist, angle, axis, arrived = rtb.p_servo(updated_joined_left, 
                                            target, 
                                            gain = 0.07, 
                                            threshold=0.01, 
                                            method='angle-axis')  # Servoing in the virtual middle frame using angle-axis representation for angular error


        # Update the virtual joined frame with the twist
        exp_twist = smb.trexp(middle_twist * dt)    # Exponential mapping
        joined = joined @ sm.SE3(exp_twist)         # Update the joined frame
        joined_ax.T = joined                        # Update the visualization of the joined frame


        # ---------------------------------------------------------------------------#
        # SECTION TO PERFORMS TWIST TRANSFORMATION IN A RIGID BODY MOTION  

        # Proper way to calculate the twist for each arm with Jacobian transformation
        jacob_l = pr2.jacobe(pr2.q, end=pr2.grippers[1], start="l_shoulder_pan_link",  tool = sm.SE3(joined_in_left) )  # Jacobian of the left arm within the end-effector frame
        jacob_r = pr2.jacobe(pr2.q, end=pr2.grippers[0], start="r_shoulder_pan_link", tool = sm.SE3(joined_in_right) )  # Jacobian of the right arm within the end-effector frame
        jacob_c = np.concatenate([jacob_l, -jacob_r], axis=1)                                                           # Concatenate Jacobians of both arms

        qdot_left = rmrc(jacob_l, middle_twist, p_only = False)
        qdot_right = rmrc(jacob_r, middle_twist,  p_only = False)
        qdotc = np.concatenate([qdot_left, qdot_right], axis=0)     # Concatenate joint velocities of both arms
        qdotc = nullspace_projection(jacob_c) @ qdotc               # Project the joint velocities to the null space of the Jacobian to maintain the constraint Jacobian * qdot = twist


        # ---------------------------------------------------------------------------#
        # SECTION TO UPDATE VISUALIZATION AND RECORD THE NECESSARY DATA

        # Update the joint angles
        pr2.q[16:23] = pr2.q[16:23] + qdotc[7:14] * dt  # right arm
        pr2.q[23:30] = pr2.q[23:30] + qdotc[0:7] * dt   # left arm

        # Visualization of the frames
        left_tip_pose = pr2.fkine(pr2.q, end=pr2.grippers[1], ).A 
        right_tip_pose = pr2.fkine(pr2.q, end=pr2.grippers[0], ).A
        updated_joined_left = left_tip_pose @ joined_in_left
        updated_joined_right = right_tip_pose @ joined_in_right
        left_ax.T = updated_joined_left
        right_ax.T = updated_joined_right

        # Record the distance between offset frames of each arm to  observe the drift of tracked frame
        dis = np.linalg.norm(updated_joined_left[0:3,3] - updated_joined_right[0:3,3])
        df.append(dis)

        env.step()


# Record and plot the distance between offset frames of each arm to  observe the drift of tracked frame
plt.figure(1)
plt.plot(df, 'k', linewidth=1)
plt.title('Drift graph')
plt.xlabel('Time')
plt.ylabel('Distance')

plt.figure(3)
plt.plot(np.diff(df), 'k', linewidth=1)
plt.title('Drift difference graph')
plt.xlabel('Time')
plt.ylabel('Distance')
plt.xscale('linear')

plt.show()
env.hold()

# Calculate the target frames for each arm
# left_target = joined.A @ linalg.inv(joined_in_left)
# left_twist, _, _ = angle_axis_python(left_tip_pose, left_target)

# righ_target = joined.A @ linalg.inv(joined_in_right)
# right_twist, _, _ = angle_axis_python(right_tip_pose, righ_target)

# jacob_l = pr2.jacob0(pr2.q, end=pr2.grippers[1], start="l_shoulder_pan_link")  # Jacobian of the left arm within the end-effector frame
# jacob_r = pr2.jacob0(pr2.q, end=pr2.grippers[0], start="r_shoulder_pan_link")  # Jacobian of the right arm within the end-effector frame

# qdot_left = rmrc(jacob_l, left_twist, p_only = False)
# qdot_right = rmrc(jacob_r, right_twist,  p_only = False)




