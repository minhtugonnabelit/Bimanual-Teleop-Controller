import roboticstoolbox as rtb
import spatialgeometry as geometry
import spatialmath.base as smb
import spatialmath as sm
import matplotlib.pyplot as plt
from swift import Swift     

import numpy as np
from scipy import linalg
from math import pi
from copy import deepcopy

from utility import *


pr2 = rtb.models.PR2()
qtest = np.zeros(31)
qtest[16:23] = [-pi/6,pi/6,-pi/3,-pi/2,0,-pi/4,pi/2]
qtest[23:30] = [pi/6,pi/6,pi/3,-pi/2,0,-pi/4,pi/2]
pr2.q = qtest

env = Swift()
env.set_camera_pose([1, 0, 1], [0,0,1])
env.launch()

left_tip_pose = pr2.fkine(pr2.q, end=pr2.grippers[1], ).A 
right_tip_pose = pr2.fkine(pr2.q, end=pr2.grippers[0], ).A
l2r = linalg.inv(sm.SE3(left_tip_pose)) @ right_tip_pose
print(np.linalg.norm(l2r[0:3,3]))

left_ax = geometry.Axes(length=0.05, pose = left_tip_pose ) 
right_ax = geometry.Axes(length=0.05, pose = right_tip_pose )

# Extract the middle point between the two tools
joined = sm.SE3()
joined.A[0:3,3] = (left_tip_pose[0:3,3] + right_tip_pose[0:3,3] ) /2
joined.A[0:3,0:3] = np.eye(3)

joined_ax = geometry.Axes(length=0.05, pose = joined.A )

joined_in_left = linalg.inv(sm.SE3(left_tip_pose)) @ joined.A
ad_left = adjoint_transform(joined_in_left)

joined_in_right = linalg.inv(sm.SE3(right_tip_pose)) @ joined.A
ad_right = adjoint_transform(joined_in_right)

# Set the target pose 

# Target pose with full twist motion
# target = joined.A @ sm.SE3(0.0, -0.2, 0.1).A @ sm.SE3.RPY(0.1,0.1,0.1).A

# Target pose for test case: Only angular motion
target = joined.A @  sm.SE3.RPY(0.2,0,0).A

# Target pose for the drift test case: Only linear motion
# target = joined.A @ sm.SE3(0.1, -0.2, -0.1).A 

target_ax = geometry.Axes(length=0.05, pose = target )

# Add cuboid as axis head for each rotation axis
l_rot_ax_head = geometry.Cuboid([0.01,0.01,0.01], pose=sm.SE3(left_tip_pose), color = [1,0,0])
r_rot_ax_head = geometry.Cuboid([0.01,0.01,0.01], pose=sm.SE3(right_tip_pose), color = [0,1,0])


env.add(pr2)
env.add(left_ax )
env.add(right_ax )
env.add(target_ax)
env.add(joined_ax)
env.add(l_rot_ax_head)
env.add(r_rot_ax_head)

df = list()
dt_f = list()
dt = 0.015
arrived = False    
while not arrived:

    middle_twist, angle, axis, arrived = rtb.p_servo(joined, 
                                        target, 
                                        gain = 0.05, 
                                        threshold=0.01, 
                                        method='angle-axis')  # Servoing in the end-effector frame using angle-axis representation for angular error


    exp_twist = smb.trexp(middle_twist * dt)    # Exponential mapping
    joined = joined @ sm.SE3(exp_twist)         # Update the joined frame
    joined_ax.T = joined                        # Update the visualization of the joined frame

    left_target = joined.A @ linalg.inv(joined_in_left)
    righ_target = joined.A @ linalg.inv(joined_in_right)
    
    # Extract axis and angle from servoing twist
    # homo_axis = np.ones(4)
    # homo_axis[:3] = axis
    
    # axis_in_left = 0.1 * joined_in_left @ homo_axis 
    # # axis_in_left = 0.1 * axis_in_left

    # axis_in_right = 0.1 * joined_in_left @ homo_axis
    # axis_in_right = 

    # print(axis_in_left)

    # ---------------------------------------------------------------------------#
    # SECTION TO PERFORMS TWIST TRANSFORMATION IN A RIGID BODY MOTION  

    # Compute the twist transformation from the middle frame to the left and right frames
    # left_twist = ad_left @ middle_twist 
    # right_twist = ad_right @ middle_twist


    left_tip_pose = pr2.fkine(pr2.q, end=pr2.grippers[1], ).A 
    right_tip_pose = pr2.fkine(pr2.q, end=pr2.grippers[0], ).A

    left_twist, _, _ = angle_axis_python(left_tip_pose, left_target)
    right_twist, _, _ = angle_axis_python(right_tip_pose, righ_target)


    # left_twist = np.zeros(6)
    # left_twist[3:6] = middle_twist[3:6]
    # left_twist[0:3] = middle_twist[0:3] + np.cross( middle_twist[3:6], linalg.inv(joined_in_left)[0:3,3])

    # right_twist = np.zeros(6)
    # right_twist[3:6] = middle_twist[3:6]
    # right_twist[0:3] = middle_twist[0:3] + np.cross( middle_twist[3:6], linalg.inv(joined_in_right)[0:3,3])
    # left_twist = ad_left @ new_twist 
    # right_twist = ad_right @ new_twist

    js = deepcopy(pr2.q)
    jacob_l = pr2.jacobe(js, end=pr2.grippers[1], start="l_shoulder_pan_link")  # Jacobian of the left arm within the end-effector frame
    jacob_r = pr2.jacobe(js, end=pr2.grippers[0], start="r_shoulder_pan_link")  # Jacobian of the right arm within the end-effector frame
    # qdot_left = rmrc(jacob_l, left_twist, p_only = False)
    # qdot_right = rmrc(jacob_r, right_twist,  p_only = False)

    qdot_left = rmrc(jacob_l, 0.05* left_twist, p_only = False)
    qdot_right = rmrc(jacob_r, 0.05* right_twist,  p_only = False)
    qdotc = np.concatenate([qdot_left, qdot_right], axis=0)                     # Composite joint velocities

    j_lb = pr2.jacob0(js, end=pr2.grippers[1], start="l_shoulder_pan_link") 
    j_rb = pr2.jacob0(js, end=pr2.grippers[0], start="r_shoulder_pan_link")   
    jc = np.c_[j_lb, -j_rb]                                                     # Constraint Jacobian matrix in the world frame

    pn = nullspace_projection(jc)   # Nullspace projection matrix of the constraint Jacobian matrix
    qd_pn = pn @ qdotc              # Joint velocities in the nullspace of the constraint Jacobian matrix





    # Update the joint angles
    pr2.q[16:23] = pr2.q[16:23] + qd_pn[7:14] * dt  
    pr2.q[23:30] = pr2.q[23:30] + qd_pn[0:7] * dt

    # Visualization of the frames
    updated_joined_left = left_tip_pose @ joined_in_left
    updated_joined_right = right_tip_pose @ joined_in_right
    left_ax.T = updated_joined_left
    right_ax.T = updated_joined_right

    # l_rot_ax_head.T = left_tip_pose @ sm.SE3(axis_in_left[:3]).A
    # r_rot_ax_head.T = right_tip_pose @ sm.SE3(axis_in_right[:3]).A

    l_rot_ax_head.T = left_target
    r_rot_ax_head.T = righ_target

    # env.remove(l_rot_ax_head)

    # Record the distance between offset frames of each arm to  observe the drift of tracked frame
    dis = np.linalg.norm(updated_joined_left[0:3,3] - updated_joined_right[0:3,3])
    df.append(dis)

    # tool_diff = np.linalg.norm(pr2.fkine(pr2.q, end=pr2.grippers[1],).A[0:3,3] - pr2.fkine(pr2.q, end=pr2.grippers[0],).A[0:3,3])
    tool_diff = np.linalg.norm((linalg.inv(sm.SE3(left_tip_pose)) @ right_tip_pose)[0:3,3])
    dt_f.append(tool_diff)

    env.step(0.05)


# Record and plot the distance between offset frames of each arm to  observe the drift of tracked frame
# plt.figure(1)
# plt.plot(df, 'k', linewidth=1)
# plt.title('Drift graph')
# plt.xlabel('Time')
# plt.ylabel('Distance')

# plt.figure(2)
# plt.plot(dt_f, 'k', linewidth=1)
# plt.title('Tool difference graph')
# plt.xlabel('Time')
# plt.ylabel('Distance')
# print(max(dt_f))


# plt.show()

env.hold()





