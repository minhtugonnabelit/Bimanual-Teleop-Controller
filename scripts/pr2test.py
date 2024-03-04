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

def rmrc(jacob, twist, p_only = True):

    if p_only:
        return np.linalg.pinv(jacob[0:3,:]) @ np.transpose(twist[0:3])
    else:
        return np.linalg.pinv(jacob) @ np.transpose(twist)

def nullspace_projection(jacob):

    return np.eye(jacob.shape[1]) - np.linalg.pinv(jacob) @ jacob

def adjoint_transform(T):
    
    R = T[0:3,0:3]
    p = T[0:3,3]
    # p_hat = 
    ad = np.zeros((6,6))
    ad[0:3,0:3] = R
    ad[3:6,3:6] = R
    ad[3:6,0:3] = np.cross(p, R)
    return ad

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
# joined_in_left = linalg.inv(joined) @ left_tip_pose
# ad_left = smb.tr2jac(joined_in_left)
ad_left = adjoint_transform(joined_in_left)

joined_in_right = linalg.inv(sm.SE3(right_tip_pose)) @ joined.A
# joined_in_right = linalg.inv(joined) @ right_tip_pose
# ad_right = smb.tr2jac(joined_in_right)
ad_right = adjoint_transform(joined_in_right)

# Set the target pose
# target = joined.A @ sm.SE3(0.0, -0.2, 0.1).A @ sm.SE3.RPY(0.2,0.2,0.2).A

# Target pose for the drift test case: Only angular motion
# target = joined.A @  sm.SE3.RPY(0.4,0,0).A

# Target pose for the drift test case: Only linear motion
target = joined.A @ sm.SE3(0.1, -0.2, -0.1).A 

target_ax = geometry.Axes(length=0.05, pose = target )

env.add(pr2)
env.add(left_ax )
env.add(right_ax )
env.add(target_ax)
# env.add(joined_ax)


df = list()
dt_f = list()
dt = 0.025
arrived = False    
while not arrived:

    middle_twist, arrived = rtb.p_servo(joined, 
                                        target, 
                                        gain = 0.05, 
                                        threshold=0.01, 
                                        method='angle-axis')  # Servoing in the end-effector frame using angle-axis representation for angular error
    
    
    exp_twist = smb.trexp(middle_twist * dt)    # Exponential mapping
    joined = joined @ sm.SE3(exp_twist)         # Update the joined frame
    joined_ax.T = joined                        # Update the visualization of the joined frame
    
    # Compute the twist transformation from the middle frame to the left and right frames

    # Swap the linear and angular velocities of middle_twist to obtain the twist transformation from the middle frame to the left and right frames
    new_twist = np.zeros(6)
    new_twist[0:3] = middle_twist[3:6]
    new_twist[3:6] = middle_twist[0:3]

    # ad_joined = smb.tr2jac(joined.A)  # Adjoint matrix of the joined frame
    # ad_joined = adjoint_transform(joined.A)
    # print(ad_joined)
    # print("left\n",ad_left)
    # print("right\n",ad_right)
    # left_twist = ad_left @ middle_twist 
    # right_twist = ad_right @ middle_twist


    left_twist = ad_left @ new_twist 
    right_twist = ad_right @ new_twist
    
    # left_twist = middle_twist
    # right_twist = middle_twist


    new_left_twist = np.zeros(6)
    new_left_twist[0:3] = left_twist[3:6]
    new_left_twist[3:6] = left_twist[0:3]
    new_right_twist = np.zeros(6)
    new_right_twist[0:3] = right_twist[3:6]
    new_right_twist[3:6] = right_twist[0:3]


    js = deepcopy(pr2.q)
    jacob_l = pr2.jacobe(js, end=pr2.grippers[1], start="l_shoulder_pan_link")  # Jacobian of the left arm within the end-effector frame
    jacob_r = pr2.jacobe(js, end=pr2.grippers[0], start="r_shoulder_pan_link")  # Jacobian of the right arm within the end-effector frame

    qdot_left = rmrc(jacob_l, left_twist, p_only = False)
    qdot_right = rmrc(jacob_r, right_twist,  p_only = False)

    qdot_left = rmrc(jacob_l, new_left_twist, p_only = False)
    qdot_right = rmrc(jacob_r, new_right_twist,  p_only = False)

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
    updated_joined_left = pr2.fkine(pr2.q, end=pr2.grippers[1],).A @ joined_in_left
    updated_joined_right = pr2.fkine(pr2.q, end=pr2.grippers[0],).A @ joined_in_right
    left_ax.T = updated_joined_left
    right_ax.T = updated_joined_right

    # Record the distance between offset frames of each arm to  observe the drift of tracked frame
    dis = np.linalg.norm(updated_joined_left[0:3,3] - updated_joined_right[0:3,3])
    df.append(dis)

    # tool_diff = np.linalg.norm(pr2.fkine(pr2.q, end=pr2.grippers[1],).A[0:3,3] - pr2.fkine(pr2.q, end=pr2.grippers[0],).A[0:3,3])
    left_tip_pose = pr2.fkine(pr2.q, end=pr2.grippers[1], ).A 
    right_tip_pose = pr2.fkine(pr2.q, end=pr2.grippers[0], ).A
    l2r = linalg.inv(sm.SE3(left_tip_pose)) @ right_tip_pose
    tool_diff = np.linalg.norm(l2r[0:3,3])
    dt_f.append(tool_diff)
    env.step(0.05)


# Record and plot the distance between offset frames of each arm to  observe the drift of tracked frame
plt.figure(1)
plt.plot(df, 'k', linewidth=1)
plt.title('Drift graph')
plt.xlabel('Time')
plt.ylabel('Distance')
# print(max(df))

plt.figure(2)
plt.plot(dt_f, 'k', linewidth=1)
plt.title('Tool difference graph')
plt.xlabel('Time')
plt.ylabel('Distance')
print(max(dt_f))


plt.show()

env.hold()





