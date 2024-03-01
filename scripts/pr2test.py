import numpy as np
import roboticstoolbox as rtb
import spatialgeometry as geometry
import spatialmath.base as smb
import spatialmath as sm
import matplotlib.pyplot as plt
from scipy import linalg
from swift import Swift     

from math import pi

def adjoint_transform(T):
    """
    Computes the adjoint transformation matrix of a given homogeneous transformation matrix for transforming twist to twist on a rigid body.
    
    Parameters:
    - T: A 4x4 homogeneous transformation matrix.

    Returns:
    - A 6x6 adjoint transformation matrix.

    """
    R = T[0:3, 0:3]
    p = T[0:3, 3]
    p_hat = sm.base.skew(p)
    A = np.zeros((6, 6))
    A[0:3, 0:3] = R
    A[3:6, 0:3] = p_hat @ R
    A[3:6, 3:6] = R
    return A

def rmrc(jacob, twist, p_only = True):

    """
    Resolved motion rate control for a robot joint to reach a target configuration."""

    # if p_only:
    #     return np.linalg.pinv(jacob[0:3,:]) @ np.transpose(twist[0:3])
    # else:
    #     return np.linalg.pinv(jacob) @ np.transpose(twist)

    return np.linalg.pinv(jacob[0:3,:]) @ np.transpose(twist[0:3]) if p_only else np.linalg.pinv(jacob) @ np.transpose(twist)

pr2 = rtb.models.PR2()
print(pr2)

qtest = np.zeros(31)
qtest[16:23] = [-pi/6,pi/6,-pi/3,-pi/2,0,-pi/4,pi/2]
qtest[23:30] = [pi/6,pi/6,pi/3,-pi/2,0,-pi/4,pi/2]
pr2.q = qtest

env = Swift()
env.launch()
env.add(pr2)



left_tip_pose = pr2.fkine(qtest, end=pr2.grippers[1]).A 
right_tip_pose = pr2.fkine(qtest, end=pr2.grippers[0]).A
left_ax = geometry.Axes(length=0.05, pose = left_tip_pose ) 
right_ax = geometry.Axes(length=0.05, pose = right_tip_pose )

env.add(left_ax )
env.add(right_ax )

l2r = linalg.inv(sm.SE3(left_tip_pose)) @ right_tip_pose
joined = sm.SE3()

# Extract the middle point between the two tools
joined.A[0:3,3] = (left_tip_pose[0:3,3] + right_tip_pose[0:3,3] ) /2
joined.A[0:3,0:3] = np.eye(3)
joined_in_left = linalg.inv(sm.SE3(left_tip_pose)) @ joined.A
joined_in_right = linalg.inv(sm.SE3(right_tip_pose)) @ joined.A

target = joined.A @ sm.SE3(0.2, 0, 0.1).A

ad_left = adjoint_transform(joined_in_left)
ad_right = adjoint_transform(joined_in_right)

offset_ax = geometry.Axes(length=0.05, pose = target )
env.add(offset_ax)

arrived = False    
while not arrived:


    updated_joined_left = pr2.fkine(qtest, end=pr2.grippers[1]).A @ joined_in_left
    left_ax.T = updated_joined_left

    updated_joined_right = pr2.fkine(qtest, end=pr2.grippers[0]).A @ joined_in_right
    right_ax.T = updated_joined_right

    middle_twist, arrived = rtb.p_servo(updated_joined_left, target, gain = 0.05, threshold=0.05)


    jacob_l = pr2.jacobe(pr2.q, end=pr2.grippers[1], start="l_shoulder_pan_link" )
    left_twist = ad_left @ middle_twist
    qdot_left = rmrc(jacob_l, left_twist, p_only = False)

    jacob_r = pr2.jacobe(pr2.q, end=pr2.grippers[0], start="r_shoulder_pan_link")
    right_twist = ad_right @ middle_twist
    qdot_right = rmrc(jacob_r, right_twist,  p_only = False)

    pr2.q[23:30] = pr2.q[23:30] + qdot_left * 0.005
    pr2.q[16:23] = pr2.q[16:23] + qdot_right * 0.005



    env.step(0.05)







