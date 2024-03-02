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
    # p_hat = sm.base.skew(p)
    A = np.zeros((6, 6))
    A[0:3, 0:3] = R
    A[0:3, 3:6] = np.cross(p, R)
    A[3:6, 3:6] = R
    return A

def rmrc(jacob, twist, p_only = True):

    """
    Resolved motion rate control for a robot joint to reach a target configuration."""

    if p_only:
        return np.linalg.pinv(jacob[0:3,:]) @ np.transpose(twist[0:3])
    else:
        return np.linalg.pinv(jacob) @ np.transpose(twist)

    # return np.linalg.pinv(jacob[0:3,:]) @ np.transpose(twist[0:3]) if p_only else np.linalg.pinv(jacob) @ np.transpose(twist)

def nullspace_projection(jacob):
    """
    Computes the nullspace projection matrix of a given Jacobian matrix.
    
    Parameters:
    - jacob: A 6xN Jacobian matrix.

    Returns:
    - A 6xN nullspace projection matrix.

    """
    return np.eye(jacob.shape[1]) - np.linalg.pinv(jacob) @ jacob


pr2 = rtb.models.PR2()
qtest = np.zeros(31)
qtest[16:23] = [-pi/6,pi/6,-pi/3,-pi/2,0,-pi/4,pi/2]
qtest[23:30] = [pi/6,pi/6,pi/3,-pi/2,0,-pi/4,pi/2]
pr2.q = qtest
print(pr2)

linalg

env = Swift()
env.set_camera_pose([0.5, 0, 1.5], sm.SE3(0.0,0,0).A[0:3,3])
env.launch()

left_tip_pose = pr2.fkine(qtest, end=pr2.grippers[1], ).A 
right_tip_pose = pr2.fkine(qtest, end=pr2.grippers[0], ).A
left_ax = geometry.Axes(length=0.05, pose = left_tip_pose ) 
right_ax = geometry.Axes(length=0.05, pose = right_tip_pose )
l2r = linalg.inv(sm.SE3(left_tip_pose)) @ right_tip_pose

# Extract the middle point between the two tools
joined = sm.SE3()
joined.A[0:3,3] = (left_tip_pose[0:3,3] + right_tip_pose[0:3,3] ) /2
joined.A[0:3,0:3] = np.eye(3)
joined_in_left = linalg.inv(sm.SE3(left_tip_pose)) @ joined.A
ad_left = adjoint_transform(joined_in_left)

joined_in_right = linalg.inv(sm.SE3(right_tip_pose)) @ joined.A
ad_right = adjoint_transform(joined_in_right)

# Set the target pose
# target = joined.A @ sm.SE3(0.1, -0.2, -0.1).A @ sm.SE3.RPY(0.1,0.2,0.1).A
target = joined.A @ sm.SE3(0.1, -0.2, -0.1).A 

target_ax = geometry.Axes(length=0.05, pose = target )


env.add(pr2)
env.add(left_ax )
env.add(right_ax )
env.add(target_ax)

plt.figure()
df = list()


arrived = False    
while not arrived:

    # Visualization of the frames
    updated_joined_left = pr2.fkine(pr2.q, end=pr2.grippers[1], ).A @ joined_in_left
    updated_joined_right = pr2.fkine(pr2.q, end=pr2.grippers[0], ).A @ joined_in_right
    left_ax.T = updated_joined_left
    right_ax.T = updated_joined_right

    # Perform the servoing in the end-effector frame
    middle_twist, arrived = rtb.p_servo(updated_joined_left, 
                                        target, 
                                        gain = 0.03, 
                                        threshold=0.01, 
                                        method='angle-axis')
    
    
    # Compute the twist transformation from the middle frame to the left and right frames
    left_twist = ad_left @ middle_twist
    right_twist = ad_right @ middle_twist

    # Jacobian taken relative to the end-effector frame as the servo is performed in the end-effector frame
    jacob_l = pr2.jacobe(pr2.q, end=pr2.grippers[1], start="l_shoulder_pan_link" )
    jacob_r = pr2.jacobe(pr2.q, end=pr2.grippers[0], start="r_shoulder_pan_link")

    # Compute the joint velocities
    qdot_left = rmrc(jacob_l, left_twist, p_only = False)
    qdot_right = rmrc(jacob_r, right_twist,  p_only = False)

    # Construct constraint Jacobian matrix from world frame Jacobians of each arm
    qdotc = np.concatenate([qdot_left, qdot_right], axis=0)
    j_lb = pr2.jacob0(pr2.q, end=pr2.grippers[1], start="l_shoulder_pan_link" )
    j_rb = pr2.jacob0(pr2.q, end=pr2.grippers[0], start="r_shoulder_pan_link")
    jc = np.c_[j_lb, -j_rb]    

    # Compute the nullspace projection of the constraint Jacobian when DLS is used
    pn = nullspace_projection(jc)
    qd_pn = pn @ qdotc
    
    # # Update the joint angles
    # pr2.q[16:23] = pr2.q[16:23] + qdot_right * 0.05
    # pr2.q[23:30] = pr2.q[23:30] + qdot_left * 0.05

    # Update the joint angles
    pr2.q[16:23] = pr2.q[16:23] + qd_pn[7:14] * 0.05
    pr2.q[23:30] = pr2.q[23:30] + qd_pn[0:7] * 0.05

    # Record the distance between offset frames of each arm to  observe the drift of tracked frame
    dis = np.linalg.norm(updated_joined_left[0:3,3] - updated_joined_right[0:3,3])
    df.append(dis)

    env.step(0.05)


# Record and plot the distance between offset frames of each arm to  observe the drift of tracked frame
plt.plot(df, 'k', linewidth=1)
plt.title('Drift graph')
plt.xlabel('Time')
plt.ylabel('Distance')
plt.show()



env.hold()





