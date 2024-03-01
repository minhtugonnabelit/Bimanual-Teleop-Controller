import numpy as np
import roboticstoolbox as rtb
import spatialmath.base as smb
import spatialmath as sm
import matplotlib.pyplot as plt
from scipy import linalg


def add_frame_to_plot(ax, tf, label=''):
    """
    Adds a transformation frame to an existing 3D plot and returns the plotted objects.
    
    Parameters:
    - ax: A matplotlib 3D axis object where the frame will be added.
    - tf: 4x4 transformation matrix representing the frame's position and orientation.
    - label: A string label to identify the frame.
    
    Returns:
    - A list of matplotlib objects (quivers and text).
    """
    # Origin of the frame
    origin = np.array(tf[0:3,3]).reshape(3, 1)

    # Directions of the frame's axes, transformed by R
    x_dir = tf[0:3,0:3] @ np.array([[1], [0], [0]])
    y_dir = tf[0:3,0:3] @ np.array([[0], [1], [0]])
    z_dir = tf[0:3,0:3] @ np.array([[0], [0], [1]])
    
    # Plotting each axis using quiver for direction visualization
    quivers = []
    quivers.append(ax.quiver(*origin, *x_dir, color='r', length=0.15, linewidth=1, normalize=True))
    quivers.append(ax.quiver(*origin, *y_dir, color='g', length=0.15, linewidth=1, normalize=True))
    quivers.append(ax.quiver(*origin, *z_dir, color='b', length=0.15, linewidth=1, normalize=True))
    
    # Optionally adding a label to the frame
    text = None
    if label:
        text = ax.text(*origin.flatten(), label, fontsize=12)
    
    return quivers, text

def animate_frame(tf, quivers, text, ax):

    if quivers:
        for q in quivers:
            q.remove()
    if text:
        text.remove()

    return add_frame_to_plot(ax, tf)

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
    if p_only:
        joint_vel = np.linalg.pinv(jacob) @ np.transpose(twist[0:3])
    else:
        joint_vel = np.linalg.pinv(jacob) @ np.transpose(twist)
    return joint_vel

# Initialize the left and right arms 
left = rtb.models.DH.Planar3()
right = rtb.models.DH.Planar3()
right.base = sm.SE3.Rx(np.pi)
left.q = [np.pi/3, -np.pi/6, -np.pi/2]
right.q = [np.pi/3, -np.pi/6, -np.pi/2]

left_tool = left.fkine(left.q).A
right_tool = right.fkine(right.q).A

# Find the transformation from the left to the right tool
left_to_right = linalg.inv(sm.SE3(left_tool)) * sm.SE3(right_tool)
joined = sm.SE3(left_to_right)

# Extract the middle point between the two tools
joined.A[0:3,3] = (left_tool[0:3,3] + right_tool[0:3,3] ) /2
joined.A[0:3,0:3] = np.eye(3)
joined_in_left = linalg.inv(sm.SE3(left_tool)) * joined
joined_in_right = linalg.inv(sm.SE3(right_tool)) * joined

# Define the targets transformfor bimanual coordination
target1 = joined @ sm.SE3(0.3, 0.3, 0)

# Plot the left and right arms
fig = plt.figure()
fig = left.plot(q = left.q, fig=fig,block=False, name='left')
fig.add(right, name='right')

# Plot the transformation frames
ax = plt.gca()
previous_quivers_left = []
previous_text_left = None

previous_quivers_right = []
previous_text_right = None

previous_quivers_joined = []
previous_text_joined = None

# Visualize the transformation frames
add_frame_to_plot(ax, joined.A)
add_frame_to_plot(ax, target1.A)

# #for pose in traj1:
adjoint_left = adjoint_transform(joined_in_left)
adjoint_right = adjoint_transform(joined_in_right)

arrived = False    
while not arrived:

    # updated_joined_left = left.fkine(left.q).A @ joined_in_left
    # previous_quivers_left, previous_text_left = animate_frame(updated_joined_left, previous_quivers_left, previous_text_left, ax)

    # updated_joined_right = right.fkine(right.q).A @ joined_in_right
    # previous_quivers_right, previous_text_right = animate_frame(updated_joined_right, previous_quivers_right, previous_text_right, ax)

    # middle_twist_left, arrived = rtb.p_servo(updated_joined_left, target1, gain = 0.05)


    # middle_twist, arrived = rtb.p_servo(joined, target1, gain = 0.08, threshold=0.05)


    # joined = joined * sm.SE3(middle_twist[0:3])
    # previous_quivers_joined, previous_text_joined = animate_frame(joined.A, previous_quivers_joined, previous_text_joined, ax)

    updated_joined_left = left.fkine(left.q).A @ joined_in_left
    previous_quivers_left, previous_text_left = animate_frame(updated_joined_left, previous_quivers_left, previous_text_left, ax)
    middle_twist, arrived = rtb.p_servo(updated_joined_left, target1, gain = 0.08, threshold=0.05)

    # updated_joined_right = right.fkine(right.q).A @ joined_in_right
    # previous_quivers_right, previous_text_right = animate_frame(updated_joined_right, previous_quivers_right, previous_text_right, ax)


    jacob_l = left.jacobe(left.q)
    left_twist = adjoint_left @ middle_twist
    qdot_left = rmrc(jacob_l, left_twist, p_only = False)

    # jacob_r = right.jacobe(right.q)[0:3,:]
    # right_twist = adjoint_right @ middle_twists
    # qdot_right = rmrc(jacob_r, right_twist)

    left.q = left.q + qdot_left
    # right.q = right.q + qdot_right 

    fig.step(0.1)


    # plt.pause(0.05)


fig.hold()

