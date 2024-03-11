import numpy as np
import spatialmath.base as smb
import math

def rmrc(jacob, twist, p_only = True):

    r"""
    Calculate the joint velocities using the Resolved Motion Rate Control (RMRC) method.
    
    Parameters:
    - jacob: The Jacobian matrix of the robot.
    - twist: The desired twist of the robot.
    - p_only: A boolean to indicate if the RMRC method should be used for position only.

    Returns:
    - The joint velocities of the robot.

    """

    # calculate manipulability
    w = np.sqrt(np.linalg.det(jacob @ np.transpose(jacob)))

    # set threshold and damping
    w_thresh = 0.1
    max_damp = 0.5


    # if manipulability is less than threshold, add damping
    damp = (1 - np.power(w/w_thresh, 2)) * max_damp if w < w_thresh else 0


    # calculate damped least square
    j_dls = np.transpose(jacob) @ np.linalg.inv(jacob @ np.transpose(jacob) + damp * np.eye(6))


    # get joint velocities, if robot is in singularity, use damped least square
    qdot = j_dls @ np.transpose(twist)

    return qdot


def nullspace_projection(jacob):

    r"""
    Calculate the projection matrix on to the null space of the Jacobian matrix.

    Parameters:
    - jacob: The Jacobian matrix of the robot.

    Returns:
    - The projection matrix on to the null space of the Jacobian matrix.
    """
    
    return np.eye(jacob.shape[1]) - np.linalg.pinv(jacob) @ jacob


def angle_axis_python(T, Td):

    r"""
    Computes the angle-axis representation of the error between two transformation matrices.

    Parameters:
    - T: The current transformation matrix.
    - Td: The desired transformation matrix.
    Returns:
    - e: A 6x1 vector representing the error in the angle-axis representation.
    - angle: The angle of rotation between the two frames.
    - axis: The axis of rotation between the two frames.

    """
    e = np.empty(6)
    e[:3] = Td[:3, -1] - T[:3, -1]
    R = Td[:3, :3] @ T[:3, :3].T
    li = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    ln = smb.norm(li)

    if smb.iszerovec(li):
        # diagonal matrix case
        if np.trace(R) > 0:
            # (1,1,1) case
            a = np.zeros((3,))
        else:
            a = np.pi / 2 * (np.diag(R) + 1)
    else:
        # non-diagonal matrix case
        a = math.atan2(ln, np.trace(R) - 1) * li / ln
        
    axis = li / ln
    angle = math.atan2(ln, np.trace(R) - 1)

    e[3:] = a

    return e, angle, axis

# Function for visualiztion of the frames in the 3D plot using Matplotlib
def add_frame_to_plot(ax, tf, label=''):

    r"""
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