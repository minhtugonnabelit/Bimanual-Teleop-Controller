import numpy as np
import spatialmath.base as smb
import math

def rmrc(jacob, twist, p_only = True):

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
    joint_vel = j_dls @ np.transpose(twist)

    return joint_vel

    # if p_only:
    #     return np.linalg.pinv(jacob[0:3,:]) @ np.transpose(twist[0:3])
    # else:
    #     return np.linalg.pinv(jacob) @ np.transpose(twist)

def nullspace_projection(jacob):

    return np.eye(jacob.shape[1]) - np.linalg.pinv(jacob) @ jacob

def adjoint_transform(T):
    R = T[0:3, 0:3]
    p = T[0:3, 3]
    
    # Create a skew-symmetric matrix from p
    p_skew = np.array([[0, -p[2], p[1]],
                       [p[2], 0, -p[0]],
                       [-p[1], p[0], 0]])
    
    ad = np.zeros((6, 6))
    ad[0:3, 0:3] = R
    ad[3:6, 3:6] = R
    ad[0:3, 3:6] = np.dot(p_skew, R)  # This is the corrected line
    
    return ad


def angle_axis_python(T, Td):
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