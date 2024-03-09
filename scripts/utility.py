import numpy as np
import spatialmath.base as smb
import math

def rmrc(jacob, twist, p_only = True):

    if p_only:
        return np.linalg.pinv(jacob[0:3,:]) @ np.transpose(twist[0:3])
    else:
        return np.linalg.pinv(jacob) @ np.transpose(twist)

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

    