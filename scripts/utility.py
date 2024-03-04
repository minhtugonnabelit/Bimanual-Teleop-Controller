import numpy as np
import spatialmath.base as smb

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

    ad = np.zeros((6,6))

    # ad[0:3,0:3] = np.transpose(R)
    # ad[3:6,3:6] = np.transpose(R)
    # ad[0:3,3:6] = - np.cross(p, np.transpose(R))

    ad[0:3,0:3] = R
    ad[3:6,3:6] = R
    ad[0:3,3:6] = np.cross(p,R)
    return ad