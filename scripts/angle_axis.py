import numpy as np
from spatialmath import SE3, base
import math

def angle_axis_python(T, Td):
    e = np.empty(6)
    e[:3] = Td[:3, -1] - T[:3, -1]
    R = Td[:3, :3] @ T[:3, :3].T
    li = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])

    if base.iszerovec(li):
        # diagonal matrix case
        if np.trace(R) > 0:
            # (1,1,1) case
            a = np.zeros((3,))
        else:
            a = np.pi / 2 * (np.diag(R) + 1)
    else:
        # non-diagonal matrix case
        ln = base.norm(li)
        a = math.atan2(ln, np.trace(R) - 1) * li / ln

    e[3:] = a

    return e