import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3
import spatialmath as sm
from spatialmath.base import smb
import spatialgeometry as gm
import math
from swift import Swift

target = SE3(0.5, 0.5, 0.5)
target_ax = gm.Axes(length=0.1, pose=target.A)
alpha = math.pi/6
axis_tail = np.array([0, 0, 1])
axis_head = np.array([1, 1, 0])

p1 = gm.Sphere(radius=0.03, color=[1, 0, 0], pose=sm.SE3(axis_tail))
p2 = gm.Sphere(radius=0.03, color=[0, 1, 0], pose=sm.SE3(axis_head))

path = rtb.ctraj(sm.SE3(axis_tail), sm.SE3(axis_head), 15)

a = axis_head - axis_tail
a = a / np.linalg.norm(a)

# R = np.eye(3) + math.sin(alpha) * smb.skew(a) + (1 - math.cos(alpha)) * smb.skew(a) @ smb.skew(a)  # Rodrigues' formula   
R = smb.rodrigues(a, alpha) 
RT = np.eye(4)
RT[:3,:3]= R

T_dot = sm.SE3(axis_tail).A
target_dot = np.linalg.inv(T_dot) @ target.A

target_dot_new = RT @ target_dot

target_new = T_dot @ target_dot_new
target_new_ax = gm.Axes(length=0.1, pose=target_new)

env = Swift()
env.launch()
env.add(target_ax)  
env.add(p1)
env.add(p2)

for p in path[1:-2]:
    env.add(gm.Sphere(radius=0.02, color=[0, 0, 1], pose=p))

env.add(target_new_ax)
env.hold()

