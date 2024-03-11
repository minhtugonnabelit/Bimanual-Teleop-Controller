import numpy as np
import roboticstoolbox as rtb
import matplotlib.pyplot as plt
import spatialmath as sm
from scipy import linalg

from utility import *

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

