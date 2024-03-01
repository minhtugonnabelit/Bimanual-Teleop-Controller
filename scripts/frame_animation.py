import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def init_frame(ax):
    """
    Initializes the frame in the plot. This function creates quiver objects
    for each axis of the frame and returns them for updating in the animation.
    """
    quivers = {
        'x': ax.quiver(0, 0, 0, 1, 0, 0, color='r', length=0.1),
        'y': ax.quiver(0, 0, 0, 0, 1, 0, color='g', length=0.1),
        'z': ax.quiver(0, 0, 0, 0, 0, 1, color='b', length=0.1)
    }
    return quivers

def update_frame(num, quivers, ax):
    """
    Updates the position and orientation of the frame for each frame of the animation.
    """
    # Angle of rotation
    angle = np.radians(num)
    
    # Circular path for translation
    t = [np.cos(angle), np.sin(angle), 0]
    
    # Rotation matrix for simple rotation around the z-axis
    R = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0,              0,             1]
    ])
    
    # Update quiver objects
    for key, q in quivers.items():
        # Compute new direction vectors based on the rotation matrix
        direction = R @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])[:, [0, 1, 2][key == 'x':][:, None]]
        
        # Update the quiver objects
        q.set_segments([np.array([[t, t + 0.1 * direction.flatten()]]).reshape(1, 2, 3)])
    
    # Optional: Update plot limits if necessary
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-0.5, 0.5])
    
    return quivers

# Create a matplotlib figure and 3D axis using pyplot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Initialize the frame in the plot
quivers = init_frame(ax)

# Create an animation
ani = FuncAnimation(fig, update_frame, frames=np.arange(0, 360, 2), fargs=(quivers, ax), blit=False)

plt.show()
