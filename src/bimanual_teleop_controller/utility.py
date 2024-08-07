# !/usr/bin/env python3

import numpy as np
import yaml
import matplotlib.pyplot as plt

import rospy, rospkg
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import TwistStamped
from visualization_msgs.msg import Marker, MarkerArray

from bimanual_teleop_controller.math_utils import CalcFuncs

from typing import Union
ArrayLike = Union[list, np.ndarray, tuple, set]

MODEL_PATH = rospkg.RosPack().get_path('bimanual_teleop_controller') + '/config/gesture_recognizer.task'
CFG_PATH = rospkg.RosPack().get_path('bimanual_teleop_controller') + '/config/bmcp_cfg.yaml'

def load_config(file_path):
    with open(file_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config

config = load_config(CFG_PATH)

class AnimateFuncs:

    @classmethod
    def add_frame_to_plot(cls, ax, tf, label=''):
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
        origin = np.array(tf[0:3, 3]).reshape(3, 1)

        # Directions of the frame's axes, transformed by R
        x_dir = tf[0:3, 0:3] @ np.array([[1], [0], [0]])
        y_dir = tf[0:3, 0:3] @ np.array([[0], [1], [0]])
        z_dir = tf[0:3, 0:3] @ np.array([[0], [0], [1]])

        # Plotting each axis using quiver for direction visualization
        quivers = []
        quivers.append(ax.quiver(*origin, *x_dir, color='r',
                                 length=0.15, linewidth=1, normalize=True))
        quivers.append(ax.quiver(*origin, *y_dir, color='g',
                                 length=0.15, linewidth=1, normalize=True))
        quivers.append(ax.quiver(*origin, *z_dir, color='b',
                                 length=0.15, linewidth=1, normalize=True))

        # Optionally adding a label to the frame
        text = None
        if label:
            text = ax.text(*origin.flatten(), label, fontsize=12)

        return quivers, text

    @classmethod
    def animate_frame(cls, tf, quivers, text, ax):

        if quivers:
            for q in quivers:
                q.remove()
        if text:
            text.remove()

        return AnimateFuncs.add_frame_to_plot(ax, tf)

class ROSUtils:

    @classmethod
    def call_service(cls, service_name, service_type, **kwargs):
        r"""
        Call a service in ROS
        :param service_name: name of the service
        :param service_type: type of the service
        :param request: request to the service
        :return: response from the service
        """
        rospy.wait_for_service(service_name)
        try:
            service = rospy.ServiceProxy(service_name, service_type)
            response = service(**kwargs)
            return response
        except rospy.ServiceException as e:
            rospy.logerr_once("Service call failed: %s" % e)
            return None
        
    @classmethod
    def create_joint_traj_msg(cls, joint_names: list, dt: float, traj_frame_id : str, joint_states: list = None, qdot: list = None, q: list = None):
        joint_traj = JointTrajectory()
        joint_traj.header.stamp = rospy.Time.now()
        joint_traj.header.frame_id = traj_frame_id
        joint_traj.joint_names = joint_names

        traj_point = JointTrajectoryPoint()
        if q is not None:
            traj_point.positions = q
        else:
            traj_point.positions = CalcFuncs.reorder_values(
                joint_states) + qdot * dt
            traj_point.velocities = qdot

        traj_point.time_from_start = rospy.Duration(dt)
        joint_traj.points = [traj_point]

        return joint_traj

    @staticmethod    
    def create_marker(namespace, text, pos, id=0):
        marker = Marker()

        marker.header.frame_id = "camera_color_optical_frame"
        marker.header.stamp = rospy.Time.now()

        # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
        marker.type = 9 #if text == 'Closed_Fist' else 1
        marker.id = id
        marker.action = Marker.ADD
        marker.ns = namespace
        marker.text = text

        # Set the scale of the marker
        marker.scale.x = .05
        marker.scale.y = .05
        marker.scale.z = .05

        # Set the color
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        if namespace == 'Right':
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0

        # Set the pose of the marker
        marker.pose.position.x = pos[0]
        marker.pose.position.y = pos[1]
        marker.pose.position.z = pos[2]
        
        return marker
    
    @staticmethod
    def create_twiststamped(twist=np.zeros(6)):
        ts = TwistStamped()
        ts.header.frame_id = "camera_color_optical_frame"
        ts.header.stamp = rospy.Time.now()
        ts.twist.linear.x = twist[0]
        ts.twist.linear.y = twist[1]
        ts.twist.linear.z = twist[2]
        ts.twist.angular.x = twist[3]
        ts.twist.angular.y = twist[4]
        ts.twist.angular.z = twist[5]
        return ts


def plot_joint_velocities(actual_data: np.ndarray, desired_data: np.ndarray, dt=0.001, title='Joint Velocities'):

    actual_data = np.array(actual_data)
    desired_data = np.array(desired_data)

    # Adjusted figsize for better visibility
    fig, ax = plt.subplots(2, 4, figsize=(18, 10))

    # Prepare data
    time_space = np.linspace(0, len(actual_data) * dt, len(actual_data))

    # Plot settings
    colors = ['r', 'b']  # Red for actual, Blue for desired
    labels = ['Actual', 'Desired']
    data_types = [actual_data, desired_data]

    for i in range(7):  # Assuming there are 7 joints
        joint_axes = ax[i // 4, i % 4]
        for data, color, label in zip(data_types, colors, labels):
            joint_data = data[:, i]
            if joint_data.shape[0] != len(time_space):
                time_space = np.linspace(
                    0, len(joint_data) * dt, len(joint_data))
            joint_axes.plot(time_space, joint_data, color,
                            linewidth=1, label=label if i == 0 else "")

            # Max and Min annotations
            max_value = np.max(joint_data)
            min_value = np.min(joint_data)
            max_time = time_space[np.argmax(joint_data)]
            min_time = time_space[np.argmin(joint_data)]

            joint_axes.annotate(f'Max: {max_value:.2f}', xy=(max_time, max_value), xytext=(10, 0),
                                textcoords='offset points', ha='center', va='bottom', color=color)
            joint_axes.annotate(f'Min: {min_value:.2f}', xy=(min_time, min_value), xytext=(10, -10),
                                textcoords='offset points', ha='center', va='top', color=color)

        joint_axes.set_title(config['JOINT_NAMES'][title][i])

    fig.legend(['Actual', 'Desired'], loc='upper right')
    fig.suptitle(title)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig, ax

def plot_manip_and_drift(constraint_distance: float, 
                         manipulabity_threshold: float, 
                         joint_limits : np.ndarray, 
                         joint_positions : np.ndarray ,
                         joint_velocities: np.ndarray, 
                         joint_efforts: np.ndarray,
                         drift: np.ndarray, 
                         manip: np.ndarray, 
                         dt=0.001):
    r"""
    Plot the manipulability and drift data for the PR2 robot.

    Parameters:
    - constraint_distance: The constraint distance data.
    - drift: The drift data.
    - manip_l: The manipulability data for the left arm.
    - manip_r: The manipulability data for the right arm.
    - dt: The time interval.

    Returns:
    - The figure and axes objects.
    """
    # plt.rcParams['text.usetex'] = True
    # plt.rcParams.update({'font.size': 10})  # Adjust font size smaller if necessary
    # plt.rcParams['figure.facecolor'] = 'white'


    # Prepare data
    fig, ax = plt.subplots(3, 2, figsize=(15, 20))
    time_space = np.linspace(0, len(drift) * dt, len(drift))
    manip_axes = ax[0, 0]
    drift_axes = ax[0, 1]
    manip_l = manip[0]
    manip_r = manip[1]
    joint_eff_axes = ax[1, :]
    joint_vel_axes = ax[2, :]

    plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.15, hspace=0.5, wspace=0.2)  # Adjust horizontal spacing
    joint_limits = {
        'left': [joint_limits[0][:7], joint_limits[1][:7]],
        'right': [joint_limits[0][7:], joint_limits[1][7:]]     
    }

    # Plot joint positions

    if len(joint_positions[0]) != len(time_space):
        time_space = np.linspace(0, len(joint_positions['l'])
                                 * dt, len(joint_positions['l']) + 1)
        
    # joint_accelerations = {
    #     'left': np.diff(joint_velocities['left'], axis=0) / dt,
    #     'right': np.diff(joint_velocities['right'], axis=0) / dt
    # }

    # for i, side in enumerate(['left', 'right']):
    #     # for j in range(7):  # Assuming there are 7 joints
    #     #     joint_data = np.array([d[j] for d in joint_positions[i]])
    #     #     joint_pos_axes[i].plot(time_space, joint_data, label=f'Joint {j+1}')
    #     #     joint_pos_axes[i].axhline(y=joint_limits[side][0][j], color='r', linestyle='--')  # Lower limit
    #     #     joint_pos_axes[i].axhline(y=joint_limits[side][1][j], color='g', linestyle='--')  # Upper limit

    #     joint_data = np.array([d[5] for d in joint_positions[i]])
    #     joint_pos_axes[i].plot(time_space, joint_data, label=f'Joint {5+1}')
    #     joint_pos_axes[i].axhline(y=joint_limits[side][0][5], color='r', linestyle='--')  # Lower limit
    #     joint_pos_axes[i].axhline(y=joint_limits[side][1][5], color='g', linestyle='--')  # Upper limit
    #     joint_pos_axes[i].set_xlabel('Time [s]')
    #     joint_pos_axes[i].set_ylabel('Joint Position [rad]')
    #     joint_pos_axes[i].set_title(f'{side.capitalize()} Arm Joint Positions')
    #     joint_pos_axes[i].legend()

    for i, side in enumerate(['left', 'right']):
        for j in range(7): 
            joint_data = np.array([d[j] for d in joint_efforts[side]])
            joint_eff_axes[i].plot(time_space, joint_data, label=f'Joint {j+1}')  # Use time_space[:-1] because np.diff reduces the length by 1
        joint_eff_axes[i].set_title(f'{side.capitalize()} Arm Joint efforts')
        joint_eff_axes[i].set_xlabel('Time [s]')
        joint_eff_axes[i].set_ylabel(r'Joint efforts [Nm]')
        joint_eff_axes[i].legend()

    # Plot joint velocities
    for i, side in enumerate(['left', 'right']):
        for j in range(7):  # Assuming there are 7 joints
            joint_data = np.array([d[j] for d in joint_velocities[side]])
            joint_vel_axes[i].plot(time_space, joint_data, label=f'Joint {j+1}')
        joint_vel_axes[i].set_title(f'{side.capitalize()} Arm Joint Velocities')
        joint_vel_axes[i].set_xlabel('Time [s]')
        joint_vel_axes[i].set_ylabel('Joint Velocity [rad/s]')
        joint_vel_axes[i].legend()

    # Plot manipulability data   
    if len(manip_r) != len(time_space):
        time_space = np.linspace(0, len(manip_r)
                                 * dt, len(manip_r) + 1)

    if len(manip_l) != len(time_space):
        time_space = np.linspace(0, len(manip_l)
                                 * dt, len(manip_l) + 1)

    manip_axes.plot(time_space, manip_l, 'r', linewidth=1, label='Left' )
    manip_axes.plot(time_space, manip_r, 'b', linewidth=1, label='Right')
    manip_axes.set_title('Manipulability')
    manip_axes.set_xlabel('Time [s]')
    manip_axes.set_ylabel(r'$\omega$')
    manip_axes.axhline(y=manipulabity_threshold, color='k',
                       linewidth=1, linestyle='--', label='Threshold')

    manip_axes.annotate(f'Min Left {np.min(manip_l):.2f}',
                        xy=(time_space[np.argmin(manip_l)], np.min(manip_l)),
                        xytext=(10, 0), textcoords='offset points',
                        ha='center', va='bottom', color='r')

    manip_axes.annotate(f'Min Right {np.min(manip_r):.2f}',
                        xy=(time_space[np.argmin(manip_r)], np.min(manip_r)),
                        xytext=(10, -10), textcoords='offset points',
                        ha='center', va='top', color='b')
    
    manip_axes.legend()


    # Plot drift data
    if len(drift) != len(time_space):
        time_space = np.linspace(0, len(drift)
                                 * dt, len(drift) + 1)

    drift_axes.plot(time_space, drift, 'k', linewidth=1)
    drift_axes.set_title('Drift graph')
    drift_axes.set_xlabel('Time [s]')
    drift_axes.set_ylabel('Distance [m]')
    drift_axes.set_ylim([constraint_distance - 0.05, constraint_distance + 0.05]) 
    drift_axes.axhline(y=constraint_distance, color='r', linewidth=1, label=f'Constraint = {constraint_distance:.4f}')
    drift_axes.axhline(y=np.max(drift), color='k', linewidth=1, linestyle='--', label=f'Max drift = {np.max(drift):.4f}')
    drift_axes.axhline(y=np.min(drift), color='k', linewidth=1, linestyle='--', label=f'Min drift = {np.min(drift):.4f}')
    drift_axes.legend()

    return fig, ax

def graph_plotter(func):

    # Use matplotlib's built-in math text renderer
    plt.rcParams['text.usetex'] = True
    plt.rcParams.update({'font.size': 10})
    
    def f(x, k):
        r"""
        Weighting function for smooth transition between two states.

        Parameters:
        - x: The input value.
        - k: The gain for steepness of the transition.

        Returns:
        - The weighted value.
        """

        return 1 / (1 + np.exp(-k * (x - 0.5)))
    
    # Generate x values
    x = np.linspace(-2/3, 1+2/3, 100)
    
    # Generate y values
    y = func(x, 10)
    k = [5, 7, 10]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8,4))
    if isinstance(k, (int, float)):
        ax.plot(x, f(x,k), 'r',label=f'$k = {k}$')
    else:
        for i in range(len(k)):
            ax.plot(x, f(x,k[i]), label=f'$k = {k[i]}$')

    plt.axvline(x=0, color='r', linestyle='--', label=r'$q_i$ at soft limit start')
    plt.axvline(x=1, color='b', linestyle='--', label=r'$q_i$ at soft limit end')
    plt.axvline(x=1+2/3, color='k', linestyle='--', label=r'$q_i$ at hard limit') 
    plt.ylim(-0, 1)

    plt.xlabel(r'Distance to limit $x$')
    ax.set_ylabel(r'Weight $\lambda$', rotation='vertical', loc='center', labelpad=1)
    ax.legend()

    # Show the plot
    plt.show()
