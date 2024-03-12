import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from math import pi

rospy.init_node('test_command')
pub = rospy.Publisher('r_arm_controller/command', JointTrajectory, queue_size=1 )

rate = rospy.Rate(10)
msg = JointTrajectory()
msg.header.frame_id = 'torso_lift_link'

while not rospy.is_shutdown():
    msg.header.stamp = rospy.Time.now()
    msg.joint_names = [
        "r_shoulder_pan_joint",
        "r_shoulder_lift_joint",
        "r_upper_arm_roll_joint",
        "r_forearm_roll_joint",
        "r_elbow_flex_joint",
        "r_wrist_flex_joint",
        "r_wrist_roll_joint",
    ]
    point = JointTrajectoryPoint()
    point.positions = [0, 0, 0, 0, 0, 0, pi]
    point.time_from_start = rospy.Duration(1)
    msg.points.append(point)
    pub.publish(msg)
    rate.sleep()