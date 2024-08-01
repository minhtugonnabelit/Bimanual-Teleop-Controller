import rospy
from visualization_msgs.msg import Marker, MarkerArray


def markers_callback(data : MarkerArray):
    print(f"{data.markers[0].ns} [{data.markers[0].pose.position.x},{data.markers[0].pose.position.y},{data.markers[0].pose.position.z}] | {data.markers[1].ns} [{data.markers[1].pose.position.x},{data.markers[1].pose.position.y},{data.markers[1].pose.position.z}]")

def main():
    try:
        rospy.init_node('hand_node_subscriber', log_level=1, anonymous=True)
        rospy.Subscriber('/hand_markers', MarkerArray, markers_callback)
        rospy.spin()
        
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()