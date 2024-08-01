#!/usr/bin/env python3

import rospy
from bimanual_teleop_controller.hand_tracker import RealsenseTracker

def main():
    try:
        rospy.init_node('hand_tracker', log_level=1, anonymous=True)
        data_plot = rospy.get_param('~data_plot', True)
        tracker = RealsenseTracker(data_plot=data_plot)
        tracker._processing_thread.start()
        
    except rospy.ROSInterruptException:
        pass
    

if __name__ == '__main__':
    main()