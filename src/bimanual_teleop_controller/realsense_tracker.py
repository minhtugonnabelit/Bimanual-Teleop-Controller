
import time
import numpy as np
import matplotlib.pyplot as plt
import threading
import cv2

import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TwistStamped
from visualization_msgs.msg import Marker, MarkerArray

from bimanual_teleop_controller.hand_tracker import HandTracker
from bimanual_teleop_controller.math_utils import LowPassFilter
from bimanual_teleop_controller.utility import *

class RealsenseTracker():

    # TODO: Add publisher for hand node velocity tracking
    def __init__(self, data_plot = False) -> None:
        
        self._data_plot = data_plot
        if self._data_plot:
            self.elapsed_times = []
            
        self._rate   = rospy.Rate(config['CONTROL_RATE'])
        self._handtracker = HandTracker()
        self._processing_thread = threading.Thread(target=self._process_results)
        
        self._cam_inf = rospy.wait_for_message('/camera/color/camera_info', CameraInfo)
        self._FX = self._cam_inf.K[0] 
        self._FY = self._cam_inf.K[4]
        self._CX = self._cam_inf.K[2]
        self._CY = self._cam_inf.K[5] 
        
        self._h = None
        self._w = None
        self._img = None
        self._depth_img = None
        self._result = None
        self._bridge = CvBridge()
        
        self._rgb_sub = rospy.Subscriber("/camera/color/image_raw", Image, self._rgb_callback)
        self._depth_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self._depth_callback)
        
        self._jump_thresh = 0.05
        self._depth_latency = 1.0

        sides = ['Left','Right']
        self._kalman = {side: RealsenseTracker.create_kalman_filter() for side in sides}
        self._prev_depth = {side: None for side in sides}
        self._prev_points = {side: None for side in sides}
        self._last_valid_time = {side: None for side in sides}

        # Initialize LowPassfilters for each velocity component
        self._filters = {
            "Left": {
                "x": LowPassFilter(alpha=0.5),
                "y": LowPassFilter(alpha=0.5),
                "z": LowPassFilter(alpha=0.5)
            },
            "Right": {
                "x": LowPassFilter(alpha=0.5),
                "y": LowPassFilter(alpha=0.5),
                "z": LowPassFilter(alpha=0.5)
            }
        }
              
        self._markers_pub = rospy.Publisher("/hand_markers", MarkerArray, queue_size=10)
        self._hand_twist_pub = {side: rospy.Publisher(f"/{side}_hand_twist", TwistStamped, queue_size=10) for side in sides}
        rospy.logdebug('Initiating camera tracker driver')
        
    def stop(self):

        if self._data_plot:
            elapsed_times = self.elapsed_times
            avg_elapsed_time = sum(elapsed_times) / len(elapsed_times)
            max_elapsed_time = max(elapsed_times)
            plt.plot(elapsed_times)
            plt.xlabel('Frame')
            plt.ylabel('Elapsed Time (s)')
            plt.title('Elapsed Time per Frame')

            # Add text at the top right of the graph
            textstr = f'Avg: {avg_elapsed_time:.3f}s\nMax: {max_elapsed_time:.3f}s'
            
            # Set the location of the text box
            plt.text(0.95, 0.95, textstr, transform=plt.gca().transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.5))

            plt.show()
    
    def _rgb_callback(self, msg : Image):
        try: 
            self._img = self._bridge.imgmsg_to_cv2(msg, "bgr8")
            self._result = self._handtracker.get_landmarks_async(self._img, int(msg.header.stamp.to_sec() * 1e+3))
            
            if self._h is None:
                self._h, self._w, _ = self._img.shape
        except CvBridgeError as e:
            rospy.logerr(e)
            
    def _depth_callback(self, msg : Image):
        try:
            depth_img = self._bridge.imgmsg_to_cv2(img_msg=msg, desired_encoding='passthrough')
            self._depth_img = depth_img
        except CvBridgeError as e:
            rospy.logerr(e)
    
    def _process_results(self):
        
        while not rospy.is_shutdown():

            if self._result is not None:

                current_points = self._get_wrist_point(result = self._result, normalized=False)

                for side in ['Left', 'Right']:
                    if current_points[side] is None:
                        continue
                    elif self._prev_points[side] is None:
                        self._prev_points[side] = current_points[side]
                        continue
                    else:
                        self._get_hand_twist(current_points[side], self._prev_points[side], side, 1/config['CONTROL_RATE'])
                        self._prev_points[side] = current_points[side]


            self._rate.sleep()


    def _get_wrist_point(self, result, node=0, normalized=True):
        
        nodes_to_average = [0, 1, 5, 9, 13, 17]
        sides = ['Left','Right']
        ges = {side: '' for side in sides}
        points = {side: [0,0,0] for side in sides}
        markers = MarkerArray()

        if self._data_plot: 
            start_time = time.time()
        
        if result:
            gestures = result.gestures
            handesness = result.handedness
            hand_landmarks = result.hand_landmarks

            for side in sides:

                u_sum = 0
                v_sum = 0
                valid_node_count = 0 
                x, y, z = 0, 0, 0
                     
                for idx in range(len(hand_landmarks)):
                    if handesness[idx][node].category_name == side:
                                                
                        # Average the x and y coordinates of the selected nodes for middle of the palm
                        for n in nodes_to_average:
                            u_sum += hand_landmarks[idx][n].x
                            v_sum += hand_landmarks[idx][n].y
                            valid_node_count += 1

                        if valid_node_count >0:
                            u_avg = u_sum / valid_node_count
                            v_avg = v_sum / valid_node_count
                            
                            if normalized:
                                x = u_avg - 0.5
                                y = v_avg - 0.5
                                z = 0

                                return x, y, z

                            else:
                                u_avg = int(u_avg * self._w)
                                v_avg = int(v_avg * self._h)
                                depth = np.round(self._depth_img[v_avg,u_avg] * 1e-3, 3)
                                
                                # Handle depth outliers
                                if depth == 0.0 or depth > 10:  
                                    depth = np.nan     
                                    
                                # Handle jump and invalid depth value
                                current_time = time.time()
                                if self._prev_depth[side] is not None:
                                    if np.isnan(depth):
                                        if (current_time - self._last_valid_time[side]) < self._depth_latency:
                                            rospy.logwarn('NaN depth value detected, using previous valid depth')
                                            depth = self._prev_depth[side]
                                        else:
                                            rospy.logwarn('NaN depth value persisted for too long')

                                    else:
                                        if np.abs(depth - self._prev_depth[side]) > self._jump_thresh:
                                            rospy.logwarn('Invalid jump detected in depth data')
                                            depth = self._prev_depth[side]
                                        else:
                                            self._prev_depth[side] = depth
                                            self._last_valid_time[side] = current_time
                                else:
                                    self._prev_depth[side] = depth
                                    self._last_valid_time[side] = current_time
                                
                                # Kalman filter for smoothing
                                measurement = np.array([[np.float32(u_avg)], [np.float32(v_avg)]], np.float32)
                                self._kalman[side].correct(measurement)
                                prediction = self._kalman[side].predict()

                                u = np.clip(int(prediction[0]), 0, self._w - 1)
                                v = np.clip(int(prediction[1]), 0, self._h - 1)
                                
                                # Transform between coordinate
                                points[side][0] = np.round((u - self._CX)*depth/self._FX, 3)
                                points[side][1] = np.round((v - self._CY)*depth/self._FY, 3)
                                points[side][2] = depth

                        ges[side] = gestures[idx][0].category_name
                        marker = ROSUtils.create_marker(namespace=side,
                                                        text=ges[side],
                                                        pos = points[side])
                        markers.markers.append(marker)
           
            self._markers_pub.publish(markers)

        else:
            rospy.logwarn('No hand detected')
            return None

        if self._data_plot:
            elapsed_time = time.time() - start_time
            self.elapsed_times.append(elapsed_time)
            rospy.logdebug(f'Elapsed time: {elapsed_time:.3f}')
            RealsenseTracker._stream_result(self._img, result)
            
        return points   

    def _get_hand_twist(self, point, prev_point, side, dt):

        twist = ROSUtils.create_twiststamped()
        if point is not None and prev_point is not None:
            velocity = np.zeros(6)
            velocity[:3] = (np.asarray(point[:3]) - np.asarray(prev_point[:3])) / dt

            # Apply low-pass filter to each velocity component
            filtered_velocity = [
                self._filters[side]["x"].filter(velocity[0]),
                self._filters[side]["y"].filter(velocity[1]),
                self._filters[side]["z"].filter(velocity[2]),
                0,0,0
            ]
            twist = ROSUtils.create_twiststamped(filtered_velocity)

        self._hand_twist_pub[side].publish(twist)
        return twist
    
    @staticmethod
    def _stream_result(img, result):
        annotated_image = HandTracker.draw_landmarks_on_image(img, result)
        cv2.imshow('Hand Tracking', annotated_image)
        cv2.waitKey(1)

    @staticmethod
    def create_kalman_filter():
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0], 
                                             [0, 1, 0, 0]], np.float32)
        kalman.transitionMatrix = np.array([[1, 0, 1, 0], 
                                            [0, 1, 0, 1], 
                                            [0, 0, 1, 0], 
                                            [0, 0, 0, 1]], np.float32)
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        return kalman
    
    