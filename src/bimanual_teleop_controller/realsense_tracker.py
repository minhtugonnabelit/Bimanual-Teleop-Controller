
import time
import numpy as np
import matplotlib.pyplot as plt
import threading
import cv2

import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray

from bimanual_teleop_controller.hand_tracker import HandTracker
from bimanual_teleop_controller.utility import *

class RealsenseTracker():

    def __init__(self, data_plot = False) -> None:
        
        self._data_plot = data_plot
        if self._data_plot:
            self.elapsed_times = []
            
        self._rate   = rospy.Rate(config['rate'])
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
        self._stamp = None
        self._depth_img = None
        self._result = None
        self._bridge = CvBridge()
        
        self._rgb_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.rgb_callback)
        self._depth_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback)
        
        self._jump_thresh = 0.05
        self._depth_latency = 1.0

        self._kalman = {
            "Left": RealsenseTracker.create_kalman_filter(),
            "Right": RealsenseTracker.create_kalman_filter()
        }
        self._prev_depth = {"Left": None, "Right": None}
        self._last_valid_time = {"Left": None, "Right": None}
              
        self._markers_pub = rospy.Publisher("/hand_markers", MarkerArray, queue_size=10)
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
    
    def rgb_callback(self, msg : Image):
        try: 
            self._img = self._bridge.imgmsg_to_cv2(msg, "bgr8")
            self._result = self._handtracker.get_landmarks_async(self._img, int(msg.header.stamp.to_sec() * 1e+3))
            
            if self._h is None:
                self._h, self._w, _ = self._img.shape
        except CvBridgeError as e:
            rospy.logerr(e)
            
    def depth_callback(self, msg : Image):
        try:
            depth_img = self._bridge.imgmsg_to_cv2(img_msg=msg, desired_encoding='passthrough')
            self._depth_img = depth_img
        except CvBridgeError as e:
            rospy.logerr(e)
    
    def _process_results(self):
        
        while not rospy.is_shutdown():

            if self._result is not None:
                self.get_wrist_point(result = self._result, normalized=False)

            self._rate.sleep()

    def get_wrist_point(self, result, node=0, normalized=True):
        
        nodes_to_average = [0, 1, 5, 9, 13, 17]
        sides = ['Left','Right']
        ges = {side: '' for side in sides}
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
                                x = np.round((u - self._CX)*depth/self._FX, 3)
                                y = np.round((v - self._CY)*depth/self._FY, 3)
                                z = depth

                        print(f'{side} : {[x,y,z]}')
                        ges[side] = gestures[idx][0].category_name
                        marker = RealsenseTracker.create_marker(namespace=side,
                                                                text=ges[side],
                                                                pos = [x,y,z])
                        markers.markers.append(marker)

            if ges['Left'] == 'Pointing_Up' and ges['Right'] == 'Pointing_Up':
                rospy.signal_shutdown('Both hands are closed')
                return None            
            self._markers_pub.publish(markers)

        else:
            rospy.logwarn('No hand detected')
            return None
        
        # annotated_image = HandTracker.draw_landmarks_on_image(self._img, result)
        # cv2.imshow('Hand Tracking', annotated_image)
        # cv2.waitKey(1)

        if self._data_plot:
            elapsed_time = time.time() - start_time
            self.elapsed_times.append(elapsed_time)
            rospy.logdebug(f'Elapsed time: {elapsed_time:.3f}')

        return x, y, z       
    
    def get_gesture(self):
        return self._handtracker.get_gesture()
    
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