import numpy as np
import cv2
import spatialmath.base as smb
import spatialmath as sm

import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import rospy, rospkg
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker

import types, time

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green
MODEL_PATH = rospkg.RosPack().get_path('bimanual_teleop_controller') + '/config/gesture_recognizer.task'


class HandTracker():

    def __init__(self, run_on_live_stream : False) -> None:

        self._timestamp_ms = 0
        self._result = None
        self._output_image = None
        self._run_on_live_stream = run_on_live_stream

        running_mode = vision.RunningMode.IMAGE
        result_callback = None
        if run_on_live_stream:
            running_mode = vision.RunningMode.LIVE_STREAM
            result_callback = self.result_callback
        
        base_options = python.BaseOptions(
            model_asset_path=MODEL_PATH)
        # options = vision.HandLandmarkerOptions(base_options=base_options,
        #                                        running_mode=running_mode,
        #                                        result_callback=result_callback,
        #                                        num_hands=2)
        # self.detector = vision.HandLandmarker.create_from_options(options)
        
        options = vision.GestureRecognizerOptions(base_options=base_options,
                                               running_mode=running_mode,
                                               result_callback=result_callback,
                                               num_hands=2)
        self.detector = vision.GestureRecognizer.create_from_options(options)
        
        


    def result_callback(self, result: mp.tasks.vision.GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
        self._result = result
        self._output_image = output_image
        self._timestamp_ms = timestamp_ms

    def get_landmarks_async(self, rgb_image, timestamp_ms):
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        self.detector.recognize_async(image, timestamp_ms)
        return self.get_result()
    
    def get_landmarks(self, rgb_image):
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        detection_result = self.detector.detect(image)
        return detection_result

    def get_result(self):
        return self._result
    
    def get_gesture(self):
        return self._result.gestures
    
    def get_timestamps(self):
        return self._timestamp_ms
    
    @staticmethod
    def draw_landmarks_on_image(rgb_image, detection_result):
        hand_landmarks_list = detection_result.hand_landmarks
        hand_world_landmarks_list = detection_result.hand_world_landmarks
        handedness_list = detection_result.handedness
        annotated_image = np.copy(rgb_image)

        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            hand_world_landmarks = hand_world_landmarks_list[idx]
            handedness = handedness_list[idx]

            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])

            # if handedness[0].category_name == 'Right':
            #     print(np.asarray(
            #         [hand_world_landmarks[0].x, hand_world_landmarks[0].y, hand_world_landmarks[0].z]))

            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style())

            # Get the top left corner of the detected hand's bounding box.
            height, width, _ = annotated_image.shape
            # [landmark.x for landmark in hand_landmarks]
            x_coordinates = hand_landmarks[0].x
            # [landmark.y for landmark in hand_landmarks]
            y_coordinates = hand_landmarks[0].y
            # int(min(x_coordinates) * width)
            text_x = int(x_coordinates * width)
            # int(min(y_coordinates) * height) - MARGIN
            text_y = int(y_coordinates * height) - MARGIN

            # Draw handedness (left or right hand) on the image.
            cv2.putText(annotated_image, f"{handedness[0].category_name}",
                        (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                        FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        return annotated_image


class RealsenseTracker():
    
    def __init__(self) -> None:
        
        self._result = None
        self._handtracker = HandTracker(run_on_live_stream=True)
        
        self._cam_inf = rospy.wait_for_message('/rs_camera/color/camera_info', CameraInfo)
        self._FX = self._cam_inf.K[0] 
        self._FY = self._cam_inf.K[4]
        self._CX = self._cam_inf.K[2]
        self._CY = self._cam_inf.K[5]
        
        self._bridge = CvBridge()
        
        self._img = None
        self._rgb_sub = rospy.Subscriber("/rs_camera/color/image_raw", Image, self.rgb_callback)
        
        self._depth_img = None
        self._depth_sub = rospy.Subscriber("/rs_camera/aligned_depth_to_color/image_raw", Image, self.depth_callback)
        
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1,0,0,0], [0,1,0,0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1,0,1,0], [0,1,0,1], [0,0,1,0], [0,0,0,1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2  
              
        self._jump_thresh = 1
        self._prev_depth = None
        self._last_valid_time = None
        self._depth_latency = 1.0
        
        # R = smb.rpy2r(-np.pi/2, np.pi/2, 0, order='xyz')
        # self._cam_transform = np.eye(4)
        # self._cam_transform[:3,:3] = R
              
        self._marker_pub = rospy.Publisher("/visualization_marker", Marker, queue_size = 2)
        
        rospy.logdebug('Initiating camera tracker driver')
    
    def rgb_callback(self, msg : Image):
        
        try:
            
            cv_image = self._bridge.imgmsg_to_cv2(msg, "bgr8")
            self._img = cv_image.copy()
            self._img_stamps = msg.header.stamp.to_sec() * 1e+3
            self._result = self._handtracker.get_landmarks_async(self._img, int(self._img_stamps))
            
        except CvBridgeError as e:
            print(e)
            
    def depth_callback(self, msg : Image):
        self._depth_img = self._bridge.imgmsg_to_cv2(img_msg=msg, desired_encoding='passthrough')
    
    @staticmethod
    def apply_bilateral_filter(depth_image):
        depth_image_32f = depth_image.astype(np.float32)
        return cv2.bilateralFilter(depth_image_32f, d=5, sigmaColor=75, sigmaSpace=75)
    
    @staticmethod    
    def create_marker(pos):
        marker = Marker()

        marker.header.frame_id = "rs_camera_color_optical_frame"
        marker.header.stamp = rospy.Time.now()

        # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
        marker.type = 2
        marker.id = 0

        # Set the scale of the marker
        marker.scale.x = .05
        marker.scale.y = .05
        marker.scale.z = .05

        # Set the color
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # Set the pose of the marker
        marker.pose.position.x = pos[0]
        marker.pose.position.y = pos[1]
        marker.pose.position.z = pos[2]
        marker.pose.orientation.x = 0
        marker.pose.orientation.y = 0
        marker.pose.orientation.z = 0
        marker.pose.orientation.w = 0
        
        return marker
            
    def get_wrist_point(self, side, node=0, normalized=True):
        
        height, width, _ = self._img.shape
        nodes_to_average = [0, 1, 5, 9, 13, 17]
        
        x = 0
        y = 0
        z = 0
        
        # R = smb.rpy2r(-np.pi/2, np.pi/2, 0, order='xyz')
        # transform = np.eye(4)
        # transform[:3,:3] = R
        
        if self._result is not None:
            
            handesness = self._result.handedness
            hand_landmarks = self._result.hand_landmarks

            u_sum = 0
            v_sum = 0
            valid_node_count = 0  
                      
            for idx in range(len(hand_landmarks)):
                if handesness[idx][node].category_name == side:
                    if normalized:
                        x = hand_landmarks[idx][node].x - 0.5
                        y = hand_landmarks[idx][node].y - 0.5
                        z = 0
                    else:
                        for n in nodes_to_average:
                            u = int(hand_landmarks[idx][n].x*width)
                            v = int(hand_landmarks[idx][n].y*height)
                            u_sum += u
                            v_sum += v
                            valid_node_count += 1
                        
                        if valid_node_count >0:
                            u_avg = u_sum // valid_node_count
                            v_avg = v_sum // valid_node_count

                        print(f"raw depth data {self._depth_img[v,u]}")
                        depth_filtered = RealsenseTracker.apply_bilateral_filter(self._depth_img.copy())
                        depth = depth_filtered[v_avg,u_avg] * 1e-3
                        
                        # Handle depth outliers
                        if depth == 0.0 or depth > 10:  
                            depth = np.nan    
                        
                        current_time = time.time()
                        
                        if self._prev_depth is not None:
                            if np.isnan(depth):
                                if (current_time - self._last_valid_time) < self._depth_latency:
                                    rospy.logwarn('NaN depth value detected, using previous valid depth')
                                    depth = self._prev_depth
                                else:
                                    rospy.logwarn('NaN depth value persisted for too long')
                            else:
                                jump_detected = np.abs(depth - self._prev_depth) > self._jump_thresh
                                if jump_detected:
                                    rospy.logwarn('Invalid jump detected in depth data')
                                    depth = self._prev_depth
                                else:
                                    self._prev_depth = depth
                                    self._last_valid_time = current_time
                        else:
                            self._prev_depth = depth
                            self._last_valid_time = current_time
                        
                        # Kalman filter for smoothing
                        measurement = np.array([[np.float32(u_avg)], [np.float32(v_avg)]], np.float32)
                        if not np.isnan(depth):
                            self.kalman.correct(measurement)
                            prediction = self.kalman.predict()

                            u = int(prediction[0])
                            v = int(prediction[1])
                            
                            u = np.clip(u, 0, width - 1)
                            v = np.clip(v, 0, height - 1)
                                                        
                        
                        x = (u - self._CX)*depth/self._FX
                        y = (v - self._CY)*depth/self._FY
                        z = depth
                        print(f'depth after filtering {depth}')

                        # pos = np.asarray([x,y,z,1])
                        # pos_tf = self._cam_transform @ pos
                        
                        self._marker_pub.publish(RealsenseTracker.create_marker([x,y,z]))

        return x, y, z       
    
    def get_gesture(self):
        return self._handtracker.get_gesture()