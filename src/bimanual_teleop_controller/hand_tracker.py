
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

from bimanual_teleop_controller.utility import *

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green
MODEL_PATH = rospkg.RosPack().get_path('bimanual_teleop_controller') + '/config/gesture_recognizer.task'

class HandTracker():

    def __init__(self):

        self._result = None
        self._timestamp_ms = 0
        base_options = python.BaseOptions(
            model_asset_path=MODEL_PATH)
        
        options = vision.GestureRecognizerOptions(base_options=base_options,
                                               running_mode=vision.RunningMode.LIVE_STREAM,
                                               result_callback=self.result_callback,
                                               num_hands=2)
        self.detector = vision.GestureRecognizer.create_from_options(options)

    def close(self):
        self.detector.close()

    def result_callback(self, result: mp.tasks.vision.GestureRecognizerResult, output_img : mp.Image, timestamp_ms: int):

        self._result = result
        self._timestamp_ms = timestamp_ms

    def get_landmarks_async(self, rgb_image, timestamp_ms):
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        self.detector.recognize_async(image, timestamp_ms)
        return self.get_result()
    
    def get_result(self):
        return self._result
    
    def get_gesture(self):
        return self._result.gestures
    
    def get_timestamps(self):
        return self._timestamp_ms
    
    @staticmethod
    def draw_landmarks_on_image(rgb_image, detection_result):
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        annotated_image = np.copy(rgb_image)

        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])

            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style())

            # Get the top left corner of the detected hand's bounding box.
            height, width, _ = annotated_image.shape
            x_coordinates = hand_landmarks[0].x
            y_coordinates = hand_landmarks[0].y
            text_x = int(x_coordinates * width)
            text_y = int(y_coordinates * height) - MARGIN

            # Draw handedness (left or right hand) on the image.
            cv2.putText(annotated_image, f"{handedness[0].category_name}",
                        (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                        FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        return annotated_image