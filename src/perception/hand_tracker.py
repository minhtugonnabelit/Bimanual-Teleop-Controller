import numpy as np
import cv2

import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

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
            model_asset_path='/home/anhminh/git/hand_tracking/hand_landmarker.task')
        options = vision.HandLandmarkerOptions(base_options=base_options,
                                               running_mode=running_mode,
                                               result_callback=result_callback,
                                               num_hands=2)
        
        self.detector = vision.HandLandmarker.create_from_options(options)

    def result_callback(self, result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self._result = result
        self._output_image = output_image
        self._timestamp_ms = timestamp_ms

    def get_landmarks_async(self, rgb_image, timestamp_ms):
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        # timestamp_ms = self.get_timestamps()

        self.detector.detect_async(image, timestamp_ms)
        return self.get_result()

    def get_result(self):
        return self._result
    
    def get_timestamps(self):
        return self._timestamp_ms
    
    def get_landmarks(self, rgb_image):
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        detection_result = self.detector.detect(image)
        return detection_result
    
    def get_hand_annotation(self, rgb_image):
        # STEP 3: Load the input image.
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # STEP 4: Detect hand landmarks from the input image.
        detection_result = self.detector.detect(image)

        # STEP 5: Process the classification result. In this case, visualize it.
        annotated_image = HandTracker.draw_landmarks_on_image(
            image.numpy_view(), detection_result)
        
        return annotated_image
    
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
            if handedness[0].category_name == 'Right':
                print(np.asarray(
                    [hand_world_landmarks[0].x, hand_world_landmarks[0].y, hand_world_landmarks[0].z]))

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
