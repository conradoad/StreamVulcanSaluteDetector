import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import cv2 as cv
import pyvirtualcam

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from datetime import datetime
import time


RUN_MODE = True         # False - print hand landmarks only | 1-execute final functions
OUTPUT_MODE = 0         # 0-imshow | 1-pyvirtualcam
SEGMENTER_MODEL = 1     # 0-selfie_segmenter | 1-selfie_segmenter_landscape | 2-multiclass_selfie_segmenter
DEBUG = False

DEFAULT_BG_IMAGE_BLUR = 0   # 0-background image | 1-background blur

ENTERPRISE_SCENE = False
ENTERPRISE_SCENE_LAST_CHANGED = None

CAMERA_IDX = 1

FRAME_WIDTH = 1080
FRAME_HEIGHT = 720

MIN_TIME_TO_CHANGE_SEC = 5

THUMB_IDX = [2, 3, 4]
INDEX_FINGER_IDX = [5, 6, 7, 8]
MIDDLE_FINGER_IDX = [9, 10, 11, 12]
RING_FINGER_IDX = [13, 14, 15, 16]
PINKY_TIP_IDX = [17, 18, 19, 20]

MAX_INDEX_MIDDLE_DEG = 4
MAX_RING_PINKY_DEG = 4
MIN_MIDDLE_RING_DEG = 8
MIN_THUMB_INDEX_DEG = 30

MARGIN = 15  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 2
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

hand_landmark_model_path = './models/hand_landmarker.task'
selfie_segmenter_model_path = './models/selfie_segmenter.tflite'
selfie_segmenter_landscape_model_path = './models/selfie_segmenter_landscape.tflite'
multiclass_selfie_segmenter_model_path = './models/selfie_multiclass_256x256.tflite'
bg_image_path_vulcan = './background_images/background2.jpg'
bg_image_path_default = './background_images/Oficinas.jpg'

# hand_landmark_model_path = 'C:\\Users\\423458\\StreamVulcanSaluteDetector\\models\\hand_landmarker.task'
# selfie_segmenter_model_path = 'C:\\Users\\423458\\StreamVulcanSaluteDetector\\models\\selfie_segmenter.tflite'
# selfie_segmenter_landscape_model_path = 'C:\\Users\\423458\\StreamVulcanSaluteDetector\\models\\selfie_segmenter_landscape.tflite'
# multiclass_selfie_segmenter_model_path = 'C:\\Users\\423458\\StreamVulcanSaluteDetector\\models\\selfie_multiclass_256x256.tflite'
# bg_image_path_vulcan = 'C:\\Users\\423458\\StreamVulcanSaluteDetector\\background_images\\background2.jpg'
# bg_image_path_default = 'C:\\Users\\423458\\StreamVulcanSaluteDetector\\background_images\\Oficinas.jpg'


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarksConnections = mp.tasks.vision.HandLandmarksConnections

ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions

def get_angle_between_fingers(result, finger1, finger2):
    finger1_start = finger1[0]
    finger1_end = finger1[len(finger1)-1]

    finger1_start_x = result.hand_landmarks[0][finger1_start].x
    finger1_start_y = 1- result.hand_landmarks[0][finger1_start].y
    finger1_end_x = result.hand_landmarks[0][finger1_end].x
    finger1_end_y = 1- result.hand_landmarks[0][finger1_end].y

    finger1_slope = (finger1_end_y-finger1_start_y)/(finger1_end_x-finger1_start_x)

    finger2_start = finger2[0]
    finger2_end = finger2[len(finger1)-1]
    finger2_start_x = result.hand_landmarks[0][finger2_start].x
    finger2_start_y = 1 - result.hand_landmarks[0][finger2_start].y
    finger2_end_x = result.hand_landmarks[0][finger2_end].x
    finger2_end_y = 1 - result.hand_landmarks[0][finger2_end].y

    finger2_slope = (finger2_end_y-finger2_start_y)/(finger2_end_x-finger2_start_x)

    tan = abs((finger2_slope - finger1_slope) / (1 + finger2_slope * finger1_slope))
    angle = np.arctan(tan) * 180 / np.pi
    return angle

def is_vulcan_greeting(result):
    thumb_index_angle = get_angle_between_fingers(result, THUMB_IDX, INDEX_FINGER_IDX)
    index_middle_angle = get_angle_between_fingers(result, INDEX_FINGER_IDX, MIDDLE_FINGER_IDX)
    middle_ring_angle = get_angle_between_fingers(result, MIDDLE_FINGER_IDX, RING_FINGER_IDX)
    ring_pink_angle = get_angle_between_fingers(result, RING_FINGER_IDX, PINKY_TIP_IDX)

    if DEBUG:
        print('thumb_index: {}'.format(thumb_index_angle))
        print('index_middle: {}'.format(index_middle_angle))
        print('middle_ring: {}'.format(middle_ring_angle))
        print('ring_pink: {}'.format(ring_pink_angle))
        print("")

    if thumb_index_angle >= MIN_THUMB_INDEX_DEG and index_middle_angle <= MAX_INDEX_MIDDLE_DEG and middle_ring_angle >= MIN_MIDDLE_RING_DEG and ring_pink_angle <= MAX_RING_PINKY_DEG:
        return True
    return False

def draw_landmarks_on_image(rgb_image, detection_result, is_vulcan_greeting):
  hand_landmarks_list = detection_result.hand_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]

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

    if is_vulcan_greeting == True:
        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width) - 100
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv.putText(annotated_image, f"VULCAN GREETING!!!",
                    (text_x, text_y), cv.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv.LINE_AA)

    return annotated_image

def hand_detected_handle(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    if len(result.handedness) > 0 and  is_vulcan_greeting(result):
        global ENTERPRISE_SCENE_LAST_CHANGED
        global ENTERPRISE_SCENE
        if ENTERPRISE_SCENE_LAST_CHANGED == None or (time.time() - ENTERPRISE_SCENE_LAST_CHANGED) > MIN_TIME_TO_CHANGE_SEC:
            ENTERPRISE_SCENE_LAST_CHANGED = time.time()
            ENTERPRISE_SCENE = not ENTERPRISE_SCENE
            print('Activating Enterprise Scene') if ENTERPRISE_SCENE else print('Deactivating Enterprise Scene')

def get_background_image(image_path):
    origin_bg_frame = cv.imread(image_path)
    origin_height = origin_bg_frame.shape[0]
    origin_width = origin_bg_frame.shape[1]
    
    h_ratio = FRAME_HEIGHT / origin_height
    w_ratio = FRAME_WIDTH / origin_width
    
    if h_ratio > 1 or w_ratio > 1: # need positive scale ?
        if h_ratio > w_ratio: #need scale by height ratio?
            new_bg_frame = cv.resize(origin_bg_frame ,None,fx=h_ratio, fy=h_ratio, interpolation = cv.INTER_CUBIC)
            new_bg_frame = new_bg_frame[:,:FRAME_WIDTH]
        else:
            new_bg_frame = cv.resize(origin_bg_frame ,None,fx=w_ratio, fy=w_ratio, interpolation = cv.INTER_CUBIC)
            new_bg_frame = new_bg_frame[:FRAME_HEIGHT,:]
    else:
        if h_ratio > w_ratio: #need scale by height ratio?
            new_bg_frame = cv.resize(origin_bg_frame ,None,fx=h_ratio, fy=h_ratio, interpolation = cv.INTER_CUBIC)
            new_bg_frame = new_bg_frame[:,:FRAME_WIDTH]
        else:
            new_bg_frame = cv.resize(origin_bg_frame ,None,fx=w_ratio, fy=w_ratio, interpolation = cv.INTER_CUBIC)
            new_bg_frame = new_bg_frame[:FRAME_HEIGHT,:]
    
    return new_bg_frame

def resize_foreground(condition : np.ndarray, frame : np.ndarray, factor : float, offset : (float, float)):
    full_size_shape = condition.shape
    resized_shape = (int(full_size_shape[0]*factor), int(full_size_shape[1]*factor), full_size_shape[2])
    offset = (int(full_size_shape[0]*offset[0]), int(full_size_shape[1]*offset[1]))
        
    condition_int = condition.astype(dtype='uint8')
    resized_condition_int = cv.resize(condition_int, (resized_shape[1], resized_shape[0]), interpolation = cv.INTER_CUBIC)
    new_condition = np.full(full_size_shape, False, dtype='bool')
    new_condition[offset[0]:offset[0]+resized_shape[0], offset[1]:offset[1]+resized_shape[1]] = resized_condition_int.astype(dtype='bool')
    
    new_frame = np.full(full_size_shape, 0, dtype='uint8')
    resized_frame = cv.resize(frame, (resized_shape[1], resized_shape[0]), interpolation = cv.INTER_CUBIC)
    new_frame[offset[0]:offset[0]+resized_shape[0], offset[1]:offset[1]+resized_shape[1]] = resized_frame
    
    return new_condition, new_frame

running_mode = VisionRunningMode.LIVE_STREAM if RUN_MODE else VisionRunningMode.VIDEO
landmarker_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=hand_landmark_model_path),
    running_mode=running_mode,
    num_hands = 2,
    result_callback = hand_detected_handle if RUN_MODE else None
    )
landmarker = HandLandmarker.create_from_options(landmarker_options)

if SEGMENTER_MODEL == 0:
    segmenter_model_path = selfie_segmenter_model_path
elif SEGMENTER_MODEL == 1:
    segmenter_model_path = selfie_segmenter_landscape_model_path
elif SEGMENTER_MODEL == 2:
    segmenter_model_path = multiclass_selfie_segmenter_model_path

segmenter_options = ImageSegmenterOptions(
    base_options=BaseOptions(segmenter_model_path),
    running_mode=VisionRunningMode.VIDEO,
    output_category_mask=True)
segmenter = ImageSegmenter.create_from_options(segmenter_options)


cap = cv.VideoCapture(CAMERA_IDX)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

cap.set(cv.CAP_PROP_CONVERT_RGB, 1)

cap.set(cv.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

FRAME_WIDTH = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
FRAME_HEIGHT = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

fps = cap.get(cv.CAP_PROP_FPS)

vcam = pyvirtualcam.Camera(FRAME_WIDTH, FRAME_HEIGHT, fps)

bg_image_vulcan = get_background_image(bg_image_path_vulcan)
bg_image_default = get_background_image(bg_image_path_default)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    frame_timestamp_ms = int(cap.get(cv.CAP_PROP_POS_MSEC)*1000)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv.cvtColor(frame, cv.COLOR_BGR2RGB))

    if RUN_MODE:
        landmarker.detect_async(mp_image, frame_timestamp_ms)
        if ENTERPRISE_SCENE:
            bg_image = bg_image_vulcan
        else: 
            bg_image = bg_image_default if DEFAULT_BG_IMAGE_BLUR == 0 else cv.blur(frame, (60,60))

        segmented_masks = segmenter.segment_for_video(mp_image, frame_timestamp_ms)
        category_mask = segmented_masks.category_mask.numpy_view()
        confidence_masks = segmented_masks.confidence_masks[0].numpy_view()
        
        if SEGMENTER_MODEL <= 1:
            condition = np.stack((confidence_masks,) * 3, axis=-1) > 0.3
        else:
            condition = np.stack((confidence_masks,) * 3, axis=-1) < 0.3
        
        if ENTERPRISE_SCENE:
            condition, frame = resize_foreground(condition, frame, 0.5, (0.273, 0.30))
        
        output_image = np.where(condition, frame, bg_image)

        if OUTPUT_MODE == 0:
            cv.imshow("teste", output_image)
            if cv.waitKey(1) == ord('q'):
                break
        else:
            vcam.send(cv.cvtColor(output_image, cv.COLOR_BGR2RGB))

    else:
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        if len(result.handedness) != 0:
            annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), result, is_vulcan_greeting(result))
        else:
            annotated_image = mp_image.numpy_view()

        cv.imshow("teste", cv.cvtColor(annotated_image, cv.COLOR_RGB2BGR))

        if cv.waitKey(1) == ord('q'):
            break


# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
