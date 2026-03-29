"""
Backward compatibility wrapper for mediapipe 0.10.x
Provides the old solutions API using the new tasks API
"""
import mediapipe as mp_new
import numpy as np
from dataclasses import dataclass
import os
import urllib.request
import tempfile


# Download mediapipe model if needed
def _download_model():
    """Download the pose landmarker model"""
    model_path = os.path.expanduser('~/.mediapipe/pose_landmarker.task')
    
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    if not os.path.exists(model_path):
        print("Downloading mediapipe pose model...")
        url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker/float16/1/pose_landmarker.task'
        try:
            urllib.request.urlretrieve(url, model_path)
            print(f"Model downloaded to {model_path}")
        except Exception as e:
            print(f"Could not download model: {e}")
            return None
    
    return model_path if os.path.exists(model_path) else None


@dataclass
class NormalizedLandmark:
    """Represents a normalized landmark"""
    x: float
    y: float
    z: float
    visibility: float = None


@dataclass
class PoseLandmarkList:
    """Represents a list of pose landmarks"""
    landmark: list


@dataclass
class PoseResult:
    """Fake pose detection result"""
    pose_landmarks: PoseLandmarkList = None
    pose_world_landmarks: PoseLandmarkList = None


class Pose:
    """Backward compatible Pose class using mediapipe tasks"""
    
    def __init__(self, static_image_mode=False, model_complexity=1,
                 smooth_landmarks=True, enable_segmentation=False,
                 smooth_segmentation=True, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Get the model path
        model_path = _download_model()
        
        # Create the new mediapipe pose landmarker
        if model_path:
            base_options = mp_new.tasks.BaseOptions(
                model_asset_path=model_path
            )
        else:
            base_options = mp_new.tasks.BaseOptions()
        
        options = mp_new.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_new.tasks.vision.RunningMode.IMAGE if static_image_mode
            else mp_new.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_tracking_confidence
        )
        
        try:
            self.pose_landmarker = mp_new.tasks.vision.PoseLandmarker.create_from_options(options)
        except Exception as e:
            print(f"Warning: Could not create PoseLandmarker: {e}")
            print("Using dummy landmarker")
            self.pose_landmarker = None
    
    def process(self, image):
        """Process an image and return pose landmarks"""
        if self.pose_landmarker is None:
            # Return empty result if landmarker could not be created
            return PoseResult()
        
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = image[:, :, ::-1]  # BGR to RGB
            else:
                rgb_image = image
            
            # Create MediaPipe Image
            mp_image = mp_new.Image(image_format=mp_new.ImageFormat.SRGB, data=rgb_image)
            
            # Detect pose
            detection_result = self.pose_landmarker.detect(mp_image)
            
            # Convert to old format
            if detection_result.pose_landmarks:
                landmarks = []
                for lm in detection_result.pose_landmarks[0]:
                    landmarks.append(NormalizedLandmark(
                        x=lm.x,
                        y=lm.y,
                        z=lm.z,
                        visibility=lm.visibility if hasattr(lm, 'visibility') else None
                    ))
                
                pose_landmarks = PoseLandmarkList(landmark=landmarks)
            else:
                pose_landmarks = None
            
            result = PoseResult(pose_landmarks=pose_landmarks)
            return result
        except Exception as e:
            print(f"Error processing image: {e}")
            return PoseResult()


class PoseLandmark:
    """Landmark indices"""
    # Define all landmark indices
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


# Create a fake solutions module
class _Solutions:
    def __init__(self):
        self.pose = type('pose', (), {
            'Pose': Pose,
            'PoseLandmark': PoseLandmark
        })()
        self.drawing_utils = mp_new.tasks.vision.drawing_utils


# Monkey patch mediapipe to add solutions compatibility
mp_new.solutions = _Solutions()
