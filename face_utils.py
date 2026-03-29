"""
face_utils.py – Modern MediaPipe + OpenCV Face Recognition Engine.
Replaces legacy Haar Cascades & LBPH with precise MediaPipe Face Mesh.
"""
import os
import pickle
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

TRAINING_PATH = 'Training images'
NAMES_FILE = 'face_encodings.pickle' # Changed to store embeddings
MODEL_PATH = 'face_landmarker.task'

# Download the model if it doesn't exist (failsafe)
if not os.path.exists(MODEL_PATH):
    import urllib.request
    print("Downloading MediaPipe Face Landmarker model...")
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        MODEL_PATH
    )

# Initialize MediaPipe Face Landmarker
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=10
)
detector = vision.FaceLandmarker.create_from_options(options)

_known_encodings = []  # List of numpy arrays
_known_names = []

def get_encodings():
    """Returns list of student names from the loaded encodings."""
    if os.path.exists(NAMES_FILE):
        with open(NAMES_FILE, 'rb') as f:
            data = pickle.load(f)
            return data['names'], data['names']
    return [], []

def load_encodings():
    """Load the trained encodings mapping."""
    global _known_encodings, _known_names
    if os.path.exists(NAMES_FILE):
        try:
            with open(NAMES_FILE, 'rb') as f:
                data = pickle.load(f)
                _known_encodings = data['encodings']
                _known_names = data['names']
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
    return False

def _extract_embedding(landmarks, image_shape):
    """
    Extracts a normalized, flattened array of 3D landmarks relative to the nose tip.
    image_shape is (H, W, C)
    """
    h, w, _ = image_shape
    
    # 478 landmarks in newer API (or 468)
    points = []
    for lm in landmarks: # list of NormalizedLandmark
        points.append([lm.x * w, lm.y * h, lm.z * w])
        
    points = np.array(points)
    
    # Use nose tip (landmark 1) as the center
    nose = points[1]
    
    # Center all points around the nose
    points_centered = points - nose
    
    # Find scale (distance between extreme left and right jaw/cheek points)
    scale = np.max(np.linalg.norm(points_centered, axis=1))
    if scale == 0:
        scale = 1.0
        
    points_normalized = points_centered / scale
    
    return points_normalized.flatten()

def encode_all_images():
    """Trains (extracts embeddings) using images in Training images folder."""
    global _known_encodings, _known_names
    if not os.path.exists(TRAINING_PATH):
        os.makedirs(TRAINING_PATH)
        
    encodings = []
    names = []
    
    images_list = [f for f in os.listdir(TRAINING_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Training on {len(images_list)} images using MediaPipe...")
    
    for idx, filename in enumerate(images_list):
        name = os.path.splitext(filename)[0]
        path = os.path.join(TRAINING_PATH, filename)
        img_bgr = cv2.imread(path)
        if img_bgr is None: continue
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results = detector.detect(mp_image)
        
        if results.face_landmarks:
            # Use the first detected face
            landmarks = results.face_landmarks[0]
            emb = _extract_embedding(landmarks, img_rgb.shape)
            encodings.append(emb)
            names.append(name)
            print(f"  ✓ Encoded {name}")
        else:
            print(f"  ✗ No face found in {filename}")

    if encodings:
        _known_encodings = encodings
        _known_names = names
        with open(NAMES_FILE, 'wb') as f:
            pickle.dump({'encodings': encodings, 'names': names}, f)
            
    return len(names)

def cosine_distance(a, b):
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    sim = dot / (norm_a * norm_b)
    # Cosine distance: 0 is completely identical, 2 is opposite
    return 1.0 - sim

def recognize_frame(img_rgb):
    """
    Detects and recognizes faces in an RGB frame.
    Returns list of (name, confidence_dist, (top, right, bottom, left)).
    """
    h, w, _ = img_rgb.shape
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    results = detector.detect(mp_image)
    recognized = []
    
    if results.face_landmarks:
        for face_landmarks in results.face_landmarks:
            # Extract bounding box from landmarks for drawing
            x_min = min([lm.x for lm in face_landmarks])
            x_max = max([lm.x for lm in face_landmarks])
            y_min = min([lm.y for lm in face_landmarks])
            y_max = max([lm.y for lm in face_landmarks])
            
            # Add padding to bbox and clamp to valid image boundaries
            padding = 0.1
            x_pad = (x_max - x_min) * padding
            y_pad = (y_max - y_min) * padding
            
            top, bottom = int(max(0, y_min - y_pad) * h), int(min(1, y_max + y_pad) * h)
            left, right = int(max(0, x_min - x_pad) * w), int(min(1, x_max + x_pad) * w)
            
            emb = _extract_embedding(face_landmarks, img_rgb.shape)
            
            best_dist = float('inf')
            best_name = "Unknown"
            
            if _known_encodings:
                for known_emb, known_name in zip(_known_encodings, _known_names):
                    d = cosine_distance(emb, known_emb)
                    if d < best_dist:
                        best_dist = d
                        best_name = known_name
            
            # Threshold for Cosine Distance (closer to 0 is better).
            # Usually < 0.05 or < 0.1 is good for FaceMesh embeddings.
            if best_dist < 0.15: # tuned threshold
                recognized.append((best_name, best_dist, (top, right, bottom, left)))
            else:
                recognized.append(("Unknown", best_dist, (top, right, bottom, left)))
                
    return recognized

def encode_single_image(filepath, name):
    """Simple trigger to rebuild the whole model on new student."""
    return encode_all_images() > 0