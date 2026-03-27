"""
face_utils.py – Stable OpenCV Face Recognition Engine (LBPH).
Simplified for maximum reliability. No complex dependencies.
"""
import os
import pickle
import numpy as np
import cv2

TRAINING_PATH = 'Training images'
LBPH_MODEL = 'face_model.xml'
NAMES_FILE = 'face_names.pickle'
HAAR_CASCADE = 'haarcascade_frontalface_default.xml'

# Initialize recognizer and detector
_recognizer = cv2.face.LBPHFaceRecognizer_create()
_detector = cv2.CascadeClassifier(HAAR_CASCADE if os.path.exists(HAAR_CASCADE) else cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
_label_to_name = {}

def get_encodings():
    """Returns list of student names from the mapping file."""
    if os.path.exists(NAMES_FILE):
        with open(NAMES_FILE, 'rb') as f:
            mapping = pickle.load(f)
            return list(mapping.values()), list(mapping.values())
    return [], []

def load_encodings():
    """Load the trained model and label-to-name mapping."""
    global _label_to_name
    if os.path.exists(LBPH_MODEL) and os.path.exists(NAMES_FILE):
        try:
            _recognizer.read(LBPH_MODEL)
            with open(NAMES_FILE, 'rb') as f:
                _label_to_name = pickle.load(f)
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
    return False

def encode_all_images():
    """Trains the LBPH recognizer from scratch using images in Training images folder."""
    global _label_to_name
    if not os.path.exists(TRAINING_PATH):
        os.makedirs(TRAINING_PATH)
        
    faces = []
    labels = []
    _label_to_name = {}
    
    images_list = [f for f in os.listdir(TRAINING_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Training on {len(images_list)} images...")
    
    for idx, filename in enumerate(images_list):
        name = os.path.splitext(filename)[0]
        path = os.path.join(TRAINING_PATH, filename)
        img_bgr = cv2.imread(path)
        if img_bgr is None: continue
        
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        detected_faces = _detector.detectMultiScale(gray, 1.1, 5)
        
        for (x, y, w, h) in detected_faces:
            faces.append(gray[y:y+h, x:x+w])
            labels.append(idx)
            _label_to_name[idx] = name
            print(f"  ✓ Trained {name}")
            break # Use first face per image

    if faces:
        _recognizer.train(faces, np.array(labels))
        _recognizer.save(LBPH_MODEL)
        with open(NAMES_FILE, 'wb') as f:
            pickle.dump(_label_to_name, f)
            
    return len(_label_to_name)

def recognize_frame(img_rgb):
    """
    Detects and recognizes faces in an RGB frame.
    Returns list of (name, confidence_dist, (top, right, bottom, left)).
    """
    # 1. Resize for speed (0.25x as requested)
    small_frame = cv2.resize(img_rgb, (0, 0), fx=0.25, fy=0.25)
    
    # 2. To Grayscale for OpenCV detection
    gray = cv2.cvtColor(small_frame, cv2.COLOR_RGB2GRAY)
    
    # 3. Detect faces
    detected_faces = _detector.detectMultiScale(gray, 1.3, 5)
    results = []
    
    for (x, y, w, h) in detected_faces:
        roi = gray[y:y+h, x:x+w]
        try:
            id, dist = _recognizer.predict(roi)
            # LBPH distance (lower is better, < 100-120 is usually a match)
            name = "Unknown"
            if dist < 120:
                name = _label_to_name.get(id, "Unknown")
            
            # Format as (top, right, bottom, left) and scale back up x4
            results.append((name, dist, (y*4, (x+w)*4, (y+h)*4, x*4)))
        except:
            results.append(("Unknown", 0.0, (y*4, (x+w)*4, (y+h)*4, x*4)))
            
    return results

def encode_single_image(filepath, name):
    """Simple trigger to rebuild the whole model on new student."""
    return encode_all_images() > 0
