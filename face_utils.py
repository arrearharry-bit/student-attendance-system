"""
face_utils.py – All face recognition helpers.
Extracted from app.py so it can be imported cleanly.
"""
import os
import pickle
import numpy as np
import face_recognition
from PIL import Image

TRAINING_PATH = 'Training images'
ENCODINGS_FILE = 'encodings.pickle'

# In-memory cache loaded at startup
_encode_list = []
_class_names = []

def get_encodings():
    """Return (encodeList, classNames) from memory."""
    return _encode_list, _class_names

def load_encodings_from_disk():
    """Load pre-computed encodings from pickle file (fast startup)."""
    global _encode_list, _class_names
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, 'rb') as f:
            data = pickle.load(f)
        _encode_list = data.get('encodings', [])
        _class_names = data.get('names', [])
        print(f"Loaded {len(_encode_list)} encodings from {ENCODINGS_FILE}")
        return True
    return False

def encode_all_images():
    """Encode all images in Training_images/ and save to pickle."""
    global _encode_list, _class_names
    _encode_list = []
    _class_names = []

    if not os.path.exists(TRAINING_PATH):
        os.makedirs(TRAINING_PATH)

    image_files = [
        f for f in os.listdir(TRAINING_PATH)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ]
    print(f"Encoding {len(image_files)} images...")

    for filename in image_files:
        filepath = os.path.join(TRAINING_PATH, filename)
        try:
            img = face_recognition.load_image_file(filepath)

            # Strip alpha channel if present
            if img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]

            # Force uint8 + C-contiguous (required by dlib)
            img = np.ascontiguousarray(img.astype(np.uint8))

            # Resize very large images
            h, w = img.shape[:2]
            if w > 1000:
                scale = 1000 / w
                pil_img = Image.fromarray(img)
                pil_img = pil_img.resize((int(w * scale), int(h * scale)))
                img = np.ascontiguousarray(np.array(pil_img).astype(np.uint8))

            # num_jitters=10 takes several random crops/shades to get a better 'mean' encoding
            encodings = face_recognition.face_encodings(img, num_jitters=10)
            if encodings:
                _encode_list.append(encodings[0])
                _class_names.append(os.path.splitext(filename)[0])
                print(f"  ✓ {filename} (High Quality)")
            else:
                print(f"  ✗ No face found: {filename}")
        except Exception as e:
            print(f"  Error processing {filename}: {e}")

    # Save to pickle
    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump({'encodings': _encode_list, 'names': _class_names}, f)

    print(f"Encoding complete. {len(_encode_list)} faces stored in {ENCODINGS_FILE}.")
    return len(_encode_list)

def load_encodings():
    """Load from pickle if available, otherwise encode from scratch."""
    if not load_encodings_from_disk():
        encode_all_images()

def recognize_frame(img_array, tolerance=0.6):
    """
    Given a numpy RGB image, return matched name or 'Unknown'.
    Returns: (name: str, confidence: float)
    """
    encode_list, class_names = get_encodings()
    if not encode_list:
        return 'Unknown', 0.0

    # Reduce resolution for speed (only by half now for better accuracy)
    small = np.ascontiguousarray(
        np.array(Image.fromarray(img_array).resize(
            (img_array.shape[1] // 2, img_array.shape[0] // 2)
        )).astype(np.uint8)
    )

    face_locs = face_recognition.face_locations(small)
    face_encs = face_recognition.face_encodings(small, face_locs)

    for enc, loc in zip(face_encs, face_locs):
        distances = face_recognition.face_distance(encode_list, enc)
        min_idx = int(np.argmin(distances))
        dist = float(distances[min_idx])
        
        # Log distance for debugging
        print(f"DEBUG: Closest match: {class_names[min_idx]} with distance: {dist:.4f}")
        
        if dist < tolerance:
            confidence = round(float(1 - dist), 3)
            return class_names[min_idx], confidence

    return 'Unknown', 0.0

def encode_single_image(filepath, name):
    """Encode a single new image and add it to the cache + pickle."""
    global _encode_list, _class_names

    try:
        img = face_recognition.load_image_file(filepath)
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        img = np.ascontiguousarray(img.astype(np.uint8))
        encodings = face_recognition.face_encodings(img, num_jitters=10)
        if encodings:
            _encode_list.append(encodings[0])
            _class_names.append(name)
            # Re-save pickle
            with open(ENCODINGS_FILE, 'wb') as f:
                pickle.dump({'encodings': _encode_list, 'names': _class_names}, f)
            return True
    except Exception as e:
        print(f"Encoding error for {name}: {e}")
    return False
