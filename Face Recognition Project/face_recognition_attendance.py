import cv2
import mediapipe as mp
import os
import json
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf

# Suppress TensorFlow log messages
tf.get_logger().setLevel('ERROR')

# Initialize Mediapipe face detection tools
mp_face_detection = mp.solutions.face_detection

# Function to load dataset of known faces
def load_dataset(dataset_path="dataset"):
    face_encodings = {}
    for file in os.listdir(dataset_path):
        if file.endswith(".jpg") or file.endswith(".png"):
            img_path = os.path.join(dataset_path, file)
            name = os.path.splitext(file)[0]
            image = cv2.imread(img_path)
            face_encoding = extract_face_encoding(image)
            if face_encoding is not None:
                face_encodings[name] = face_encoding
    return face_encodings

# Function to extract face encoding
def extract_face_encoding(image):
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.detections:
            bbox = results.detections[0].location_data.relative_bounding_box
            h, w, _ = image.shape
            x_min = max(0, int(bbox.xmin * w))
            y_min = max(0, int(bbox.ymin * h))
            x_max = min(w, x_min + int(bbox.width * w))
            y_max = min(h, y_min + int(bbox.height * h))
            cropped_face = image[y_min:y_max, x_min:x_max]
            resized_face = cv2.resize(cropped_face, (100, 100))
            return resized_face.flatten()
    return None

# Function to recognize face
def recognize_face(face_encoding, dataset):
    best_match_name = None
    best_match_score = 0.0

    for name, known_encoding in dataset.items():
        known_encoding = np.array(known_encoding).reshape(1, -1)
        face_encoding = np.array(face_encoding).reshape(1, -1)
        similarity = cosine_similarity(known_encoding, face_encoding)[0][0]
        if similarity > best_match_score:
            best_match_score = similarity
            best_match_name = name

    return best_match_name if best_match_score > 0.8 else None

# Function to save the output to a JSON file
def save_to_json(name=None):
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S") if name else None
    output = {
        "name": name,
        "time": timestamp
    }
    
    # Define the directory and file path
    directory = r"C:\Users\Shrik\OneDrive\Documents\UiPath\face recognition project"  # Update this with your desired folder path
    file_path = os.path.join(directory, "attendance.json")
    
    # Ensure the directory exists before saving
    if not os.path.exists(directory):
        os.makedirs(directory)  # Create directory if it doesn't exist
    
    # Save to file
    try:
        with open(file_path, "w") as json_file:
            json.dump(output, json_file, indent=4)
        print(f"Attendance saved: {name} at {timestamp}")
    except Exception as e:
        print(f"Error saving to JSON: {e}")

# Main function to run face recognition
def main():
    dataset = load_dataset()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        face_encoding = extract_face_encoding(frame)
        if face_encoding is not None:
            matched_name = recognize_face(face_encoding, dataset)
            if matched_name:
                save_to_json(matched_name)
                cap.release()
                cv2.destroyAllWindows()
                return matched_name  # Only the name will be returned

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    save_to_json()  # Save empty JSON if no face is recognized
    return "No face recognized"

if __name__ == "__main__":
    print(main())  # Only name output
