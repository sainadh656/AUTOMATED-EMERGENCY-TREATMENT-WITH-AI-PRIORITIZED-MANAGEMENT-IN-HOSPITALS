import os
import cv2
import face_recognition
import numpy as np
import pickle
from sklearn import svm
from sklearn.preprocessing import LabelEncoder




IMG_SIZE = 100
dataset_dir = "Biometric_Data/faces"
MODEL_FILE = "models/svm_face_model.pkl"
LABEL_FILE = "models/svm_face_labels.pkl"

# =========================================================
# ✅ FUNCTION 1 → TRAIN MODEL
# =========================================================
def face_train_model():
    X = []
    y = []

    print("Loading dataset...")

    for person_name in os.listdir(dataset_dir):
        person_path = os.path.join(dataset_dir, person_name)

        if not os.path.isdir(person_path):
            continue

        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)

            img = cv2.imread(image_path)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            features = gray.flatten()

            X.append(features)
            y.append(person_name)

    if len(X) == 0:
        print("Dataset empty!")
        return

    X = np.array(X)
    y = np.array(y)

    print("Samples Loaded:", len(X))

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    print("Training SVM...")

    model = svm.SVC(kernel='linear', probability=True)
    model.fit(X, y_encoded)

    pickle.dump(model, open(MODEL_FILE, "wb"))
    pickle.dump(encoder, open(LABEL_FILE, "wb"))

    print("✅ Training Complete & Model Saved")


# =========================================================
# ✅ FUNCTION 2 → PREDICT IMAGE
# =========================================================
def predict_face(image_path):
    try:
        model = pickle.load(open(MODEL_FILE, "rb"))
        encoder = pickle.load(open(LABEL_FILE, "rb"))
    except:
        print("❌ Model not found. Train first.")
        return

    img = cv2.imread(image_path)

    if img is None:
        print("❌ Invalid Image")
        return

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features = gray.flatten().reshape(1, -1)

    prediction = model.predict(features)
    probabilities = model.predict_proba(features)

    label = encoder.inverse_transform(prediction)[0]
    confidence = np.max(probabilities)

    print("✅ Predicted User:", label)
    print("✅ Confidence:", confidence)

    return label, confidence


import numpy as np

def compare_faces(img1_path, img2_path):

    # Load images
    img1 = face_recognition.load_image_file(img1_path)
    img2 = face_recognition.load_image_file(img2_path)

    # Encode faces
    encodings1 = face_recognition.face_encodings(img1)
    encodings2 = face_recognition.face_encodings(img2)

    if len(encodings1) == 0 or len(encodings2) == 0:
        print("No face detected in one of the images.")
        return False, 0

    face1 = encodings1[0]
    face2 = encodings2[0]

    # Compute distance
    distance = np.linalg.norm(face1 - face2)

    # Convert distance → confidence
    confidence = (1 - distance) * 100
    confidence = max(0, min(100, confidence))
    confidence = round(confidence, 2)

    # Typical threshold ~0.6
    matched = distance < 0.6

    print("\n===== FACE COMPARISON RESULT =====")
    print("Distance:", round(distance, 4))
    print("Matched:", matched)
    print("Confidence:", confidence, "%")

    return matched, confidence