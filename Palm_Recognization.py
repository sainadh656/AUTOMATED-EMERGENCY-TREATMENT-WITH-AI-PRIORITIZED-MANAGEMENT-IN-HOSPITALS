import os
import cv2
import numpy as np
import pickle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ---------------- SETTINGS ----------------
DATASET_DIR = "Biometric_Data/palms"
IMG_SIZE = 64
MODEL_PATH = "models/palm_cnn_model.h5"
ENCODER_PATH = "models/label_encoder.pkl"

# =========================================================
# ✅ FUNCTION 1 → TRAIN & SAVE MODEL
# =========================================================
def palm_train_model():

    images = []
    labels = []

    for user in os.listdir(DATASET_DIR):
        user_path = os.path.join(DATASET_DIR, user)

        if os.path.isdir(user_path):
            for img_name in os.listdir(user_path):

                img_path = os.path.join(user_path, img_name)

                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

                images.append(img)
                labels.append(user)

    images = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    images = images / 255.0

    # Encode labels
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    categorical_labels = to_categorical(encoded_labels)

    # Save encoder
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)

    X_train, X_test, y_train, y_test = train_test_split(
        images, categorical_labels, test_size=0.2, random_state=42
    )

    # Build model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(128, activation='relu'),
        Dense(len(le.classes_), activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

    model.save(MODEL_PATH)

    print("✅ Model trained & saved successfully!")


# =========================================================
# ✅ FUNCTION 2 → PREDICT IMAGE
# =========================================================
def palm_predict_image(image_path):

    model = load_model(MODEL_PATH)

    with open(ENCODER_PATH, "rb") as f:
        le = pickle.load(f)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    img = img / 255.0

    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    predicted_user = le.inverse_transform([class_index])[0]

    print("Predicted User:", predicted_user)
    print("Confidence:", confidence)

    if confidence > 0.90:
        print("✅ Genuine Prediction")
    else:
        print("❌ Low Confidence Prediction")
    return predicted_user, confidence

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# 1. Load Pretrained Model
# -----------------------------
model = models.resnet50(pretrained=True)

# Remove classification layer → keep feature extractor
model = nn.Sequential(*list(model.children())[:-1])
model.eval()

# -----------------------------
# 2. Image Preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# 3. Feature Extraction Function
# -----------------------------
def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        features = model(image)

    features = features.squeeze().numpy()
    return features

# -----------------------------
# 4. Similarity Function
# -----------------------------
def palm_similarity(img1, img2):
    feat1 = extract_features(img1)
    feat2 = extract_features(img2)

    similarity = cosine_similarity(
        feat1.reshape(1, -1),
        feat2.reshape(1, -1)
    )[0][0]

    confidence = round(similarity * 100, 2)
    print(confidence)
    print(f"Similarity Score: {similarity}")
    print(f"Confidence: {confidence}%")

    if similarity > 0.75:
        print("Result: Same Palm (Match)")
    else:
        print("Result: Different Palm (No Match)")

    return similarity, confidence