import tensorflow as tf
import cv2
import numpy as np

print("Loading NeuroVision model...")

model = tf.keras.models.load_model("neurovision_model.h5")

print("Model loaded successfully! ✅")

# Load MRI image
image_path = "Y164.JPG"

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Resize exactly like training
img = cv2.resize(img, (128, 128))

# Normalize
img = img / 255.0

# Reshape for CNN
img = img.reshape(1, 128, 128, 1)

# Prediction
prediction = model.predict(img, verbose=0)[0][0]

print("\nPrediction Score:", prediction)

if prediction > 0.5:
    print("Result: TUMOR")
else:
    print("Result: NORMAL")