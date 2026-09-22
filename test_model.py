import tensorflow as tf
import cv2
import numpy as np

print("Loading NeuroVision model...")

model = tf.keras.models.load_model(
    "neurovision_model.h5",
    compile=False
)

print("Model loaded successfully!")

image_path = "Y164.JPG"

img = cv2.imread(
    image_path,
    cv2.IMREAD_GRAYSCALE
)

if img is None:
    print("Error: Unable to load the MRI image.")
    exit()

img = cv2.resize(
    img,
    (128, 128),
    interpolation=cv2.INTER_AREA
)

img = img.astype(np.float32) / 255.0

img = img.reshape(
    1,
    128,
    128,
    1
)

prediction = model.predict(
    img,
    verbose=0
)[0][0]

prediction = float(prediction)

if prediction >= 0.5:
    result = "TUMOR"
    confidence = prediction * 100
else:
    result = "NORMAL"
    confidence = (1 - prediction) * 100

print("\nPrediction Score:", prediction)
print("Result:", result)
print("Confidence:", f"{confidence:.2f}%")