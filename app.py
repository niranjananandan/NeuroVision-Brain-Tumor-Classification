import os

# Force TensorFlow to use CPU on Render
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from flask import Flask, render_template, request
import tensorflow as tf
import cv2
import numpy as np

app = Flask(__name__)

# Load trained NeuroVision model
model = tf.keras.models.load_model("neurovision_model.h5")

print("NeuroVision model loaded successfully!")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    error = None

    if request.method == "POST":

        # Check whether image was uploaded
        if "image" not in request.files:
            error = "Please select an MRI image."

        else:
            file = request.files["image"]

            if file.filename == "":
                error = "Please select an MRI image."

            else:

                # Temporary image path
                temp_path = "temp_image.jpg"

                try:
                    # Save uploaded image temporarily
                    file.save(temp_path)

                    # Read image as grayscale
                    img = cv2.imread(
                        temp_path,
                        cv2.IMREAD_GRAYSCALE
                    )

                    if img is None:
                        error = "Unable to read the uploaded image."

                    else:

                        # Same preprocessing used during training
                        img = cv2.resize(
                            img,
                            (128, 128)
                        )

                        img = img.astype(np.float32) / 255.0

                        img = img.reshape(
                            1,
                            128,
                            128,
                            1
                        )

                        # Model prediction
                        prediction_score = float(
                            model.predict(
                                img,
                                verbose=0
                            )[0][0]
                        )

                        # Classification
                        if prediction_score > 0.5:
                            prediction = "TUMOR"
                            confidence = prediction_score * 100
                        else:
                            prediction = "NORMAL"
                            confidence = (1 - prediction_score) * 100

                except Exception as e:
                    error = "An error occurred while processing the image."
                    print("Prediction error:", e)

                finally:
                    # Always delete temporary image
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        error=error
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
