import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "neurovision_model.h5")

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MAX_FILE_SIZE = 10 * 1024 * 1024

app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

model = None


def allowed_file(filename):
    if not filename or "." not in filename:
        return False

    extension = filename.rsplit(".", 1)[1].lower()
    return extension in ALLOWED_EXTENSIONS


def load_neurovision_model():
    global model

    if model is None:

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "neurovision_model.h5 was not found."
            )

        print("Loading NeuroVision model...")

        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        print("NeuroVision model loaded successfully.")
        print("Input shape:", model.input_shape)
        print("Output shape:", model.output_shape)

    return model


def preprocess_image(file):
    image = Image.open(file)

    image = image.convert("L")

    image = image.resize(
        (128, 128),
        Image.Resampling.LANCZOS
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=-1
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        if "image" not in request.files:
            return jsonify({
                "success": False,
                "error": "Please select an MRI image."
            }), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({
                "success": False,
                "error": "Please select an MRI image."
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "error": "Invalid file type. Please upload JPG, JPEG or PNG."
            }), 400

        neuro_model = load_neurovision_model()

        image_array = preprocess_image(file)

        prediction_output = neuro_model.predict(
            image_array,
            verbose=0
        )

        prediction_score = float(
            np.asarray(prediction_output).reshape(-1)[0]
        )

        if prediction_score >= 0.5:

            prediction = "TUMOR"
            confidence = prediction_score * 100

        else:

            prediction = "NORMAL"
            confidence = (1.0 - prediction_score) * 100

        confidence = max(
            0.0,
            min(100.0, confidence)
        )

        print("Prediction:", prediction)
        print("Confidence:", f"{confidence:.2f}%")

        return jsonify({
            "success": True,
            "prediction": prediction,
            "confidence": round(confidence, 2)
        })

    except Exception as e:

        print("Prediction error:", repr(e))

        return jsonify({
            "success": False,
            "error": "Unable to analyze the MRI image. Please try another image."
        }), 500


@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "success": False,
        "error": "File is too large. Maximum allowed size is 10 MB."
    }), 413


@app.route("/health")
def health():
    return "NeuroVision is running successfully!", 200


if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )