from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "model_weights", "vgg_unfrozen.h5")

IMG_SIZE = 128

# ✅ CORRECT CLASS MAPPING (FROM TRAINING)
CLASS_MAP = {
    0: "NORMAL",
    1: "PNEUMONIA"
}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- LOAD MODEL ----------------
model = load_model(MODEL_PATH)

# ---------------- IMAGE PREPROCESS ----------------
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ---------------- ROUTES ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_path = None
    error = None

    if request.method == "POST":
        # ✅ MATCHES HTML: name="image"
        file = request.files.get("image")

        if file is None or file.filename == "":
            error = "Please upload a chest X-ray image."
        else:
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(image_path)

            img = preprocess_image(image_path)
            preds = model.predict(img)[0]

            class_id = int(np.argmax(preds))
            prediction = CLASS_MAP[class_id]
            confidence = round(float(preds[class_id]) * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path,
        error=error
    )

# ---------------- RUN ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

