from flask import Flask, request, send_from_directory, jsonify
import tensorflow as tf
import numpy as np
import os

# Initialize Flask
app = Flask(__name__, static_folder="static")

# Load Model
model = tf.keras.models.load_model("X-ray.h5")

# Upload Folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Function to Load & Process Image
def load(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=(150,150))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Serve Static HTML File (Without Jinja)
@app.route("/")
def home():
    return send_from_directory("template", "index.html")

# Handle File Upload & Prediction
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Save File
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # Load Image & Predict
    img_array = load(file_path)
    prediction = model.predict(img_array)
    result = "Pneumonia Detected" if prediction[0][0] > 0.5 else "Normal"


    return jsonify({"result": result})

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
