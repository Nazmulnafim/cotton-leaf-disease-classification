from flask import Flask, request, jsonify
from tensorflow.keras import models
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load the pre-trained model
loaded_model = models.load_model("model_mobilenetv2_densenet201_9941_acc.hdf5")

@app.route('/', methods=['GET', 'POST'])
def welcome():
    return "Cotton Leaf Disease Classification"

@app.route('/predict/', methods=['POST'])
def handle_request():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        uploaded_file = request.files['file']

        if uploaded_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Load and preprocess the image
        img = Image.open(uploaded_file).convert('RGB')
        resized_image = img.resize((224, 224))

        # Model prediction
        x = np.array(resized_image)
        x = x / 255
        x = np.expand_dims(x, axis=0)

        preds = loaded_model.predict(x)
        class_idx = np.argmax(preds, axis=1)[0]
        confidence = round(float(preds[0][class_idx]), 3)

        class_labels = [
            "The leaf is diseased cotton leaf. The disease is Aphids",
            "The leaf is diseased cotton plant. The disease is Army worm",
            "The leaf is diseased cotton leaf. The disease is Bacterial blight",
            "The leaf is Healthy.",
            "Powdery mildew",
            "The leaf is diseased cotton plant. The disease is Target Spot",
            "The leaf is diseased cotton plant. The disease is Curl Virus",
            "The leaf is diseased cotton plant. The disease is Fussium Wilt"
        ]
        predicted_class = class_labels[class_idx]

        # Display the results
        return {
            'Predicted class': predicted_class,
            'confidence': confidence
        }
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({'error': 'An error occurred while processing the request'}), 500

if __name__ == '__main__':
    # Run with Gunicorn
    app.run(host='0.0.0.0', port=8000)
