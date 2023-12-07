import flask
import werkzeug
from tensorflow.keras import models
import numpy as np
from PIL import Image
import os
from waitress import serve


app = flask.Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def welcome():
    return "Hello World"


@app.route('/predict/', methods=['GET', 'POST'])
def handle_request():
    def model_predict(model, image):
        x = np.array(image)
        x = x / 255
        x = np.expand_dims(x, axis=0)

        preds = model.predict(x)
        class_idx = np.argmax(preds, axis=1)[0]
        # confidence = float(preds[0][class_idx])
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

        return predicted_class, confidence

    imagefile = flask.request.files['image0']
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("/nReceived image File name: " + imagefile.filename)
    imagefile.save(filename)

    # Load and preprocess the image
    img = Image.open(filename).convert('RGB')
    resized_image = img.resize((224, 224))
    # img_array = np.array(img)
    # img_array = img_array / 255.0  # Normalize pixel values to between 0 and 1

    # Ensure the image has the correct shape (224, 224, 3) for RGB
    # if resized_image.shape != (224, 224, 3):
    #     return "Image size mismatch " + str(img_array.shape) + ".\nOnly (224, 224, 3) is acceptable."

    # Reshape the image array for prediction
    # img_array = img_array.reshape(224, 224, 3)

    # Load the pre-trained model
    loaded_model = models.load_model(
        "model_mobilenetv2_densenet201_9941_acc.hdf5")

    # Make predictions
    predicted_class, confidence = model_predict(loaded_model, resized_image)

    return {
        'Predicted class': predicted_class,
        'confidence': confidence
    }


# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=int(os.environ.get('PORT', 5000)), debug=True)

# if __name__ == '__main__':
#     app.run()

if __name__ == "__main__":
    serve(app, host="127.0.0.1", port=8000)

# waitress-serve --host=192.168.1.115 --port=5000 server:app
