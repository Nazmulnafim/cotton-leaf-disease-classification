import streamlit as st
from tensorflow.keras import models
import numpy as np
from PIL import Image
import cv2

# Load the pre-trained model
loaded_model = models.load_model("model_mobilenetv2_densenet201_9941_acc.hdf5")

def welcome():
    st.write("Hello World")

def capture_image():
    # Capture an image using the webcam
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        st.write("Failed to open webcam")
        return

    # Capture and display frames until the user presses ESC
    while True:
        success, frame = video_capture.read()

        if not success:
            break

        cv2.imshow('Webcam', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27: # ESC key to quit
            break

    video_capture.release()

    # Convert the captured image to RGB format and resize it
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(rgb_image, (224, 224))

    # Convert the image to a NumPy array
    image_array = np.array(resized_image)

    # Normalize the image pixel values
    image_array = image_array / 255

    # Add a batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Make predictions using the model
    preds = loaded_model.predict(image_array)
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

    return predicted_class, confidence

def handle_request():
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
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
        st.write({
            'Predicted class': predicted_class,
            'confidence': confidence
        })

    # Capture
# Run the Streamlit app
if __name__ == '__main__':
    welcome()
    handle_request()
