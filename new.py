import streamlit as st
from tensorflow.keras import models
import numpy as np
from PIL import Image

# Load the pre-trained model
loaded_model = models.load_model("model_mobilenetv2_densenet201_9941_acc.hdf5")

@st.route('/')
def welcome():
    st.write("Hello World")

@st.route('/predict/')
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

# Run the Streamlit app
if __name__ == '__main__':
    st.run()
