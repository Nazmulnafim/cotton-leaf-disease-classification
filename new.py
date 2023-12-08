import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from tensorflow.keras import models
import numpy as np
from PIL import Image

# Load the pre-trained model
loaded_model = models.load_model("model_mobilenetv2_densenet201_9941_acc.hdf5")

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = loaded_model
        self.class_labels = [
            "The leaf is diseased cotton leaf. The disease is Aphids",
            "The leaf is diseased cotton plant. The disease is Army worm",
            "The leaf is diseased cotton leaf. The disease is Bacterial blight",
            "The leaf is Healthy.",
            "Powdery mildew",
            "The leaf is diseased cotton plant. The disease is Target Spot",
            "The leaf is diseased cotton plant. The disease is Curl Virus",
            "The leaf is diseased cotton plant. The disease is Fussium Wilt"
        ]

    def transform(self, frame):
        # Convert the frame to a PIL Image
        img = Image.fromarray(frame)
        
        # Resize the image to match the model's input size
        resized_image = img.resize((224, 224))
        
        # Model prediction
        x = np.array(resized_image)
        x = x / 255
        x = np.expand_dims(x, axis=0)

        preds = self.model.predict(x)
        class_idx = np.argmax(preds, axis=1)[0]
        confidence = round(float(preds[0][class_idx]), 3)
        predicted_class = self.class_labels[class_idx]

        # Display the results on the frame
        st.write({
            'Predicted class': predicted_class,
            'Confidence': confidence
        })

# Run the Streamlit app
def main():
    st.header("Cotton Leaf Disease Classification with Camera")
    
    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

if __name__ == '__main__':
    main()
