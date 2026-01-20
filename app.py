import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
# Load trained model
model = tf.keras.models.load_model("catDog.h5")

IMG_SIZE = 150  # must match training size

def predict_image(image):
    # Convert PIL image to array
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)

    if prediction > 0.5:
        return f"🐶 Dog"
    else:
        return f"🐱 Cat"


# Gradio Interface
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload Dog or Cat Image"),
    outputs=gr.Textbox(label="Prediction"),
    title="🐶🐱 Dog vs Cat Classifier",
    description="Upload an image to classify whether it is a Dog or a Cat"
)

interface.launch()
