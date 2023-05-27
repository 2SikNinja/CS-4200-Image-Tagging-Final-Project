import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tkinter import Tk, Label, Button, filedialog, Text, Canvas
from tensorflow.keras.models import load_model

# Load the pre-trained MobileNetV2 model
model = load_model('animal_classifier_model.h5')

# Function to predict the image using the loaded model


def predict_image(image_path, model):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    max_prob_index = np.argmax(predictions[0])

    animal_labels = ['Butterfly', 'Cat', 'Chicken', 'Cow',
                     'Dog', 'Elephant', 'Horse', 'Sheep', 'Spider', 'Squirrel']
    predicted_label = animal_labels[max_prob_index]

    max_prob = predictions[0][max_prob_index]

    return predicted_label, max_prob


# Define the animal labels
animal_labels = ['butterfly', 'cat', 'chicken', 'cow', 'dog',
                 'elephant', 'horse', 'sheep', 'spider', 'squirrel']

# Class for the image classifier GUI


class ImageClassifierGUI:
    def __init__(self, master):
        self.master = master
        master.geometry("800x600")
        master.title("Image Classifier")

        self.label = Label(
            master, text="Select an image to classify as Butterfly, Cat, Chicken, Cow, Dog, Elephant, Horse, Sheep, Spider, or Squirrel.")
        self.label.pack()

        self.select_button = Button(
            master, text="Select Image", command=self.select_image)
        self.select_button.pack()

        self.canvas = Canvas(master, width=400, height=400)
        self.canvas.pack()

        self.classification_text = Text(master, height=1, wrap="none")
        self.classification_text.pack()

    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            classification, confidence = predict_image(file_path, model)
            confidence_percentage = confidence * 100
            self.classification_text.delete(1.0, "end")
            self.classification_text.insert(
                "end", f"Classification: {classification} ({confidence_percentage:.2f}% confidence)")

            image = Image.open(file_path)
            image.thumbnail((400, 400))
            photo = ImageTk.PhotoImage(image)
            self.canvas.create_image(0, 0, image=photo, anchor="nw")
            self.canvas.image = photo


root = Tk()
gui = ImageClassifierGUI(root)
root.mainloop()
