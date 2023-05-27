import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model


# Set the path to the folder containing the raw images from Kaggle which contains 10 categories of animals
root_folder = 'E:\Documents\School\Spring 2023\CS 4200 Artificial Intelligence\FINALPROJECTFROMSCRATCH\/raw-img'

# Set the image size and batch size for training
image_size = (224, 224)
batch_size = 32

# Set the number of training epochs
num_epochs = 10

# Create an ImageDataGenerator for data augmentation and normalization
data_generator = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Create train and validation generators using the ImageDataGenerator
train_generator = data_generator.flow_from_directory(
    root_folder,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = data_generator.flow_from_directory(
    root_folder,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Load the pre-trained MobileNetV2 model without the top layer
base_model = MobileNetV2(
    weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add a global average pooling layer and a dense output layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=num_epochs
)

# Save the trained model
model.save('animal_classifier_model.h5')
