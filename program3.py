import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers

# Define the DnCNN model for color images
def DnCNN(depth=17, filters=64, image_channels=3, use_bnorm=True):
    input_layer = tf.keras.Input(shape=(None, None, image_channels), name='input')
    x = layers.Conv2D(filters=filters, kernel_size=3, padding='same', activation='relu')(input_layer)

    for _ in range(depth-2):
        x = layers.Conv2D(filters=filters, kernel_size=3, padding='same')(x)
        if use_bnorm:
            x = layers.BatchNormalization(axis=3)(x)
        x = layers.Activation('relu')(x)

    output_layer = layers.Conv2D(filters=image_channels, kernel_size=3, padding='same')(x)

    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer, name='DnCNN')
    return model

# Load a pre-trained model or create a new one for color images
model = DnCNN(depth=17, filters=64, image_channels=3, use_bnorm=True)
model.compile(optimizer='adam', loss='mean_squared_error')

# Function to add noise to an image
def add_noise(image, noise_factor=0.5):
    noisy_image = image + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=image.shape)
    noisy_image = np.clip(noisy_image, 0., 1.)
    return noisy_image

# Load and preprocess a color image
def preprocess_image(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, color_mode='rgb')
    image = tf.keras.preprocessing.image.img_to_array(image).astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Specify the correct path to your color image file
image_path = '/home/NETID/vdhaya/data/DnCNN/image_to_test.png'

# Ensure the color image file exists at the specified path
import os
if not os.path.exists(image_path):
    raise FileNotFoundError(f"The specified image file does not exist: {image_path}")

image = preprocess_image(image_path)
noisy_image = add_noise(image)

# Denoise the color image using the DnCNN model
predicted_noise = model.predict(noisy_image)
denoised_image = noisy_image - predicted_noise

# Display the images
def display_images(original, noisy, denoised):
  plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(original[0])  # Ensure to plot back in the [0, 1] range for RGB images
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Noisy Image')
    plt.imshow(noisy[0])  # Ensure to plot back in the [0, 1] range for RGB images
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Denoised Image')
    plt.imshow(denoised[0])  # Ensure to plot back in the [0, 1] range for RGB images
    plt.axis('off')

    plt.show()

display_images(image, noisy_image, denoised_image)
