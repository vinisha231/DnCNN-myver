import tensorflow as tf
import numpy as np

# Load the model
model = tf.keras.models.load_model('dncnn_model.h5', compile=False)
model.compile(optimizer='adam', loss='mean_squared_error')

# Generate synthetic data for demonstration
def generate_noisy_images(images, noise_factor=0.5):
    noisy_images = images + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=images.shape)
    noisy_images = np.clip(noisy_images, 0., 1.)
    return noisy_images

# Load dataset (e.g., MNIST) and preprocess
(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.astype('float32') / 255.
test_images = test_images.astype('float32') / 255.

train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

noisy_train_images = generate_noisy_images(train_images)
noisy_test_images = generate_noisy_images(test_images)

# Train the model
model.fit(noisy_train_images, train_images, epochs=1, batch_size=64, validation_data=(noisy_test_images, test_images))

# Evaluate the model
loss = model.evaluate(noisy_test_images, test_images)
print(f"Test Loss: {loss}")

# Save the trained model
model.save('dncnn_trained_model.h5')
