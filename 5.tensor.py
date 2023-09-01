import os
import numpy as np
import tensorflow as tf

height = 224    # Height of the images in pixels
width = 224     # Width of the images in pixels
channels = 1    # Number of color channels (1 for grayscale)

batch_size = 162
input_folder = './minmax/leaf spot'
output_folder = './data_formatting/leaf spot'

# Get a list of image file names in the input folder
image_files = os.listdir(input_folder)

# Initialize an empty list to store processed images
processed_images = []

# Load and preprocess each image
for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(height, width), color_mode="grayscale")
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    processed_images.append(image_array)

# Convert the list of processed images into a numpy array
images_np = np.array(processed_images)

# Normalize pixel values to be in the range [0, 1]
images_np = images_np / 255.0

# Convert numpy array to TensorFlow tensor
images_tf = tf.convert_to_tensor(images_np, dtype=tf.float32)

# Reshape the tensor to match the CNN input shape (batch_size x height x width x channels)
images_tf_reshaped = tf.reshape(images_tf, (batch_size, height, width, channels))

# Save the reshaped images to the output folder
for i in range(batch_size):
    output_path = os.path.join(output_folder, f'processed_image_{i}.png')
    processed_image = tf.keras.preprocessing.image.array_to_img(images_tf_reshaped[i])
    processed_image.save(output_path)
