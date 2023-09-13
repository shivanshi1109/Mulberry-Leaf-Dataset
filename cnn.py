import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from PIL import Image
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import random

# Step 1: Split the Data
data_dir = "./5.data_formatting"
disease_labels = ["disease free leaves", "leaf rust", "leaf spot"]

# Load and preprocess data
X = []
y = []
for label_idx, label in enumerate(disease_labels):
    label_dir = os.path.join(data_dir, label)
    for image_filename in os.listdir(label_dir):
        image_path = os.path.join(label_dir, image_filename)
        # Load the preprocessed image using Pillow (PIL)
        image = Image.open(image_path)
        image = np.array(image)  # Convert to a NumPy array
        X.append(image)
        y.append(label_idx)

X = np.array(X)
y = np.array(y)

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Reshape grayscale images for input data
X_train = X_train.reshape(-1, 224, 224, 1)
X_val = X_val.reshape(-1, 224, 224, 1)
X_test = X_test.reshape(-1, 224, 224, 1)

# One-hot encode labels
y_train_encoded = to_categorical(y_train, num_classes=len(disease_labels))
y_val_encoded = to_categorical(y_val, num_classes=len(disease_labels))
y_test_encoded = to_categorical(y_test, num_classes=len(disease_labels))

# Step 2: Choose a Machine Learning Model
model = Sequential()

# Add convolutional layers, pooling layers, and fully connected layers as needed
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)))  # Use input_shape=(224, 224, 1) for grayscale
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(len(disease_labels), activation='softmax'))  # Number of classes equals the length of disease_labels

# Step 3: Model Compilation
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 4: Data Augmentation (Optional)
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Step 5: Model Training
batch_size = 32
epochs = 10

history = model.fit(
    datagen.flow(X_train, y_train_encoded, batch_size=batch_size),
    validation_data=(X_val, y_val_encoded),
    steps_per_epoch=len(X_train) // batch_size,
    epochs=epochs
)

# Step 7: Evaluate the Model
test_loss, test_accuracy = model.evaluate(X_test, y_test_encoded)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

test_loss, test_accuracy = model.evaluate(X_test, y_test_encoded)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Generate predictions on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Generate a classification report
print(classification_report(np.argmax(y_test_encoded, axis=1), y_pred_classes, target_names=disease_labels))

# Generate a confusion matrix
cm = confusion_matrix(np.argmax(y_test_encoded, axis=1), y_pred_classes)
print("Confusion Matrix:")
print(cm)
# The rest of the code (Steps 6, 8, 9, and 10) remains the same.

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.show()

index = random.randint(0, len(X_test) - 1)
sample_image = X_test[index]
true_label = disease_labels[np.argmax(y_test_encoded[index])]

# Make a prediction
prediction = model.predict(np.expand_dims(sample_image, axis=0))
predicted_label = disease_labels[np.argmax(prediction)]

# Display the image and prediction
plt.figure(figsize=(4, 4))
plt.imshow(np.squeeze(sample_image), cmap='gray')
plt.title(f"True Label: {true_label}\nPredicted Label: {predicted_label}")
plt.axis('off')
plt.show()
