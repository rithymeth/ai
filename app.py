# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load and Preprocess the MNIST Data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Reshape the images to add the channel dimension (from (28, 28) to (28, 28, 1))
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

# Resize images to 32x32 (VGG16 requires at least 32x32)
train_images = tf.image.resize(train_images, [32, 32])
test_images = tf.image.resize(test_images, [32, 32])

# Convert grayscale images to 3-channel RGB (since VGG16 expects 3 channels)
train_images = tf.image.grayscale_to_rgb(tf.convert_to_tensor(train_images)).numpy()
test_images = tf.image.grayscale_to_rgb(tf.convert_to_tensor(test_images)).numpy()

# Normalize the images to [0, 1] range by dividing by 255
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# One-hot encode the labels (i.e., convert digit labels into vectors)
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Step 2: Load Pre-trained VGG16 Model for Transfer Learning
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze the convolutional base to avoid retraining the pre-trained layers
for layer in vgg_model.layers:
    layer.trainable = False

# Step 3: Create the Custom Top Layer with Regularization
transfer_model = models.Sequential()
transfer_model.add(vgg_model)
transfer_model.add(layers.Flatten())

# Add L2 regularization to dense layers
transfer_model.add(layers.Dense(128, activation='relu', 
                                kernel_regularizer=regularizers.l2(0.001)))
# Add dropout to prevent overfitting
transfer_model.add(layers.Dropout(0.5))

# Output layer
transfer_model.add(layers.Dense(10, activation='softmax'))

# Compile the model
transfer_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
transfer_model.summary()

# Step 4: Implement Callbacks (Early Stopping, Learning Rate Reduction, and Model Checkpointing)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001)

# Use the .keras format for saving the model
checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')

# Step 5: Data Augmentation for Training
datagen = ImageDataGenerator(
    rotation_range=10, 
    zoom_range=0.1, 
    width_shift_range=0.1, 
    height_shift_range=0.1
)

# Fit the data generator on the training data
datagen.fit(train_images)

# Step 6: Train the Model with Callbacks and Data Augmentation
transfer_history = transfer_model.fit(datagen.flow(train_images, train_labels, batch_size=64),
                                      epochs=10, 
                                      validation_data=(test_images, test_labels),
                                      callbacks=[early_stopping, reduce_lr, checkpoint])

# Step 7: Fine-tune the Model
for layer in vgg_model.layers[-8:]:  # Unfreezing more layers for better fine-tuning
    layer.trainable = True

# Re-compile the model with a lower learning rate
transfer_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
                       loss='categorical_crossentropy', 
                       metrics=['accuracy'])

# Fine-tune the model
fine_tune_history = transfer_model.fit(datagen.flow(train_images, train_labels, batch_size=64),
                                       epochs=5, 
                                       validation_data=(test_images, test_labels),
                                       callbacks=[early_stopping, reduce_lr, checkpoint])

# Step 8: Evaluate the Best Saved Model
best_model = tf.keras.models.load_model('best_model.keras')

# Evaluate the best model
best_test_loss, best_test_acc = best_model.evaluate(test_images, test_labels)
print(f"Best Model Test Accuracy: {best_test_acc}")

# Step 9: Evaluate with Confusion Matrix and Classification Report
test_predictions = np.argmax(best_model.predict(test_images), axis=1)
true_labels = np.argmax(test_labels, axis=1)

# Confusion Matrix
conf_matrix = confusion_matrix(true_labels, test_predictions)
print("Confusion Matrix:\n", conf_matrix)

# Classification Report
class_report = classification_report(true_labels, test_predictions)
print("Classification Report:\n", class_report)

# Step 10: Visualizing Accuracy and Loss

# Plot training & validation accuracy values
plt.plot(fine_tune_history.history['accuracy'])
plt.plot(fine_tune_history.history['val_accuracy'])
plt.title('Model Accuracy with Fine-tuning')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(fine_tune_history.history['loss'])
plt.plot(fine_tune_history.history['val_loss'])
plt.title('Model Loss with Fine-tuning')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
