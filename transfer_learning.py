import tensorflow as tf
import matplotlib.pyplot as plt
import os

# --- Configuration ---
IMG_HEIGHT = 224 # EfficientNetB0 uses 224x224 images
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 10 # Initial training epochs
FINE_TUNE_EPOCHS = 5 # Epochs for fine-tuning
DATA_DIR = 'data/train'

# --- 1. Load the Data ---
print("Loading and splitting the dataset...")
train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
  DATA_DIR,
  validation_split=0.2,
  subset="both",
  seed=123,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE)

class_names = train_ds.class_names
print("Class names:", class_names)

# Configure the dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- 2. Create the Model using EfficientNetB0 ---
# Define data augmentation layers
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal_and_vertical'),
  tf.keras.layers.RandomRotation(0.2),
])

# Preprocessing layer specific to EfficientNet
preprocess_input = tf.keras.applications.efficientnet.preprocess_input

# Load the pre-trained EfficientNetB0 model
# include_top=False means we don't include the final classifier layer
# weights='imagenet' means we use the weights learned from the ImageNet dataset
print("Loading pre-trained EfficientNetB0 model...")
base_model = tf.keras.applications.EfficientNetB0(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
                                               include_top=False,
                                               weights='imagenet')

# Freeze the base model so its weights don't change during initial training
base_model.trainable = False

# Build the new model
inputs = tf.keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False) # Set training=False for frozen layers
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs, outputs)

# --- 3. Compile and Train the Model (Initial Phase) ---
print("Compiling the model for initial training...")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

model.summary()

print("\nStarting initial training...")
history = model.fit(train_ds,
                    epochs=EPOCHS,
                    validation_data=val_ds)

# --- 4. Fine-Tuning ---
# Unfreeze the base model to fine-tune it
base_model.trainable = True

# Let's see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Freeze all the layers before the `fine_tune_at` layer
fine_tune_at = 100 # Fine-tune from this layer onwards
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable = False

# Compile the model for fine-tuning with a very low learning rate
print("\nRe-compiling the model for fine-tuning...")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), # Lower learning rate
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

print("\nStarting fine-tuning...")
history_fine = model.fit(train_ds,
                         epochs=EPOCHS + FINE_TUNE_EPOCHS,
                         initial_epoch=history.epoch[-1],
                         validation_data=val_ds)

# --- 5. Visualize and Save ---
# Combine history objects
history.history['accuracy'].extend(history_fine.history['accuracy'])
history.history['val_accuracy'].extend(history_fine.history['val_accuracy'])
history.history['loss'].extend(history_fine.history['loss'])
history.history['val_loss'].extend(history_fine.history['val_loss'])

# Plotting
plt.figure(figsize=(12, 6))
# ... (plotting code remains the same as before)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()

# Save the final model
model.save('skin_lesion_model_transfer.keras')
print("\nModel saved as skin_lesion_model_transfer.keras")