import tensorflow as tf
import matplotlib.pyplot as plt

# --- Configuration ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 15
DATA_DIR = '../data_multiclass/train' # Change this line

# --- 1. Load Data ---
print("\nLoading multi-class dataset...")
train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
  DATA_DIR,
  validation_split=0.2,
  subset="both",
  seed=123,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Class names:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- 2. Build Model ---
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal_and_vertical'),
  tf.keras.layers.RandomRotation(0.2),
])
preprocess_input = tf.keras.applications.efficientnet.preprocess_input

print("Building pre-trained EfficientNetB0 model...")
base_model = tf.keras.applications.EfficientNetB0(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False # Freeze the base model

inputs = tf.keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x) # Changed for multi-class
model = tf.keras.Model(inputs, outputs)

# --- 3. Compile and Train ---
print("\nCompiling model...")
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(), # Changed for multi-class
              metrics=['accuracy'])

print("\nStarting model training...")
history = model.fit(train_ds,
                    epochs=EPOCHS,
                    validation_data=val_ds)

# --- 4. Visualize and Save ---
print("\nPlotting results...")
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()

model.save('skin_lesion_model_multiclass.keras')
print("\nModel saved.")