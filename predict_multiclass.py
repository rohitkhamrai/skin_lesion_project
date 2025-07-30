import tensorflow as tf
import numpy as np

# --- Configuration ---
# Change this to the path of an image you want to test
IMAGE_PATH = 'C:/Users/rohit/OneDrive/Desktop/test1.png'
IMG_HEIGHT = 224
IMG_WIDTH = 224

# IMPORTANT: These class names must match the output from your Colab training cell
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# --- 1. Load the Trained Model ---
print("Loading multi-class model...")
model = tf.keras.models.load_model('skin_lesion_model_multiclass.keras')

# --- 2. Load and Preprocess the Image ---
print(f"Loading image: {IMAGE_PATH}")
img = tf.keras.utils.load_img(
    IMAGE_PATH, target_size=(IMG_HEIGHT, IMG_WIDTH)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = np.expand_dims(img_array, 0) # Create a batch
img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

# --- 3. Make a Prediction ---
print("Making a prediction...")
predictions = model.predict(img_array)

# --- 4. Interpret the Result ---
# Use argmax to find the index of the class with the highest probability
predicted_class_index = np.argmax(predictions[0])
predicted_class_name = CLASS_NAMES[predicted_class_index]
confidence_score = predictions[0][predicted_class_index]

print("\n--- Prediction Result ---")
print(f"The model predicts this is a '{predicted_class_name}'.")
print(f"Confidence: {confidence_score*100:.2f}%")

print("\n--- All Class Probabilities ---")
for i, class_name in enumerate(CLASS_NAMES):
    print(f"{class_name}: {predictions[0][i]*100:.2f}%")