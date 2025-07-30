import tensorflow as tf
import numpy as np

# --- Configuration ---
# You can change this to the path of any image you want to test
IMAGE_PATH = 'C:/Users/rohit/OneDrive/Desktop/test1.png' 
IMG_HEIGHT = 224
IMG_WIDTH = 224
# Make sure these match the order from your training output
CLASS_NAMES = ['benign', 'malignant'] 

# --- 1. Load the Trained Model ---
print("Loading model...")
model = tf.keras.models.load_model('skin_lesion_model_transfer.keras')

# --- 2. Load and Preprocess the Image ---
print(f"Loading image: {IMAGE_PATH}")
# Load the image file
img = tf.keras.utils.load_img(
    IMAGE_PATH, target_size=(IMG_HEIGHT, IMG_WIDTH)
)
# Convert the image to a numpy array
img_array = tf.keras.utils.img_to_array(img)
# Add a batch dimension (the model expects a batch of images)
img_array = tf.expand_dims(img_array, 0) # Create a batch

# --- 3. Make a Prediction ---
print("Making a prediction...")
predictions = model.predict(img_array)
score = predictions[0][0] # Get the single prediction value

# --- 4. Interpret the Result ---
print("\n--- Prediction Result ---")
print(f"The model is {(1 - score)*100:.2f}% confident this is '{CLASS_NAMES[0]}'.")
print(f"The model is {score*100:.2f}% confident this is '{CLASS_NAMES[1]}'.")

if score < 0.5:
    print(f"\nFinal Verdict: This looks like a '{CLASS_NAMES[0]}' lesion.")
else:
    print(f"\nFinal Verdict: This looks like a '{CLASS_NAMES[1]}' lesion.")