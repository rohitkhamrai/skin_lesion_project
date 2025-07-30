import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- Page Configuration ---
st.set_page_config(page_title="Skin Lesion Analyzer", layout="wide", initial_sidebar_state="expanded")

# --- Sidebar ---
with st.sidebar:
    st.title("🔬 About")
    st.write("""
    This app uses a deep learning model (EfficientNetB0) to classify skin lesions from images. 
    It was trained on the HAM10000 dataset and can identify 7 different types of lesions.
    """)
    st.warning("This is an educational tool and not a substitute for professional medical advice.", icon="⚠️")

# --- Mappings and Model Loading ---
CLASS_INFO = {
    'akiec': ('Actinic Keratoses', 'Pre-cancerous'),
    'bcc': ('Basal Cell Carcinoma', 'Malignant'),
    'bkl': ('Benign Keratosis-like Lesions', 'Benign'),
    'df': ('Dermatofibroma', 'Benign'),
    'mel': ('Melanoma', 'Malignant'),
    'nv': ('Melanocytic Nevi (Mole)', 'Benign'),
    'vasc': ('Vascular Lesions', 'Benign')
}
CLASS_NAMES = list(CLASS_INFO.keys())

@st.cache_resource
def load_my_model():
    """Loads the trained Keras model."""
    model = tf.keras.models.load_model('skin_lesion_model_multiclass.keras')
    return model

model = load_my_model()

# --- Main Page ---
st.title("Advanced Skin Lesion Analyzer")
st.write("Upload an image of a skin lesion for a detailed classification.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # --- Create Columns for Layout ---
    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file)

    with col1:
        st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    with col2:
        with st.spinner('Classifying...'):
            # Preprocess the image
            img_array = np.array(image.resize((224, 224)))
            if img_array.ndim == 2:
                img_array = np.stack((img_array,)*3, axis=-1)
            if img_array.shape[2] == 4:
                img_array = img_array[...,:3]
            img_array = np.expand_dims(img_array, axis=0)
            img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
            
            # Make prediction
            prediction = model.predict(img_array)
            
            # Interpret result
            predicted_class_index = np.argmax(prediction[0])
            predicted_class_name = CLASS_NAMES[predicted_class_index]
            confidence_score = prediction[0][predicted_class_index]
            full_name, category = CLASS_INFO[predicted_class_name]

        st.subheader("Prediction Result")
        if category == 'Malignant' or category == 'Pre-cancerous':
            st.error(f"### Verdict: {full_name} ({category})", icon="⚠️")
        else:
            st.success(f"### Verdict: {full_name} ({category})", icon="✅")
        
        st.metric(label="Confidence", value=f"{confidence_score*100:.2f}%")
        
        with st.expander("See All Class Probabilities"):
            for i, class_name in enumerate(CLASS_NAMES):
                st.write(f"**{CLASS_INFO[class_name][0]}:** `{prediction[0][i]*100:.2f}%`")