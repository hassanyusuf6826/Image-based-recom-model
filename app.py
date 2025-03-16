# import streamlit as st
# import numpy as np
# import pickle as pkl
# import tensorflow as tf
# from tensorflow.keras.applications.resnet50 import preprocess_input
# from tensorflow.keras.preprocessing.image import img_to_array, load_img
# from PIL import Image as PILImage  # Import fixed
# from sklearn.neighbors import NearestNeighbors
# from numpy.linalg import norm

# # Load saved features and filenames
# image_features_path = "Images_features.pkl"
# filenames_path = "filenames.pkl"

# with open(image_features_path, "rb") as f:
#     image_features = pkl.load(f)

# with open(filenames_path, "rb") as f:
#     filenames = pkl.load(f)

# # Load pre-trained model
# model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# model = tf.keras.models.Sequential([model, tf.keras.layers.GlobalMaxPool2D()])

# # Build Nearest Neighbors Model
# neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='cosine')
# neighbors.fit(image_features)

# # Feature Extraction Function
# def feature_extraction(img_path, model):
#     img = load_img(img_path, target_size=(224, 224))  # Load image correctly
#     img_array = img_to_array(img)  # Convert to array
#     expanded_img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions
#     preprocessed_img = preprocess_input(expanded_img_array)  # Preprocess image
#     result = model.predict(preprocessed_img).flatten()  # Extract features
#     normalized_result = result / norm(result)  # Normalize feature vector
#     return normalized_result

# # Fix recommend function
# def recommend_similar_images(uploaded_img):
#     temp_path = "temp_uploaded_image.jpg"  # Temporary save path
#     uploaded_img.save(temp_path)  # Save the image
#     feature_vector = feature_extraction(temp_path, model)  # Pass the file path
#     distances, indices = neighbors.kneighbors([feature_vector])
#     return [filenames[i] for i in indices[0]]

# # Streamlit UI
# st.title("Fashion Recommendation System")
# st.write("Upload an image to find visually similar fashion items.")

# uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

# if uploaded_file:
#     image = PILImage.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)
#     st.write("Finding recommendations...")

#     recommendations = recommend_similar_images(image)  # Fixed function call

#     st.write("## Recommended Items:")
#     cols = st.columns(len(recommendations))
#     for col, img_path in zip(cols, recommendations):
#         col.image(img_path, use_column_width=True)

import os
import streamlit as st
import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image as PILImage  # Import fixed
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Load saved features and filenames
image_features_path = "Images_features.pkl"
filenames_path = "filenames.pkl"

with open(image_features_path, "rb") as f:
    image_features = pkl.load(f)

with open(filenames_path, "rb") as f:
    filenames = pkl.load(f)

# Load pre-trained model
model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = tf.keras.models.Sequential([model, tf.keras.layers.GlobalMaxPool2D()])

# Build Nearest Neighbors Model
neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='cosine')
neighbors.fit(image_features)

# Feature Extraction Function
def feature_extraction(img_path, model):
    img = load_img(img_path, target_size=(224, 224))  # Load image correctly
    img_array = img_to_array(img)  # Convert to array
    expanded_img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions
    preprocessed_img = preprocess_input(expanded_img_array)  # Preprocess image
    result = model.predict(preprocessed_img).flatten()  # Extract features
    normalized_result = result / norm(result)  # Normalize feature vector
    return normalized_result

# Update recommend function
def recommend_similar_images(uploaded_img):
    temp_path = "temp_uploaded_image.jpg"
    uploaded_img.save(temp_path)  # Save uploaded image locally
    feature_vector = feature_extraction(temp_path, model)

    distances, indices = neighbors.kneighbors([feature_vector])
    
    # Load recommended images from local 'images/' directory
    recommended_images = [os.path.join("ecommerce", os.path.basename(filenames[i])) for i in indices[0]]
    
    return recommended_images

# Streamlit UI
st.title("Fashion Recommendation System")
st.write("Upload an image to find visually similar fashion items.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = PILImage.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)  # Fixed

    st.write("Finding recommendations...")
    recommendations = recommend_similar_images(image)  # Fixed function call

    st.write("## Recommended Items:")
    cols = st.columns(len(recommendations))
    for col, img_path in zip(cols, recommendations):
        if os.path.exists(img_path):  # Check if file exists
            col.image(img_path, use_container_width=True)  # Fixed
        else:
            st.warning(f"Image {img_path} not found.")

