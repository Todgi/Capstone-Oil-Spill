import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import glob
import seaborn as sns
import tensorflow as tf
from keras import layers, models, callbacks
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, AveragePooling2D
from keras.layers import Concatenate, concatenate, BatchNormalization, Dropout, Lambda, Activation
from keras.applications import ResNet50
from tqdm import tqdm
from skimage.io import imread, imshow
from skimage.transform import resize
import random
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_fscore_support
from keras import backend as K

# Parameters
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CLASSES = 5
IMG_CHANNELS = 3

# Paths
IMG_PATH = 'C:/Users/alvito/Documents/00. Capstone/archive/oil-spill/train/images'
LABELS_PATH = 'C:/Users/alvito/Documents/00. Capstone/archive/oil-spill/train/labels'
IMG_PATH_TEST = 'C:/Users/alvito/Documents/00. Capstone/archive/oil-spill/test/images'
LABELS_PATH_TEST = 'C:/Users/alvito/Documents/00. Capstone/archive/oil-spill/test/labels'

# Get image filenames
IMG_IDS = sorted(os.listdir(IMG_PATH))
LABELS_IDS = sorted(os.listdir(LABELS_PATH))
IMG_IDS_TEST = sorted(os.listdir(IMG_PATH_TEST))
LABELS_IDS_TEST = sorted(os.listdir(LABELS_PATH_TEST))

# Define COLOR_MAP (sesuaikan dengan kebutuhan Anda)
COLOR_MAP = {
    0: [0, 0, 0],      # Background
    1: [255, 0, 0],    # Class 1
    2: [0, 255, 0],    # Class 2
    3: [0, 0, 255],    # Class 3
    4: [255, 255, 0]   # Class 4
}

def process_mask(mask, color_map):
    """Convert RGB mask to one-hot encoded mask"""
    # Create empty one-hot encoded mask
    one_hot_mask = np.zeros((mask.shape[0], mask.shape[1], len(color_map)), dtype=np.uint8)
    
    # Map each color to corresponding class
    for class_idx, color in color_map.items():
        # Find pixels matching current color
        class_pixels = np.all(mask == color, axis=-1)
        one_hot_mask[..., class_idx] = class_pixels.astype(np.uint8)
    
    return one_hot_mask

def load_and_preprocess_data():
    train_images = []
    train_masks = []
    
    for image_filename, mask_filename in tqdm(zip(IMG_IDS, LABELS_IDS), total=len(IMG_IDS)):
        image_path = os.path.join(IMG_PATH, image_filename)
        mask_path = os.path.join(LABELS_PATH, mask_filename)
        
        # Check if files exist
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            continue
        if not os.path.exists(mask_path):
            print(f"Mask file not found: {mask_path}")
            continue
            
        # Load and process image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Failed to read image: {image_path}")
            continue
            
        image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
        image = image / 255.0
        
        # Load and process mask
        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        mask = cv2.resize(mask, (IMG_HEIGHT, IMG_WIDTH))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        processed_mask = process_mask(mask, COLOR_MAP)
        grayscale_mask = np.argmax(processed_mask, axis=-1)
        grayscale_mask = np.expand_dims(grayscale_mask, axis=-1)
        
        train_images.append(image)
        train_masks.append(grayscale_mask)
    
    # Convert to numpy arrays
    train_images = np.array(train_images)
    train_masks = np.array(train_masks)
    
    return train_images, train_masks

def load_trained_model(model_path):
    model = load_model(model_path)
    return model

def predict_mask(model, image):
    # Preprocess the image
    image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    
    # Make prediction
    pred_mask = model.predict(image)
    pred_mask = np.argmax(pred_mask, axis=-1)
    pred_mask = np.expand_dims(pred_mask, axis=-1)
    
    return pred_mask[0]

def main():
    # Load trained model
    model_path = 'c:/Users/alvito/Documents/00. Capstone/webgis/models/saved_unetmodel_tf'
    model = load_trained_model(model_path)
    
    # Load and preprocess data
    train_images, train_masks = load_and_preprocess_data()
    
    # Example prediction
    sample_image = train_images[0]
    predicted_mask = predict_mask(model, sample_image)
    
    # Visualization
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(sample_image)
    plt.title('Input Image')
    
    plt.subplot(1, 3, 2)
    plt.imshow(train_masks[0].squeeze(), cmap='gray')
    plt.title('Ground Truth')
    
    plt.subplot(1, 3, 3)
    plt.imshow(predicted_mask.squeeze(), cmap='gray')
    plt.title('Predicted Mask')
    
    plt.show()

if __name__ == "__main__":
    main()