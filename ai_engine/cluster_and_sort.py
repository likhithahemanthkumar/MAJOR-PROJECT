import os
import shutil
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.cluster import KMeans

RAW_DIR = "ai_engine/dataset/oct_raw/"  # Change to fundus_raw/ if processing fundus
OUT_DIR = "ai_engine/dataset/organized_stages/"
STAGES = ["0_Preclinical", "1_Early_Stage", "2_MCI", "3_Severe"]

for stage in STAGES:
    os.makedirs(os.path.join(OUT_DIR, stage), exist_ok=True)

# Load frozen feature extractor
encoder = ResNet50(weights="imagenet", include_top=False, pooling="avg")

features_list = []
file_paths = []

print("Analyzing unlabelled eye scans and extracting structural feature vectors...")
for fname in os.listdir(RAW_DIR):
    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        path = os.path.join(RAW_DIR, fname)
        img = image.load_img(path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        feat = encoder.predict(x, verbose=0)
        features_list.append(feat.flatten())
        file_paths.append(path)

# Cluster into 4 Alzheimer's risk profiles
print("Segmenting data profiles via K-Means Clustering...")
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(np.array(features_list))

for path, label_idx in zip(file_paths, labels):
    fname = os.path.basename(path)
    dest = os.path.join(OUT_DIR, STAGES[label_idx], fname)
    shutil.copy(path, dest)

print("SUCCESS: Your raw image pool has been programmatically sorted into 4 clinical stages!")