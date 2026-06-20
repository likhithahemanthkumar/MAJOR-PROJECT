import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, UploadFile, File, Form
import numpy as np
import cv2
import tensorflow as tf
from backend.database import generate_file_hash, check_cached_hash, save_scan_record, fetch_all_history
from ai_engine.preprocess import clean_fundus_image, clean_oct_image

app = FastAPI(title="DeepRetina AI Gateway")

# Global variables to store our model in system memory
MODEL_PATH = "ai_engine/deepretina_model.h5"
model = None
STAGES = ["0_Preclinical", "1_Early_Stage", "2_MCI", "3_Severe"]

@app.on_event("startup")
def load_deep_learning_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Backend Model loaded into GPU/CPU memory successfully.")
    else:
        print("WARNING: deepretina_model.h5 not found. Please execute training script first.")

@app.post("/predict")
async def predict_alzheimers_risk(
    file: UploadFile = File(...),
    patient_id: str = Form(...),
    modality: str = Form(...)
):
    image_bytes = await file.read()
    img_hash = generate_file_hash(image_bytes)
    
    # 1. Integrity check: Serve instant cached prediction if hash exists
    cached = check_cached_hash(img_hash)
    if cached:
        return {"status": "Cached", "stage": cached[0], "confidence": cached[1], "hash": img_hash}
        
    # Decode bytes back to OpenCV raw format matrix
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 2. Preprocess based on selected scanning technology
    if modality == "Optical Coherence Tomography (OCT)":
        processed_img = clean_oct_image(img)
    else:
        processed_img = clean_fundus_image(img)
        
    # Prepare dimensions for model tensor expectation (1, 224, 224, 3)
    input_tensor = np.expand_dims(processed_img, axis=0)
    
    # 3. Model Inference Engine execution
    if model is None:
        return {"status": "Error", "message": "Model not initialized"}
        
    predictions = model.predict(input_tensor)[0]
    best_idx = np.argmax(predictions)
    predicted_stage = STAGES[best_idx]
    confidence_value = float(predictions[best_idx])
    
    # 4. Save results to secure history tables
    save_scan_record(patient_id, img_hash, modality, predicted_stage, confidence_value)
    
    return {"status": "Processed", "stage": predicted_stage, "confidence": confidence_value, "hash": img_hash}

@app.get("/history")
def get_system_logs():
    return {"history": fetch_all_history()}