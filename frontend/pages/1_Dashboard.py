import streamlit as st
import requests

st.set_page_config(page_title="Upload Dashboard", layout="centered")
st.title("🩺 Patient Intake & Upload Console")

patient_id = st.text_input("Enter Unique Patient ID", value="PT-8801")
modality = st.radio("Select Retinal Imaging Modality Type", 
                    ["Color Fundus Photography (CFP)", "Optical Coherence Tomography (OCT)"])
uploaded_file = st.file_uploader("Upload Raw Ophthalmic Scan File", type=["png", "jpg", "jpeg"])

if uploaded_file is not None and st.button("Submit to DeepRetina Engine"):
    file_bytes = uploaded_file.read()
    
    with st.spinner("Communicating with neural core..."):
        files = {"file": (uploaded_file.name, file_bytes, uploaded_file.type)}
        data = {"patient_id": patient_id, "modality": modality}
        
        try:
            # Send data to FastAPI backend
            response = requests.post("http://127.0.0.1:8000/predict", files=files, data=data)
            res_json = response.json()
            
            # Save all raw and calculated data into global Session State memory
            st.session_state["patient_id"] = patient_id
            st.session_state["modality"] = modality
            st.session_state["raw_image_bytes"] = file_bytes
            st.session_state["prediction_results"] = res_json
            
            st.success("Data processed successfully! Proceeding to the Analysis tab.")
            st.info("👈 Please click on '2 Analysis' in the sidebar to review the detailed clinical results.")
            
        except requests.exceptions.ConnectionError:
            st.error("Backend Server Offline. Ensure uvicorn is running on port 8000.")