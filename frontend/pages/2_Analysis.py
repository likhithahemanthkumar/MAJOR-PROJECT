import streamlit as st
import numpy as np
import cv2

st.set_page_config(page_title="Diagnostic Analysis", layout="wide")
st.title("📊 DeepRetina Biomarker Analysis Core")

# Guard Clause: Verify if a patient scan has actually been pushed to memory yet
if "prediction_results" not in st.session_state:
    st.warning("⚠️ No active diagnostic data found. Please go back to the Dashboard upload screen first.")
else:
    # Pull variables out from global storage
    patient_id = st.session_state["patient_id"]
    modality = st.session_state["modality"]
    file_bytes = st.session_state["raw_image_bytes"]
    res_json = st.session_state["prediction_results"]

    st.info(f"Viewing Active Records for Patient Reference: **{patient_id}** | Modality: **{modality}**")

    # 1. VISUAL COMPARISON SPLIT
    st.subheader("Anatomical Pattern Optimization")
    img_arr = np.frombuffer(file_bytes, np.uint8)
    raw_img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    raw_img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.image(raw_img_rgb, caption="Submitted Raw Retinal Scan", use_container_width=True)
    with col_img2:
        # Applying structural filtering for visual optimization feedback
        gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        if "OCT" in modality:
            enhanced = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
            caption_str = "Bilateral Denoising Filter (Tissue Boundary Mapping)"
        else:
            enhanced = cv2.equalizeHist(gray) # Simulating blood vessel accentuation
            caption_str = "Histogram Contrast Equalization (Vascular Mapping)"
        st.image(enhanced, caption=caption_str, use_container_width=True)

    st.write("---")

    # 2. METRIC VISUALIZATION PANELS
    st.subheader("Neural Network Classification Targets")
    
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric(label="Predicted Cognitive Degradation Stage", value=res_json['stage'].replace("_", " "))
    with metric_col2:
        st.metric(label="Algorithm Model Confidence Assessment", value=f"{res_json['confidence'] * 100:.2f}%")

    # 3. CLINICAL ACTION PATH RECOMMENDATIONS
    st.subheader("Automated Clinical Action Path Recommendations")
    stage_str = res_json['stage']
    
    if "Preclinical" in stage_str:
        st.success("🟢 **Observation Protocol:** Normal structural parameters verified. Baseline metrics logged securely in historical tables.")
    elif "Early_Stage" in stage_str or "MCI" in stage_str:
        st.warning("🟡 **Intervention Protocol:** Noticeable layer thinning or vascular attenuation detected. Scheduling physical neurological cognitive testing is highly recommended.")
    else:
        st.error("🔴 **Critical Escalation Protocol:** Advanced neuro-retinal structural atrophy observed. Urgent referral to specialist neuro-ophthalmological clinical teams required.")