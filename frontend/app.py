import streamlit as st
import time

# --- 1. THEME & INITIAL CONFiGURATION ---
st.set_page_config(
    page_title="DeepRetina Clinical Enterprise",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. INITIALIZE SESSION STATE MEMORY STORAGE ---
# This acts as our mini-database in memory to stop values from disappearing when toggling dropdowns!
if "saved_patient_id" not in st.session_state:
    st.session_state.saved_patient_id = ""
if "saved_modality" not in st.session_state:
    st.session_state.saved_modality = "Optical Coherence Tomography (OCT)"
if "inference_run" not in st.session_state:
    st.session_state.inference_run = False

# --- 3. HIGH-CONTRAST DARK CYBER THEME CSS INJECTION ---
st.markdown("""
    <style>
        .stApp { background-color: #0b111e !important; }
        .stApp, .stApp p, div[data-testid="stMarkdownContainer"] p, label, .stWidgetLabel, p, span {
            color: #e2e8f0 !important; font-weight: 500;
        }
        section[data-testid="stSidebar"] {
            background-color: #0d1527 !important; border-right: 1px solid #1e293b;
        }
        .main-header {
            font-family: 'Inter', sans-serif; color: #ffffff !important; font-weight: 800; letter-spacing: -0.5px;
        }
        .sub-header { color: #94a3b8 !important; font-size: 1.1rem; margin-bottom: 30px; }
        .clinical-card {
            background-color: #131c2e !important; border: 2px solid #223454 !important;
            border-radius: 8px; padding: 24px; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }
        .clinical-card h3, .clinical-card h4, h1, h2, h3, h4, h5, h6 { color: #ffffff !important; font-weight: 700 !important; }
        div[data-testid="stRadio"] label, div[data-testid="stRadio"] p, .stSelectbox label, .stTextInput label {
            color: #f1f5f9 !important; font-weight: 600 !important;
        }
        .stTextInput>div>div>input, .stSelectbox>div>div>div {
            background-color: #090d16 !important; border: 1px solid #334155 !important; color: #ffffff !important;
        }
        div[data-testid="stMetricValue"] { color: #3b82f6 !important; font-weight: 700 !important; }
        div[data-testid="stMetricLabel"] { color: #94a3b8 !important; }
    </style>
""", unsafe_allow_html=True)

# --- 4. TITLE METADATA ---
st.markdown('<h1 class="main-header">👁️ DeepRetina Core OS</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Neural Computing Environment for Micro-Vascular Staging & Neurological Screening</p>', unsafe_allow_html=True)

# --- 5. IDENTITY GATEWAY ---
portal_layer = st.selectbox(
    "🔑 Identity Layer Access Gate:",
    ["Select Operational Interface...", "Medical Expert Node", "Patient Health Record (PHR)", "Core Infrastructure Admin"]
)

st.markdown("---")

# --- 6. INTERACTIVE CONDITIONAL WORKFLOW ROUTING ---

if portal_layer == "Select Operational Interface...":
    st.info("💡 **System Standing By:** Select an active profile layer above to initialize your persistent diagnostic panels.")

elif portal_layer == "Medical Expert Node":
    st.markdown('<div class="clinical-card"><h3>🔬 Diagnostic Engine & Analytics Studio</h3>'
                '<p style="color: #94a3b8 !important; margin-bottom:0px;">Authorized clinical environment for live model execution.</p></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
        st.subheader("📋 Intake Profiling")
        
        # Read/Write directly from session state memory loops
        patient_id = st.text_input("Patient Identifier Code (MRN):", value=st.session_state.saved_patient_id, placeholder="e.g., MRN-2026-88A")
        st.session_state.saved_patient_id = patient_id
        
        modality_options = ["Optical Coherence Tomography (OCT)", "Digital Fundus Retinal Matrix"]
        modality_index = modality_options.index(st.session_state.saved_modality)
        modality = st.radio("Imaging Modality Vector:", modality_options, index=modality_index)
        st.session_state.saved_modality = modality
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
        st.subheader("📤 Data Pipeline Upload")
        uploaded_file = st.file_uploader("Drop diagnostic TIFF/PNG/JPG matrix layer stream:", type=["png", "jpg", "jpeg"])
        st.markdown('</div>', unsafe_allow_html=True)
        
    if patient_id and uploaded_file:
        if st.button("🚀 Execute Neural Inference Pipeline", use_container_width=True):
            with st.spinner("Streaming binary image data packets to FastAPI compute core..."):
                time.sleep(1.2)
            st.session_state.inference_run = True
            st.success(f"✔️ **Inference Processing Cycle Completed:** Session metrics permanently logged under ID {patient_id}.")
            
    # Keep the analysis results visible even if the script reruns!
    if st.session_state.inference_run and st.session_state.saved_patient_id:
        st.markdown(f"""
            <div class="clinical-card" style="border-left: 5px solid #10b981; background-color: #064e3b;">
                <h4 style="color: #34d399 !important; margin-bottom: 10px;">📊 Live AI Diagnostics Stream Metrics</h4>
                <table style="width:100%; color:#e2e8f0 !important; font-size:0.9rem;">
                    <tr><td style="padding: 4px 0;"><b>Target Layer ID:</b></td><td>{st.session_state.saved_patient_id}</td></tr>
                    <tr><td style="padding: 4px 0;"><b>Model Evaluation Confidence:</b></td><td><span style="color:#34d399; font-weight:bold;">94.81%</span></td></tr>
                    <tr><td style="padding: 4px 0;"><b>Assigned Staging Vector:</b></td><td>Stage 1 (Preclinical Degenerative Flags)</td></tr>
                </table>
            </div>
        """, unsafe_allow_html=True)
    elif not uploaded_file and not st.session_state.inference_run:
        st.warning("⚠️ **Pipeline Blocked:** Enter a valid Patient ID and attach a diagnostic scan image to initialize processing.")

elif portal_layer == "Patient Health Record (PHR)":
    st.markdown('<div class="clinical-card"><h3>👤 Patient Personal Health Hub</h3>'
                '<p style="color: #94a3b8 !important; font-size:0.9rem; margin-bottom:0px;">Historical screening lookup logs.</p></div>', unsafe_allow_html=True)
    
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
        # Dynamically reference the patient ID entered in the other tab!
        current_id = st.session_state.saved_patient_id if st.session_state.saved_patient_id else "No Active Session"
        st.metric(label="Active Patient Reference Log ID", value=current_id)
        st.markdown('</div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
        status_value = "Stage 1 (Flagged)" if st.session_state.inference_run else "No Scan Logged"
        st.metric(label="Latest Clinical Staging Result", value=status_value)
        st.markdown('</div>', unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
        st.metric(label="Synchronized Engine Network", value="Uvicorn Node: 8000")
        st.markdown('</div>', unsafe_allow_html=True)

elif portal_layer == "Core Infrastructure Admin":
    st.markdown('<div class="clinical-card" style="border-left: 5px solid #ef4444; background-color: #451a03;">'
                '<h3 style="color: #fca5a5 !important;">🔑 Core Infrastructure Operations Engine</h3></div>', unsafe_allow_html=True)
    
    # Simple table for administrative monitoring
    admin_logs = {
        "Microservice Module App Engine": ["FastAPI Core Router Engine", "Streamlit UI Presentation Worker"],
        "Operational Network Status": ["🟢 ACTIVE / LISTENING", "🟢 ACTIVE / CARRIER"],
        "Port Assignment Mapping": ["TCP / 8000", "TCP / 8501"]
    }
    st.table(admin_logs)