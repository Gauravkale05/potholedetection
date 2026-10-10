import os
import json
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
from pathlib import Path

# -------------------------------------------------------------
# PAGE CONFIGURATION & METADATA
# -------------------------------------------------------------
st.set_page_config(
    page_title="PotholeVision",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# PRODUCTION CSS STYLING
# -------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .top-navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        padding: 1.25rem 2rem;
        border-radius: 16px;
        color: white;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
        margin-bottom: 1.8rem;
    }
    
    .brand-title {
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #F8FAFC;
        margin: 0;
    }
    
    .brand-subtitle {
        font-size: 0.95rem;
        color: #94A3B8;
        margin-top: 0.25rem;
        font-weight: 500;
    }
    
    .status-pill {
        background: rgba(34, 197, 94, 0.15);
        border: 1px solid rgba(34, 197, 94, 0.3);
        color: #4ADE80;
        padding: 0.4rem 0.9rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .card-base {
        padding: 1.5rem;
        border-radius: 14px;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease;
    }
    
    .card-pothole {
        background-color: #FEF2F2;
        border: 1px solid #FCA5A5;
        border-left: 6px solid #DC2626;
    }
    
    .card-normal {
        background-color: #F0FDF4;
        border: 1px solid #86EFAC;
        border-left: 6px solid #16A34A;
    }
    
    .status-badge-pothole {
        color: #991B1B;
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    
    .status-badge-normal {
        color: #166534;
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    
    .action-recommendation {
        font-size: 1.05rem;
        font-weight: 600;
        margin-top: 0.6rem;
        color: #334155;
    }
    
    .score-highlight {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0F172A;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 10px;
        padding-left: 20px;
        padding-right: 20px;
        font-weight: 600;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper for importing src module
import sys
sys.path.append(os.path.abspath("."))

from src.predict import predict_image, load_pothole_model
from src.severity import analyze_pothole_severity
from src.batch_processor import process_image_batch
from src.gis_map import get_sample_pothole_locations, CITY_PRESETS

MODEL_PATH = "models/pothole_mobilenetv2.keras"
SAMPLES_DIR = Path("samples")

# -------------------------------------------------------------
# TOP NAVIGATION HEADER / BRAND BAR
# -------------------------------------------------------------
st.markdown("""
<div class="top-navbar">
    <div>
        <div class="brand-title">🛣️ PotholeVision</div>
        <div class="brand-subtitle">Municipal Infrastructure Monitoring & Defect Detection System</div>
    </div>
    <div style="display: flex; gap: 15px; align-items: center;">
        <span class="status-pill">🟢 AI Engine Active</span>
        <span style="font-size: 0.85rem; color: #64748B;">v2.4 Production</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------------
st.sidebar.title("⚙️ Inspection Controls")

st.sidebar.subheader("🖼️ Sample Inspector")
sample_choice = st.sidebar.selectbox(
    "Select pre-loaded sample:",
    ["Upload Custom Image", "Sample Pothole Road", "Sample Normal Road"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**System Details:**
- **Model**: MobileNetV2 (Deep Learning)
- **Input Size**: 224 x 224 RGB
- **Accuracy**: Production Model
""")
st.sidebar.caption("© 2026 PotholeVision")

# -------------------------------------------------------------
# DEPLOYMENT TABS
# -------------------------------------------------------------
tab_single, tab_batch, tab_map = st.tabs([
    "🔍 Single Image Diagnostic",
    "📁 Multi-Photo Batch Audit & Export",
    "🗺️ Municipal GIS Incident Map"
])

# -------------------------------------------------------------
# TAB 1: SINGLE IMAGE DIAGNOSTIC
# -------------------------------------------------------------
with tab_single:
    col1, col2 = st.columns([1, 1], gap="large")
    
    image_to_predict = None
    image_source_name = ""
    
    with col1:
        st.subheader("1. Road Surface Photo Input")
        
        if sample_choice == "Sample Pothole Road":
            sample_p_path = SAMPLES_DIR / "sample_pothole.jpg"
            if sample_p_path.exists():
                image_to_predict = Image.open(sample_p_path)
                image_source_name = "Sample Pothole Road"
            else:
                st.error("Sample pothole image file missing.")
        elif sample_choice == "Sample Normal Road":
            sample_n_path = SAMPLES_DIR / "sample_normal.jpg"
            if sample_n_path.exists():
                image_to_predict = Image.open(sample_n_path)
                image_source_name = "Sample Normal Road"
            else:
                st.error("Sample normal image file missing.")
        else:
            uploaded_file = st.file_uploader(
                "Drag & drop or browse road image (JPG, PNG)...",
                type=["jpg", "jpeg", "png"],
                key="single_uploader"
            )
            if uploaded_file is not None:
                try:
                    image_to_predict = Image.open(uploaded_file)
                    image_source_name = uploaded_file.name
                except Exception as e:
                    st.error(f"Error reading image file: {str(e)}")

        if image_to_predict is not None:
            st.image(image_to_predict, caption=f"Selected Photo: {image_source_name}", use_container_width=True)
            
            # File Info Badge
            w, h = image_to_predict.size
            st.caption(f"📏 Resolution: **{w} x {h} px** | Format: **{image_to_predict.format or 'RGB'}**")
        else:
            st.info("👆 Please upload a road photo or select a sample from the sidebar.")

    with col2:
        st.subheader("2. AI Inspection Diagnostic")
        
        if image_to_predict is not None:
            if not Path(MODEL_PATH).exists():
                st.warning(f"⚠️ Model file missing at `{MODEL_PATH}`. Please run `python src/train.py`.")
            else:
                with st.spinner("Analyzing road surface texture & structural integrity..."):
                    try:
                        model = load_pothole_model(MODEL_PATH)
                        pred_res = predict_image(image_to_predict, model=model)
                        
                        label = pred_res["label"]
                        class_name = pred_res["class_name"]
                        confidence = pred_res["confidence"]
                        raw_score = pred_res["raw_score"]
                        
                        sev_res = analyze_pothole_severity(
                            confidence=confidence,
                            pil_img=image_to_predict,
                            class_name=class_name
                        )
                        
                        if class_name == "Pothole":
                            st.markdown(f"""
                            <div class="card-base card-pothole">
                                <div class="status-badge-pothole">🚨 POTHOLE DETECTED</div>
                                <div class="action-recommendation">{sev_res['action_recommended']}</div>
                                <hr style="margin: 1rem 0; border-color: #FCA5A5;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div>
                                        <div style="font-size: 0.85rem; color: #7F1D1D; font-weight: 600;">CONFIDENCE SCORE</div>
                                        <div class="score-highlight" style="color: #DC2626;">{confidence:.1f}%</div>
                                    </div>
                                    <div>
                                        <div style="font-size: 0.85rem; color: #7F1D1D; font-weight: 600;">DEFECT AREA</div>
                                        <div class="score-highlight" style="color: #991B1B;">{sev_res['area_percentage']:.1f}%</div>
                                    </div>
                                    <div>
                                        <div style="font-size: 0.85rem; color: #7F1D1D; font-weight: 600;">HAZARD TIER</div>
                                        <div style="font-size: 1.2rem; font-weight: 800; color: #B91C1C;">{sev_res['severity_level'].upper()}</div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="card-base card-normal">
                                <div class="status-badge-normal">✅ NORMAL ROAD SURFACE</div>
                                <div class="action-recommendation">Clear asphalt road surface. No emergency repair required.</div>
                                <hr style="margin: 1rem 0; border-color: #86EFAC;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div>
                                        <div style="font-size: 0.85rem; color: #14532D; font-weight: 600;">CONFIDENCE SCORE</div>
                                        <div class="score-highlight" style="color: #16A34A;">{confidence:.1f}%</div>
                                    </div>
                                    <div>
                                        <div style="font-size: 0.85rem; color: #14532D; font-weight: 600;">CONDITION</div>
                                        <div style="font-size: 1.2rem; font-weight: 800; color: #15803D;">GOOD CONDITION</div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        # Probability breakdown bar
                        st.markdown("#### 📈 Probability Meter")
                        st.progress(float(raw_score))
                        
                        m1, m2 = st.columns(2)
                        with m1:
                            st.metric("Pothole Risk Index", f"{raw_score * 100:.1f}%")
                        with m2:
                            st.metric("Normal Road Index", f"{(1.0 - raw_score) * 100:.1f}%")

                    except Exception as e:
                        st.error(f"Diagnostic Error: {str(e)}")
        else:
            st.write("Awaiting image upload to generate diagnostic report...")

# -------------------------------------------------------------
# TAB 2: MULTI-PHOTO BATCH AUDIT & CSV
# -------------------------------------------------------------
with tab_batch:
    st.subheader("📁 Batch Road Photo Inspection")
    st.write("Upload multiple road inspection photos to generate automated incident logs and downloadable CSV reports.")
    
    batch_files = st.file_uploader(
        "Upload multiple road images...",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch_uploader"
    )
    
    if batch_files:
        if not Path(MODEL_PATH).exists():
            st.error("Model file missing.")
        else:
            with st.spinner("Executing batch predictions & defect analysis..."):
                model = load_pothole_model(MODEL_PATH)
                df_batch = process_image_batch(batch_files, model=model)
                
            st.success(f"Audit completed for {len(df_batch)} images!")
            
            # Summary KPI Cards
            b_total = len(df_batch)
            b_potholes = len(df_batch[df_batch["Classification"] == "Pothole Detected"])
            b_critical = len(df_batch[df_batch["Severity Rating"] == "Critical Hazard"])
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Total Photos Audited", b_total)
            with c2:
                st.metric("Potholes Flagged", b_potholes)
            with c3:
                st.metric("Critical Hazards", b_critical)
                
            st.markdown("---")
            st.subheader("📋 Incident Master Table")
            st.dataframe(df_batch, use_container_width=True)
            
            # Export CSV
            csv_data = df_batch.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Inspection Report (CSV)",
                data=csv_data,
                file_name="roadguard_pothole_inspection_report.csv",
                mime="text/csv"
            )

# -------------------------------------------------------------
# TAB 3: MUNICIPAL GIS INCIDENT MAP
# -------------------------------------------------------------
with tab_map:
    st.subheader("🗺️ Municipal Road Defect Incident Map")
    st.write("Geographic location tracking of reported road hazards for municipal maintenance crews.")
    
    col_city1, col_city2 = st.columns([1, 1])
    with col_city1:
        selected_city_name = st.selectbox(
            "Select Target City / Region:",
            list(CITY_PRESETS.keys()) + ["Custom GPS Coordinates"],
            key="gis_city_selector"
        )
    
    if selected_city_name == "Custom GPS Coordinates":
        with col_city2:
            c_lat = st.number_input("Latitude", value=19.0760, format="%.4f")
            c_lon = st.number_input("Longitude", value=72.8777, format="%.4f")
    else:
        c_lat, c_lon = CITY_PRESETS[selected_city_name]
        with col_city2:
            st.info(f"📍 Center Coordinates for **{selected_city_name}**: `{c_lat:.4f}, {c_lon:.4f}`")

    map_df = get_sample_pothole_locations(center_lat=c_lat, center_lon=c_lon)
    
    col_map1, col_map2 = st.columns([2, 1])
    with col_map1:
        st.map(map_df, latitude="lat", longitude="lon", size=20)
    with col_map2:
        st.subheader("📍 Incident Registry")
        st.dataframe(map_df[["Incident ID", "Severity", "Confidence (%)"]], use_container_width=True, height=350)

# -------------------------------------------------------------
# FOOTER
# -------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748B; font-size: 0.85rem; font-weight: 500;'>"
    "PotholeVision - Infrastructure Management System &copy; 2026 | Powered by MobileNetV2 Deep Learning & Streamlit"
    "</div>",
    unsafe_allow_html=True

)
