"""
InvoiceFlow Interactive Streamlit Web Application
Multimodal Document Intelligence Platform for Tractor Loan Quotation Extraction.
"""
import time
import json
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import cv2
import numpy as np
import streamlit as st

# Configure Page
st.set_page_config(
    page_title="InvoiceFlow Document Intelligence",
    page_icon="🚜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Financial & Agri-Fintech Dashboard
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .field-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .field-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0F172A;
    }
    .tag-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-present {
        background-color: #DCFCE7;
        color: #166534;
    }
    .badge-absent {
        background-color: #FEE2E2;
        color: #991B1B;
    }
</style>
""", unsafe_allow_html=True)

def get_pipeline(vlm_api_key: Optional[str] = None, use_local_vlm: bool = True, vlm_primary: bool = True):
    from core.pipeline import InvoiceFlowPipeline
    return InvoiceFlowPipeline(
        languages=["en"],
        vlm_api_key=vlm_api_key,
        use_local_vlm=use_local_vlm,
        vlm_primary=vlm_primary
    )

def draw_visual_detections(image_bgr: np.ndarray, signature_bbox, stamp_bbox) -> np.ndarray:
    """Draws styled bounding boxes on the image in [ymin, xmin, ymax, xmax] coordinates (range 0-1000)."""
    h, w = image_bgr.shape[:2]
    annotated = image_bgr.copy()

    # Draw Stamp Box (Purple/Magenta)
    if stamp_bbox:
        ymin, xmin, ymax, xmax = stamp_bbox
        y1, x1 = int((ymin / 1000.0) * h), int((xmin / 1000.0) * w)
        y2, x2 = int((ymax / 1000.0) * h), int((xmax / 1000.0) * w)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (180, 0, 180), 3)
        cv2.putText(
            annotated, "STAMP", (x1 + 6, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 0, 180), 2, cv2.LINE_AA
        )

    # Draw Signature Box (Deep Blue)
    if signature_bbox:
        ymin, xmin, ymax, xmax = signature_bbox
        y1, x1 = int((ymin / 1000.0) * h), int((xmin / 1000.0) * w)
        y2, x2 = int((ymax / 1000.0) * h), int((xmax / 1000.0) * w)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (220, 100, 0), 3)
        cv2.putText(
            annotated, "SIGNATURE", (x1 + 6, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 100, 0), 2, cv2.LINE_AA
        )

    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

# --- Sidebar Architecture & Engine Settings ---
with st.sidebar:
    st.markdown("### ⚙️ Engine Architecture")
    engine_mode = st.radio(
        "Extraction Mode:",
        options=[
            "🎯 Local Vision-AI Primary (Qwen2.5-VL-3B • Maximum Accuracy across all formats)",
            "⚡ Hybrid Fast Routing (EasyOCR 2s + VLM Safety Net)"
        ],
        index=0,
        help="Local Vision-AI mode runs direct multimodal reasoning on every invoice (100% offline). Hybrid mode runs fast EasyOCR first."
    )
    vlm_primary = ("Vision-AI Primary" in engine_mode)

    st.markdown("---")
    st.markdown("### 🧠 Local Vision AI (Qwen2.5-VL-3B)")
    use_local_vlm = st.toggle(
        "Enable Local Qwen2.5-VL-3B (RTX 3050 GPU)",
        value=True,
        help="Runs 100% locally on your RTX 3050 GPU (6GB VRAM) for tough handwriting, checkboxes, and multi-model tables."
    )
    if use_local_vlm:
        st.success("🟢 Local Qwen2.5-VL-3B Active (RTX 3050)")
    else:
        st.info("ℹ️ Running in Fast Rule Mode only")

    st.markdown("---")
    st.markdown("### 🔐 Optional Cloud VLM (Gemini)")
    user_vlm_key = st.text_input(
        "Google Gemini API Key:",
        type="password",
        help="Optional. If provided, edge cases can also route to Gemini 1.5 Flash."
    )
    if user_vlm_key:
        st.success("🟢 Cloud Gemini Safety Net Active")

# --- Header Section ---
st.markdown('<div class="main-header">🚜 InvoiceFlow Document Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated 6-Field Tractor Loan Quotation & Invoice Intelligence Engine • Powered by NVIDIA RTX 3050 GPU</div>', unsafe_allow_html=True)

# --- Prominent Input Controls Right On The Main Screen ---
with st.container():
    input_tab1, input_tab2 = st.tabs(["📤 Upload Your Own Invoice / Quotation", "📋 Pick From Dataset Samples (495 documents)"])
    
    target_file_path = None

    with input_tab1:
        uploaded_file = st.file_uploader(
            "Upload an invoice image (PNG, JPG) or document PDF:",
            type=["png", "jpg", "jpeg", "pdf"],
            help="Drag and drop or browse files from your computer."
        )
        if uploaded_file is not None:
            temp_dir = Path(tempfile.gettempdir()) / "invoiceflow_uploads"
            temp_dir.mkdir(exist_ok=True)
            temp_file = temp_dir / uploaded_file.name
            temp_file.write_bytes(uploaded_file.read())
            target_file_path = temp_file

    with input_tab2:
        raw_dir = Path("datasets/raw")
        available_files = sorted([p.name for p in raw_dir.glob("*.png")]) if raw_dir.is_dir() else []
        
        default_samples = [
            "172427893_3_pg11.png",
            "173892975_1_pg13.png",
            "90019495644_OTHERS_v1_pg1.png",
            "90019035429_1_v1_pg2.png",
            "90018535123_OTHERS_v1_pg1.png"
        ]
        options = default_samples + [f for f in available_files if f not in default_samples]
        
        selected_sample = st.selectbox(
            "Select an invoice file from the dataset:",
            options=options,
            index=0
        )
        if selected_sample and target_file_path is None:
            sample_path = raw_dir / selected_sample
            if sample_path.is_file():
                target_file_path = sample_path

# Action Buttons Bar
act_col1, act_col2 = st.columns([1, 4])
with act_col1:
    extract_btn = st.button("🚀 Extract 6 Target Fields", type="primary", use_container_width=True)
with act_col2:
    if target_file_path:
        st.caption(f"Active file: `{target_file_path.name}` ({target_file_path.stat().st_size / 1024:.1f} KB)")

st.markdown("---")

# --- Document Display and Results ---
if target_file_path and target_file_path.is_file():
    img_bgr = cv2.imread(str(target_file_path))
    if img_bgr is None:
        st.error(f"Failed to load image from: {target_file_path}")
        st.stop()

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns([1.1, 1.0], gap="large")

    with col1:
        st.subheader("📄 Document Canvas")
        image_placeholder = st.empty()
        image_placeholder.image(img_rgb, caption=f"Loaded Document: {target_file_path.name}", use_container_width=True)

    with col2:
        st.subheader("🎯 Extracted Intelligence")

        if extract_btn:
            with st.spinner("Extracting fields with RTX 3050 GPU (Dual-Tier Pipeline)..."):
                pipeline = get_pipeline(
                    vlm_api_key=user_vlm_key if user_vlm_key else None,
                    use_local_vlm=use_local_vlm,
                    vlm_primary=vlm_primary
                )
                start_t = time.time()
                try:
                    res = pipeline.process_document(target_file_path)
                except Exception as e:
                    st.error(f"❌ Extraction error: {e}")
                    st.stop()
                elapsed = round(time.time() - start_t, 2)

            # Draw visual boxes on document canvas
            sig_box = res.dealer_signature.bounding_box if res.dealer_signature.present else None
            stamp_box = res.dealer_stamp.bounding_box if res.dealer_stamp.present else None
            annotated_rgb = draw_visual_detections(img_bgr, sig_box, stamp_box)
            image_placeholder.image(
                annotated_rgb,
                caption=f"Detections: Stamp (Purple) | Signature (Blue)",
                use_container_width=True
            )

            # Pipeline Tier Indicator Badge
            if res.metadata.fallback_triggered:
                st.markdown(f"""
                <div style="background-color: #F3E8FF; border: 1px solid #D8B4FE; border-radius: 8px; padding: 10px 14px; margin-bottom: 12px; color: #6B21A8; font-weight: 600;">
                    🟣 Multimodal VLM Safety Net Engaged ({res.metadata.ocr_engine})
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background-color: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 8px; padding: 10px 14px; margin-bottom: 12px; color: #065F46; font-weight: 600;">
                    🟢 Fast Local Dynamic Engine (100% Offline Air-Gapped • GPU-Accelerated)
                </div>
                """, unsafe_allow_html=True)

            # Performance Pill
            st.success(f"⏱️ Extracted in **{elapsed}s** using **{res.metadata.ocr_engine}**")

            # 1. Dealer Name Card
            st.markdown(f"""
            <div class="metric-card">
                <div class="field-title">1. Legal Dealership Name</div>
                <div class="field-value">{res.dealer_name}</div>
            </div>
            """, unsafe_allow_html=True)

            # 2. Model Name Card
            st.markdown(f"""
            <div class="metric-card">
                <div class="field-title">2. Tractor Model Designation</div>
                <div class="field-value">🚜 {res.model_name}</div>
            </div>
            """, unsafe_allow_html=True)

            # 3. Horse Power & 4. Asset Cost (2 columns)
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="field-title">3. Engine Horsepower</div>
                    <div class="field-value">{res.horse_power:.1f} HP</div>
                </div>
                """, unsafe_allow_html=True)

            with m_col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="field-title">4. Total Asset Cost</div>
                    <div class="field-value">₹ {res.asset_cost:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)

            # 5. Signature & 6. Stamp Cards (2 columns)
            v_col1, v_col2 = st.columns(2)
            with v_col1:
                sig_status = "PRESENT" if res.dealer_signature.present else "ABSENT"
                sig_badge = "badge-present" if res.dealer_signature.present else "badge-absent"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="field-title">5. Authorized Signature</div>
                    <div><span class="tag-badge {sig_badge}">{sig_status}</span> <b>{int(res.dealer_signature.confidence*100)}%</b></div>
                    <small style="color:#64748B;">Box: {res.dealer_signature.bounding_box}</small>
                </div>
                """, unsafe_allow_html=True)

            with v_col2:
                stamp_status = "PRESENT" if res.dealer_stamp.present else "ABSENT"
                stamp_badge = "badge-present" if res.dealer_stamp.present else "badge-absent"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="field-title">6. Dealership Stamp</div>
                    <div><span class="tag-badge {stamp_badge}">{stamp_status}</span> <b>{int(res.dealer_stamp.confidence*100)}%</b></div>
                    <small style="color:#64748B;">Box: {res.dealer_stamp.bounding_box}</small>
                </div>
                """, unsafe_allow_html=True)

            # Full JSON Output Expander
            with st.expander("🔍 View Complete Verified Pydantic JSON", expanded=False):
                json_str = json.dumps(res.model_dump(), indent=2)
                st.code(json_str, language="json")
                st.download_button(
                    label="📥 Download JSON Result",
                    data=json_str,
                    file_name=f"{target_file_path.stem}_extraction.json",
                    mime="application/json",
                    use_container_width=True
                )
        else:
            st.info("👆 Click **'🚀 Extract 6 Target Fields'** to run GPU inference on this document.")
else:
    st.info("Please upload an invoice or select a sample from the tabs above.")
