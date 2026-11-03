import json
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

from app.inference import CropShieldInference
from app.report import build_report
from app.utils import (
    format_class_name,
    format_confidence,
)


# ============================================================
# HISTORY STORAGE
# ============================================================

HISTORY_FILE = PROJECT_ROOT / "data" / "history.json"
MAX_HISTORY_ENTRIES = 50


def load_history():
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []
    return []


def save_history(history):
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def add_history_entry(filename, result):
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": filename,
        "status": result["status"],
    }

    if result["status"] != "not_leaf":
        entry["predicted_class"] = result.get("predicted_class")
        entry["confidence"] = result.get("confidence")

    entry["leaf_probability"] = result.get("leaf_probability")

    st.session_state.history.insert(0, entry)
    st.session_state.history = st.session_state.history[:MAX_HISTORY_ENTRIES]
    save_history(st.session_state.history)


if "history" not in st.session_state:
    st.session_state.history = load_history()


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CropShield",
    page_icon="🌱",
    layout="centered",
)


# ============================================================
# THEME / CUSTOM CSS
# ============================================================

SEVERITY_COLORS = {
    "none": "#2e7d32",
    "low": "#9e9d24",
    "moderate": "#ef6c00",
    "high": "#d84315",
    "severe": "#b71c1c",
    "unknown": "#616161",
}

st.markdown(
    """
    <style>
        :root {
            --cs-green: #2e7d32;
            --cs-green-light: #e8f5e9;
            --cs-cream: #faf9f4;
        }

        .stApp {
            background-color: var(--cs-cream);
        }

        section[data-testid="stSidebar"] {
            background-color: #f1f4ec;
        }

        h1, h2, h3 {
            font-family: 'Segoe UI', sans-serif;
            color: #1b3a1f;
        }

        .cs-hero {
            background: linear-gradient(135deg, #e8f5e9 0%, #f4f7ef 100%);
            border-radius: 16px;
            padding: 28px 32px;
            margin-bottom: 24px;
            border: 1px solid #d7e8d9;
        }

        .cs-hero h1 {
            margin: 0 0 6px 0;
            font-size: 2.1rem;
        }

        .cs-hero p {
            color: #3f5a44;
            margin: 0;
            font-size: 1rem;
        }

        .cs-card {
            background: white;
            border-radius: 14px;
            padding: 22px 24px;
            margin-bottom: 18px;
            border: 1px solid #e4e4e4;
            box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        }

        .cs-badge {
            display: inline-block;
            padding: 4px 14px;
            border-radius: 999px;
            color: white;
            font-weight: 600;
            font-size: 0.8rem;
            letter-spacing: 0.02em;
            text-transform: uppercase;
        }

        .stButton > button {
            border-radius: 10px;
            font-weight: 600;
        }

        div[data-testid="stFileUploaderDropzone"] {
            border-radius: 12px;
        }

        .cs-history-item {
            border-radius: 10px;
            padding: 10px 12px;
            margin-bottom: 8px;
            background: white;
            border: 1px solid #e0e0e0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR — HISTORY
# ============================================================

with st.sidebar:
    st.markdown("### 📜 Prediction History")

    if not st.session_state.history:
        st.caption("No predictions yet. Analyze a leaf to get started.")
    else:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            save_history([])
            st.rerun()

        st.divider()

        for entry in st.session_state.history:
            if entry["status"] == "not_leaf":
                label = "🌿 No leaf detected"
                sub = ""
            else:
                label = f"**{format_class_name(entry.get('predicted_class', 'Unknown'))}**"
                sub = (
                    f"Confidence: {format_confidence(entry['confidence'])}"
                    if entry.get("confidence") is not None
                    else ""
                )

            st.markdown(
                f"""
                <div class="cs-history-item">
                    <div style="font-size: 0.75rem; color: #888;">{entry["timestamp"]}</div>
                    <div style="margin-top: 2px;">{label}</div>
                    <div style="font-size: 0.8rem; color: #555;">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="cs-hero">
        <h1>🌱 CropShield</h1>
        <p>AI-Powered Plant Disease Detection — upload a clear leaf photo
        and CropShield verifies it's a leaf before diagnosing disease.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_crop_shield():

    return CropShieldInference()


try:

    crop_shield = load_crop_shield()

except Exception as error:

    st.error(
        "CropShield models could not be loaded."
    )

    st.exception(error)

    st.stop()


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a plant leaf image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
    ],
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded Image",
        use_container_width=True,
    )

    analyze_button = st.button(
        "🔍 Analyze Image",
        type="primary",
        use_container_width=True,
    )

    if analyze_button:

        with st.spinner(
            "Analyzing image..."
        ):

            result = crop_shield.predict(
                uploaded_file
            )

        add_history_entry(uploaded_file.name, result)

        # ----------------------------------------------------
        # NOT A LEAF
        # ----------------------------------------------------

        if result["status"] == "not_leaf":

            st.markdown('<div class="cs-card">', unsafe_allow_html=True)

            st.warning(
                "🌿 No suitable plant leaf detected."
            )

            st.write(
                result["message"]
            )

            st.metric(
                "Leaf Probability",
                format_confidence(
                    result["leaf_probability"]
                ),
            )

            st.markdown('</div>', unsafe_allow_html=True)

        # ----------------------------------------------------
        # LOW CONFIDENCE
        # ----------------------------------------------------

        elif result["status"] == "low_confidence":

            st.markdown('<div class="cs-card">', unsafe_allow_html=True)

            st.warning(
                "⚠️ Prediction confidence is low."
            )

            st.write(
                result["message"]
            )

            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Leaf Probability",
                    format_confidence(
                        result["leaf_probability"]
                    ),
                )
            with col2:
                st.metric(
                    "Confidence",
                    format_confidence(
                        result["confidence"]
                    ),
                )

            st.write(
                "**Possible Disease:**",
                format_class_name(
                    result["predicted_class"]
                ),
            )

            st.markdown('</div>', unsafe_allow_html=True)

            pdf_buffer = build_report(uploaded_file, result)
            st.download_button(
                "📄 Download PDF Report",
                data=pdf_buffer,
                file_name="cropshield_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        # ----------------------------------------------------
        # CONFIDENT DIAGNOSIS
        # ----------------------------------------------------

        else:

            st.markdown('<div class="cs-card">', unsafe_allow_html=True)

            st.success(
                "✅ Leaf detected successfully."
            )

            st.subheader(
                "Disease Prediction"
            )

            col1, col2 = st.columns([2, 1])
            with col1:
                st.write("**Predicted Class:**")
                st.write(
                    format_class_name(
                        result["predicted_class"]
                    )
                )
            with col2:
                st.metric(
                    "Confidence",
                    format_confidence(
                        result["confidence"]
                    ),
                )

            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="cs-card">', unsafe_allow_html=True)

            st.subheader(
                "Top 3 Predictions"
            )

            for prediction in result["top_3"]:

                class_name = format_class_name(
                    prediction["class_name"]
                )

                confidence = format_confidence(
                    prediction["confidence"]
                )

                st.write(
                    f"**{class_name}** — "
                    f"{confidence}"
                )

                st.progress(
                    prediction["confidence"]
                )

            st.markdown('</div>', unsafe_allow_html=True)

            # ----------------------------------------------------
            # TREATMENT INFO
            # ----------------------------------------------------
            treatment_info = result.get("treatment_info")
            if treatment_info:
                st.markdown('<div class="cs-card">', unsafe_allow_html=True)

                st.subheader("Treatment & Prevention")

                severity = treatment_info["severity"]
                color = SEVERITY_COLORS.get(severity, SEVERITY_COLORS["unknown"])
                st.markdown(
                    f'<span class="cs-badge" style="background-color: {color};">'
                    f'{severity}</span>',
                    unsafe_allow_html=True,
                )

                st.write("")
                st.write("**Recommended treatment:**")
                for tip in treatment_info["treatment"]:
                    st.write(f"- {tip}")

                if treatment_info["prevention"]:
                    st.write("**Prevention tips:**")
                    for tip in treatment_info["prevention"]:
                        st.write(f"- {tip}")

                st.markdown('</div>', unsafe_allow_html=True)

            pdf_buffer = build_report(uploaded_file, result)
            st.download_button(
                "📄 Download PDF Report",
                data=pdf_buffer,
                file_name="cropshield_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )