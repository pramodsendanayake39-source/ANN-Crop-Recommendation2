import streamlit as st
import tensorflow as tf
import joblib
import pandas as pd
import numpy as np


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Crop Recommendation",
    page_icon="🌾",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* --------------------------------------------------------
       MAIN APPLICATION
    -------------------------------------------------------- */

    .stApp {
        background-color: #ffffff;
        color: #1f2937;
    }

    /* Make normal Streamlit text readable */
    .stMarkdown,
    .stMarkdown p,
    p,
    label {
        color: #1f2937 !important;
    }

    /* Widget labels */
    [data-testid="stWidgetLabel"] p {
        color: #1f2937 !important;
        font-weight: 700 !important;
        font-size: 15px !important;
    }

    /* Help / caption text */
    [data-testid="stCaptionContainer"] {
        color: #4b5563 !important;
    }

    [data-testid="stCaptionContainer"] p {
        color: #4b5563 !important;
    }

    /* --------------------------------------------------------
       NUMBER INPUTS
    -------------------------------------------------------- */

    [data-testid="stNumberInput"] input {
        color: #111827 !important;
        background-color: #ffffff !important;
        font-weight: 600 !important;
    }

    [data-testid="stNumberInput"] button {
        color: #ffffff !important;
    }


    /* --------------------------------------------------------
       HEADER
    -------------------------------------------------------- */

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        color: #1b5e20 !important;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #4b6350 !important;
        margin-bottom: 30px;
    }


    /* --------------------------------------------------------
       INTRODUCTION BOX
    -------------------------------------------------------- */

    .intro-box {
        background: #e8f5e9;
        color: #1f2937 !important;
        border-left: 6px solid #43a047;
        padding: 20px 24px;
        border-radius: 12px;
        margin-bottom: 25px;
        line-height: 1.7;
    }

    .intro-box b {
        color: #1b5e20 !important;
    }


    /* --------------------------------------------------------
       SECTION TITLES
    -------------------------------------------------------- */

    .section-title {
        font-size: 22px;
        font-weight: 750;
        color: #2e7d32 !important;
        margin-top: 20px;
        margin-bottom: 10px;
    }


    /* --------------------------------------------------------
       PREDICTION BUTTON
    -------------------------------------------------------- */

    button[kind="primary"] {
        background-color: #2e7d32 !important;
        border-color: #2e7d32 !important;
        color: #ffffff !important;
        font-weight: 750 !important;
        min-height: 50px !important;
        font-size: 16px !important;
    }

    button[kind="primary"]:hover {
        background-color: #1b5e20 !important;
        border-color: #1b5e20 !important;
        color: #ffffff !important;
    }


    /* --------------------------------------------------------
       RESULT CARD
    -------------------------------------------------------- */

    .result-card {
        background: linear-gradient(
            135deg,
            #e8f5e9,
            #f1f8e9
        );

        border: 2px solid #81c784;
        border-radius: 22px;
        padding: 32px;
        text-align: center;
        margin-top: 18px;
        margin-bottom: 20px;
    }

    .result-icon {
        font-size: 50px;
        margin-bottom: 5px;
    }

    .result-label {
        font-size: 17px;
        color: #455a4a !important;
        font-weight: 650;
    }

    .crop-name {
        font-size: 48px;
        font-weight: 850;
        color: #1b5e20 !important;
        text-transform: capitalize;
        margin: 8px 0;
    }

    .confidence {
        font-size: 25px;
        font-weight: 750;
        color: #2e7d32 !important;
    }

    .confidence-text {
        color: #558b2f !important;
        font-size: 14px;
        font-weight: 650;
        margin-top: 4px;
    }

    .result-note {
        color: #4b5563 !important;
        font-size: 14px;
        margin-top: 18px;
    }


    /* --------------------------------------------------------
       TOP 3 PREDICTIONS
    -------------------------------------------------------- */

    .prediction-row {
        display: flex;
        justify-content: space-between;
        align-items: center;

        padding: 10px 5px 5px 5px;

        font-size: 17px;
    }

    .prediction-name {
        color: #1f2937 !important;
        font-weight: 750;
    }

    .prediction-percent {
        color: #1b5e20 !important;
        font-weight: 850;
    }

    /* Progress bar */
    [data-testid="stProgress"] > div > div {
        background-color: #43a047 !important;
    }


    /* --------------------------------------------------------
       INPUT SUMMARY TABLE
    -------------------------------------------------------- */

    [data-testid="stDataFrame"] {
        color: #1f2937 !important;
    }


    /* --------------------------------------------------------
       FOOTER
    -------------------------------------------------------- */

    .footer {
        margin-top: 35px;
        padding-top: 15px;
        border-top: 1px solid #d7ddd7;
        text-align: center;
        color: #555555 !important;
        font-size: 12px;
        line-height: 1.7;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD TRAINED MODEL AND PREPROCESSING OBJECTS
# ============================================================

@st.cache_resource
def load_system():

    model = tf.keras.models.load_model(
        "final_crop_recommendation_ann.keras"
    )

    scaler = joblib.load(
        "crop_scaler.pkl"
    )

    label_encoder = joblib.load(
        "crop_label_encoder.pkl"
    )

    return model, scaler, label_encoder


model, scaler, label_encoder = load_system()


# ============================================================
# PAGE HEADER
# ============================================================

st.markdown(
    """
<div class="main-title">
🌾 Smart Crop Recommendation System
</div>

<div class="subtitle">
Artificial Neural Network Based Agricultural Decision Support System
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# INTRODUCTION
# ============================================================

st.markdown(
    """
<div class="intro-box">
<b>🌱 How does it work?</b><br><br>

Enter the soil nutrient and climatic conditions of the agricultural area below.<br><br>

The trained Artificial Neural Network analyses seven environmental
parameters and recommends the most suitable crop from 22 crop categories.
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    '<div class="section-title">🧪 Soil Nutrient Parameters</div>',
    unsafe_allow_html=True
)

st.caption(
    "Enter soil nutrient values within the ranges represented "
    "in the ANN training dataset."
)


# ------------------------------------------------------------
# N, P, K
# ------------------------------------------------------------

col1, col2, col3 = st.columns(3)


with col1:

    nitrogen = st.number_input(
        "Nitrogen (N) — 0–140 kg/ha",
        min_value=0.0,
        max_value=140.0,
        value=70.0,
        step=1.0,
        help="Training dataset range: 0–140 kg/ha"
    )


with col2:

    phosphorus = st.number_input(
        "Phosphorus (P) — 5–145 kg/ha",
        min_value=5.0,
        max_value=145.0,
        value=50.0,
        step=1.0,
        help="Training dataset range: 5–145 kg/ha"
    )


with col3:

    potassium = st.number_input(
        "Potassium (K) — 5–205 kg/ha",
        min_value=5.0,
        max_value=205.0,
        value=50.0,
        step=1.0,
        help="Training dataset range: 5–205 kg/ha"
    )


# ------------------------------------------------------------
# CLIMATE
# ------------------------------------------------------------

st.markdown(
    '<div class="section-title">🌦️ Climatic Conditions</div>',
    unsafe_allow_html=True
)


col4, col5 = st.columns(2)


with col4:

    temperature = st.number_input(
        "Temperature — 8.83–43.68 °C",
        min_value=8.83,
        max_value=43.68,
        value=25.0,
        step=0.1,
        format="%.2f",
        help="Training dataset range: 8.83–43.68 °C"
    )


with col5:

    humidity = st.number_input(
        "Relative Humidity — 14.26–99.98 %",
        min_value=14.26,
        max_value=99.98,
        value=70.0,
        step=0.1,
        format="%.2f",
        help="Training dataset range: 14.26–99.98%"
    )


# ------------------------------------------------------------
# SOIL pH + RAINFALL
# ------------------------------------------------------------

st.markdown(
    '<div class="section-title">🌍 Soil & Rainfall Conditions</div>',
    unsafe_allow_html=True
)


col6, col7 = st.columns(2)


with col6:

    ph = st.number_input(
        "Soil pH — 3.50–9.94",
        min_value=3.50,
        max_value=9.94,
        value=6.50,
        step=0.01,
        format="%.2f",
        help="pH is unitless. Training dataset range: 3.50–9.94"
    )


with col7:

    rainfall = st.number_input(
        "Rainfall — 20.21–298.56 mm",
        min_value=20.21,
        max_value=298.56,
        value=100.0,
        step=0.1,
        format="%.2f",
        help="Training dataset range: 20.21–298.56 mm"
    )


# ============================================================
# INPUT GUIDANCE
# ============================================================

st.info(
    """
    ℹ️ The displayed ranges correspond to the minimum and
    maximum values represented in the ANN training dataset.
    Predictions outside these ranges are not supported by
    this application.
    """
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.write("")

predict = st.button(
    "🌾 Analyse Conditions & Recommend Crop",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict:

    # --------------------------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        {
            "N": [nitrogen],
            "P": [phosphorus],
            "K": [potassium],
            "temperature": [temperature],
            "humidity": [humidity],
            "ph": [ph],
            "rainfall": [rainfall]
        }
    )


    # --------------------------------------------------------
    # SCALE INPUT USING TRAINING SCALER
    # --------------------------------------------------------

    input_scaled = scaler.transform(
        input_data
    )


    # --------------------------------------------------------
    # ANN PREDICTION
    # --------------------------------------------------------

    probabilities = model.predict(
        input_scaled,
        verbose=0
    )[0]


    # --------------------------------------------------------
    # HIGHEST-PROBABILITY CLASS
    # --------------------------------------------------------

    predicted_index = int(
        np.argmax(probabilities)
    )


    predicted_crop = label_encoder.inverse_transform(
        [predicted_index]
    )[0]


    confidence = float(
        probabilities[predicted_index] * 100
    )


    # --------------------------------------------------------
    # CONFIDENCE DESCRIPTION
    # --------------------------------------------------------

    if confidence >= 90:

        confidence_description = (
            "Very High Model Confidence"
        )

    elif confidence >= 70:

        confidence_description = (
            "High Model Confidence"
        )

    elif confidence >= 50:

        confidence_description = (
            "Moderate Model Confidence"
        )

    else:

        confidence_description = (
            "Low Model Confidence"
        )


    # ========================================================
    # PRIMARY RESULT
    # ========================================================

    # IMPORTANT:
    # HTML begins immediately after the triple quotes.
    # This prevents Streamlit Markdown from displaying it
    # as a code block.

    result_html = f"""<div class="result-card">
<div class="result-icon">🌱</div>

<div class="result-label">
ANN Recommended Crop
</div>

<div class="crop-name">
{predicted_crop.title()}
</div>

<div class="confidence">
{confidence:.2f}% Model Confidence
</div>

<div class="confidence-text">
{confidence_description}
</div>

<div class="result-note">
Recommendation generated using the soil nutrient and
climatic conditions entered by the user.
</div>
</div>"""


    st.markdown(
        result_html,
        unsafe_allow_html=True
    )


    # ========================================================
    # TOP 3 RESULTS
    # ========================================================

    st.markdown(
        '<div class="section-title">📊 Top 3 ANN Predictions</div>',
        unsafe_allow_html=True
    )


    top3_indices = np.argsort(
        probabilities
    )[-3:][::-1]


    for rank, index in enumerate(
        top3_indices,
        start=1
    ):

        crop = label_encoder.inverse_transform(
            [int(index)]
        )[0]


        probability = float(
            probabilities[index] * 100
        )


        # Custom readable row instead of st.metric()
        prediction_html = f"""<div class="prediction-row">
<div class="prediction-name">
{rank}. {crop.title()}
</div>

<div class="prediction-percent">
{probability:.2f}%
</div>
</div>"""


        st.markdown(
            prediction_html,
            unsafe_allow_html=True
        )


        st.progress(
            min(
                probability / 100,
                1.0
            )
        )


    # ========================================================
    # INPUT SUMMARY
    # ========================================================

    with st.expander(
        "📋 View submitted environmental conditions"
    ):

        summary = pd.DataFrame(
            {
                "Parameter": [
                    "Nitrogen (N)",
                    "Phosphorus (P)",
                    "Potassium (K)",
                    "Temperature",
                    "Relative Humidity",
                    "Soil pH",
                    "Rainfall"
                ],

                "Value": [
                    f"{nitrogen:.2f} kg/ha",
                    f"{phosphorus:.2f} kg/ha",
                    f"{potassium:.2f} kg/ha",
                    f"{temperature:.2f} °C",
                    f"{humidity:.2f} %",
                    f"{ph:.2f}",
                    f"{rainfall:.2f} mm"
                ]
            }
        )


        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# DISCLAIMER
# ============================================================

st.info(
    """
    ⚠️ This application is an academic Artificial Neural
    Network prototype. The recommendation is based on patterns
    learned from the selected crop recommendation dataset.

    The model output should not replace professional
    agricultural advice, soil laboratory testing, or
    site-specific field assessment.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="footer">

<b>Artificial Neural Network Crop Recommendation Project</b>

<br><br>

Model Inputs:
N • P • K • Temperature • Humidity • pH • Rainfall

<br>

Output:
Recommended Crop from 22 Crop Classes

</div>
""",
    unsafe_allow_html=True
)
