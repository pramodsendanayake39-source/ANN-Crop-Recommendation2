
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

    .stApp {
        background-color: #f7faf7;
    }

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 750;
        color: #1b5e20;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #607d60;
        margin-bottom: 30px;
    }

    .intro-box {
        background: #e8f5e9;
        border-left: 6px solid #43a047;
        padding: 18px 22px;
        border-radius: 12px;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 700;
        color: #2e7d32;
        margin-top: 15px;
        margin-bottom: 10px;
    }

    .result-card {
        background: linear-gradient(
            135deg,
            #e8f5e9,
            #f1f8e9
        );

        border: 2px solid #81c784;
        border-radius: 22px;
        padding: 30px;
        text-align: center;
        margin-top: 18px;
    }

    .result-label {
        font-size: 17px;
        color: #607d60;
    }

    .crop-name {
        font-size: 46px;
        font-weight: 800;
        color: #1b5e20;
        text-transform: capitalize;
        margin: 8px 0;
    }

    .confidence {
        font-size: 24px;
        font-weight: 700;
        color: #2e7d32;
    }

    .confidence-text {
        color: #558b2f;
        font-size: 14px;
    }

    .footer {
        margin-top: 35px;
        padding-top: 15px;
        border-top: 1px solid #d7ddd7;
        text-align: center;
        color: #777;
        font-size: 12px;
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
        Artificial Neural Network Based Agricultural
        Decision Support System
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class="intro-box">

    <b>🌱 How does it work?</b><br><br>

    Enter the soil nutrient and climatic conditions of
    the agricultural area below.

    The trained Artificial Neural Network analyzes
    seven environmental parameters and recommends
    the most suitable crop from 22 crop categories.

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


col1, col2, col3 = st.columns(3)


with col1:

    nitrogen = st.number_input(
        "Nitrogen (N)",
        min_value=0.0,
        max_value=140.0,
        value=70.0,
        step=1.0,
        help="Nitrogen value represented in the training dataset."
    )


with col2:

    phosphorus = st.number_input(
        "Phosphorus (P)",
        min_value=5.0,
        max_value=145.0,
        value=50.0,
        step=1.0
    )


with col3:

    potassium = st.number_input(
        "Potassium (K)",
        min_value=5.0,
        max_value=205.0,
        value=50.0,
        step=1.0
    )


st.markdown(
    '<div class="section-title">🌦️ Climatic Conditions</div>',
    unsafe_allow_html=True
)


col4, col5 = st.columns(2)


with col4:

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=8.0,
        max_value=44.0,
        value=25.0,
        step=0.1
    )


with col5:

    humidity = st.number_input(
        "Humidity (%)",
        min_value=14.0,
        max_value=100.0,
        value=70.0,
        step=0.1
    )


st.markdown(
    '<div class="section-title">🌍 Soil & Rainfall Conditions</div>',
    unsafe_allow_html=True
)


col6, col7 = st.columns(2)


with col6:

    ph = st.number_input(
        "Soil pH",
        min_value=3.5,
        max_value=10.0,
        value=6.5,
        step=0.01
    )


with col7:

    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=20.0,
        max_value=300.0,
        value=100.0,
        step=0.1
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


    # Apply the scaler learned from training data
    input_scaled = scaler.transform(input_data)


    # ANN prediction
    probabilities = model.predict(
        input_scaled,
        verbose=0
    )[0]


    # Highest-probability crop
    predicted_index = int(
        np.argmax(probabilities)
    )


    predicted_crop = label_encoder.inverse_transform(
        [predicted_index]
    )[0]


    confidence = float(
        probabilities[predicted_index] * 100
    )


    # Confidence description
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

    st.markdown(
        f"""
        <div class="result-card">

            <div style="font-size:48px;">
                🌱
            </div>

            <div class="result-label">
                ANN Recommended Crop
            </div>

            <div class="crop-name">
                {predicted_crop}
            </div>

            <div class="confidence">
                {confidence:.2f}% Model Confidence
            </div>

            <div class="confidence-text">
                {confidence_description}
            </div>

        </div>
        """,
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


        left, right = st.columns(
            [3, 1]
        )


        with left:

            st.write(
                f"**{rank}. {crop.title()}**"
            )

            st.progress(
                min(
                    probability / 100,
                    1.0
                )
            )


        with right:

            st.metric(
                "Probability",
                f"{probability:.2f}%"
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
                    "Nitrogen",
                    "Phosphorus",
                    "Potassium",
                    "Temperature",
                    "Humidity",
                    "Soil pH",
                    "Rainfall"
                ],

                "Value": [
                    nitrogen,
                    phosphorus,
                    potassium,
                    temperature,
                    humidity,
                    ph,
                    rainfall
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
    This application is an academic Artificial Neural
    Network prototype. The recommendation is based on
    patterns learned from the selected dataset and should
    not replace professional agricultural or site-specific
    field assessment.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

    Artificial Neural Network Crop Recommendation Project

    <br>

    Inputs: N • P • K • Temperature • Humidity • pH • Rainfall

    </div>
    """,
    unsafe_allow_html=True
)
