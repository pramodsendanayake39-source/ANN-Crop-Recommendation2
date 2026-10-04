import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf


st.set_page_config(
    page_title="Smart Crop Recommendation System",
    page_icon="🌾",
    layout="centered"
)


MODEL_FILE = "final_crop_recommendation_ann.keras"
SCALER_FILE = "crop_scaler.pkl"
ENCODER_FILE = "crop_label_encoder.pkl"


@st.cache_resource
def load_system():

    model = tf.keras.models.load_model(
        MODEL_FILE,
        compile=False
    )

    scaler = joblib.load(
        SCALER_FILE
    )

    encoder = joblib.load(
        ENCODER_FILE
    )

    return model, scaler, encoder


model, scaler, label_encoder = load_system()


# ============================================================
# HEADER
# ============================================================

st.title("🌾 Smart Crop Recommendation System")

st.write(
    "Artificial Neural Network Based Agricultural Decision Support System"
)

st.caption(
    "UI BUILD 4.0 — NEW DEPLOYMENT"
)


st.success(
    """
    🌱 Enter the soil nutrient and climatic conditions below.

    The ANN will recommend the most suitable crop from
    22 crop classes.
    """
)


# ============================================================
# INPUT FORM
# ============================================================

with st.form("crop_form"):

    st.header("📋 Environmental Conditions")

    st.subheader("🧪 Soil Nutrients")

    col1, col2, col3 = st.columns(3)

    with col1:
        nitrogen = st.number_input(
            "Nitrogen (N)",
            min_value=0.0,
            max_value=140.0,
            value=70.0,
            step=1.0
        )
        st.write("Range: **0–140**")

    with col2:
        phosphorus = st.number_input(
            "Phosphorus (P)",
            min_value=5.0,
            max_value=145.0,
            value=50.0,
            step=1.0
        )
        st.write("Range: **5–145**")

    with col3:
        potassium = st.number_input(
            "Potassium (K)",
            min_value=5.0,
            max_value=205.0,
            value=50.0,
            step=1.0
        )
        st.write("Range: **5–205**")


    st.divider()

    st.subheader("🌦️ Climatic Conditions")

    col4, col5 = st.columns(2)

    with col4:
        temperature = st.number_input(
            "Temperature (°C)",
            min_value=8.83,
            max_value=43.68,
            value=25.00,
            step=0.10
        )
        st.write("Range: **8.83–43.68 °C**")

    with col5:
        humidity = st.number_input(
            "Relative Humidity (%)",
            min_value=14.26,
            max_value=99.98,
            value=70.00,
            step=0.10
        )
        st.write("Range: **14.26–99.98 %**")


    st.divider()

    st.subheader("🌍 Soil & Rainfall")

    col6, col7 = st.columns(2)

    with col6:
        ph = st.number_input(
            "Soil pH",
            min_value=3.50,
            max_value=9.94,
            value=6.50,
            step=0.01
        )
        st.write("Range: **3.50–9.94**")

    with col7:
        rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=20.21,
            max_value=298.56,
            value=100.00,
            step=0.10
        )
        st.write("Range: **20.21–298.56 mm**")


    submitted = st.form_submit_button(
        "🌾 Analyse Conditions & Recommend Crop",
        type="primary",
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

if submitted:

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


    input_scaled = scaler.transform(
        input_data
    )


    probabilities = model.predict(
        input_scaled,
        verbose=0
    )[0]


    predicted_index = int(
        np.argmax(probabilities)
    )


    predicted_crop = label_encoder.inverse_transform(
        [predicted_index]
    )[0]


    confidence = float(
        probabilities[predicted_index] * 100
    )


    # ========================================================
    # MAIN RESULT
    # ========================================================

    st.divider()

    st.header("🎯 Recommendation")

    col_result1, col_result2 = st.columns(
        [2, 1]
    )

    with col_result1:

        st.subheader("🌱 Recommended Crop")

        st.title(
            predicted_crop.title()
        )


    with col_result2:

        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )


    if confidence >= 90:
        st.success("Very High Model Confidence")

    elif confidence >= 70:
        st.success("High Model Confidence")

    elif confidence >= 50:
        st.warning("Moderate Model Confidence")

    else:
        st.warning("Low Model Confidence")


    # ========================================================
    # TOP 3
    # ========================================================

    st.subheader("📊 Top 3 Predictions")


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
            probabilities[index]
        )

        percentage = probability * 100


        left, right = st.columns(
            [4, 1]
        )


        with left:
            st.write(
                f"**{rank}. {crop.title()}**"
            )

        with right:
            st.write(
                f"**{percentage:.2f}%**"
            )


        st.progress(
            probability
        )


    # ========================================================
    # INPUT SUMMARY
    # ========================================================

    st.subheader("📋 Input Summary")


    summary = pd.DataFrame(
        {
            "Parameter": [
                "Nitrogen",
                "Phosphorus",
                "Potassium",
                "Temperature",
                "Humidity",
                "pH",
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
            ],

            "Unit": [
                "Dataset nutrient units",
                "Dataset nutrient units",
                "Dataset nutrient units",
                "°C",
                "%",
                "Unitless",
                "mm"
            ]
        }
    )


    st.dataframe(
        summary,
        hide_index=True,
        use_container_width=True
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.warning(
    """
    ⚠️ Academic prototype.

    This model is intended for educational purposes and should
    not replace professional agricultural or site-specific
    advice.
    """
)
