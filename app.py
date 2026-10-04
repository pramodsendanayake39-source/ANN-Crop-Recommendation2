import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Crop Recommendation System",
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# ============================================================
# MODEL FILES
# ============================================================

MODEL_FILE = "final_crop_recommendation_ann.keras"
SCALER_FILE = "crop_scaler.pkl"
ENCODER_FILE = "crop_label_encoder.pkl"


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

required_files = [
    MODEL_FILE,
    SCALER_FILE,
    ENCODER_FILE
]

missing_files = [
    file for file in required_files
    if not os.path.exists(file)
]

if missing_files:
    st.error(
        "Missing required files: "
        + ", ".join(missing_files)
    )
    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_system():

    model = tf.keras.models.load_model(
        MODEL_FILE,
        compile=False
    )

    scaler = joblib.load(
        SCALER_FILE
    )

    label_encoder = joblib.load(
        ENCODER_FILE
    )

    return model, scaler, label_encoder


model, scaler, label_encoder = load_system()


# ============================================================
# HEADER
# ============================================================

st.title("🌾 Smart Crop Recommendation System")

st.markdown(
    "**Artificial Neural Network Based Agricultural Decision Support System**"
)

st.caption("UI BUILD 4.0 — Clean Native Streamlit Version")


# ============================================================
# INTRODUCTION
# ============================================================

st.success(
    """
    🌱 **How does it work?**

    Enter the soil nutrient and climatic conditions below.

    The trained Artificial Neural Network analyses seven
    environmental parameters and recommends the most suitable
    crop from 22 crop classes.
    """
)


# ============================================================
# INPUT FORM
# ============================================================

st.header("📋 Enter Environmental Conditions")


with st.form("crop_input_form"):

    # ========================================================
    # SOIL NUTRIENTS
    # ========================================================

    st.subheader("🧪 Soil Nutrient Parameters")

    col1, col2, col3 = st.columns(3)


    with col1:

        nitrogen = st.number_input(
            "Nitrogen (N)",
            min_value=0.0,
            max_value=140.0,
            value=70.0,
            step=1.0,
            help="Training range: 0–140"
        )

        st.caption(
            "**Range:** 0–140 | Dataset nutrient units"
        )


    with col2:

        phosphorus = st.number_input(
            "Phosphorus (P)",
            min_value=5.0,
            max_value=145.0,
            value=50.0,
            step=1.0,
            help="Training range: 5–145"
        )

        st.caption(
            "**Range:** 5–145 | Dataset nutrient units"
        )


    with col3:

        potassium = st.number_input(
            "Potassium (K)",
            min_value=5.0,
            max_value=205.0,
            value=50.0,
            step=1.0,
            help="Training range: 5–205"
        )

        st.caption(
            "**Range:** 5–205 | Dataset nutrient units"
        )


    st.divider()


    # ========================================================
    # CLIMATE
    # ========================================================

    st.subheader("🌦️ Climatic Conditions")

    col4, col5 = st.columns(2)


    with col4:

        temperature = st.number_input(
            "Temperature (°C)",
            min_value=8.83,
            max_value=43.68,
            value=25.00,
            step=0.10,
            format="%.2f"
        )

        st.caption(
            "**Range:** 8.83–43.68 °C"
        )


    with col5:

        humidity = st.number_input(
            "Relative Humidity (%)",
            min_value=14.26,
            max_value=99.98,
            value=70.00,
            step=0.10,
            format="%.2f"
        )

        st.caption(
            "**Range:** 14.26–99.98 %"
        )


    st.divider()


    # ========================================================
    # SOIL + RAINFALL
    # ========================================================

    st.subheader("🌍 Soil & Rainfall Conditions")

    col6, col7 = st.columns(2)


    with col6:

        ph = st.number_input(
            "Soil pH",
            min_value=3.50,
            max_value=9.94,
            value=6.50,
            step=0.01,
            format="%.2f"
        )

        st.caption(
            "**Range:** 3.50–9.94 | Unitless"
        )


    with col7:

        rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=20.21,
            max_value=298.56,
            value=100.00,
            step=0.10,
            format="%.2f"
        )

        st.caption(
            "**Range:** 20.21–298.56 mm"
        )


    st.write("")


    submitted = st.form_submit_button(
        "🌾 Analyse Conditions & Recommend Crop",
        type="primary",
        width="stretch"
    )


# ============================================================
# INPUT INFORMATION
# ============================================================

st.info(
    """
    ℹ️ The displayed ranges correspond to values represented
    in the ANN training dataset.

    The model should only be used within these ranges.
    """
)


# ============================================================
# PREDICTION
# ============================================================

if submitted:

    # Create one input record
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


    # Scale using the saved scaler
    input_scaled = scaler.transform(
        input_data
    )


    # ANN prediction
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
    # RESULT
    # ========================================================

    st.divider()

    st.header("🎯 ANN Crop Recommendation")


    result_col1, result_col2 = st.columns(
        [2, 1]
    )


    with result_col1:

        st.subheader("🌱 Recommended Crop")

        st.title(
            predicted_crop.title()
        )


    with result_col2:

        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )


    # Confidence interpretation
    if confidence >= 90:

        st.success(
            "✅ Very High Model Confidence"
        )

    elif confidence >= 70:

        st.success(
            "✅ High Model Confidence"
        )

    elif confidence >= 50:

        st.warning(
            "⚠️ Moderate Model Confidence"
        )

    else:

        st.warning(
            "⚠️ Low Model Confidence"
        )


    # ========================================================
    # TOP 3
    # ========================================================

    st.subheader("📊 Top 3 ANN Predictions")


    top3_indices = np.argsort(
        probabilities
    )[-3:][::-1]


    for rank, index in enumerate(
        top3_indices,
        start=1
    ):

        crop_name = label_encoder.inverse_transform(
            [int(index)]
        )[0]


        crop_probability = float(
            probabilities[index]
        )


        crop_percent = (
            crop_probability * 100
        )


        name_col, percentage_col = st.columns(
            [4, 1]
        )


        with name_col:

            st.markdown(
                f"### {rank}. {crop_name.title()}"
            )


        with percentage_col:

            st.markdown(
                f"### {crop_percent:.2f}%"
            )


        st.progress(
            min(
                max(
                    crop_probability,
                    0.0
                ),
                1.0
            )
        )


    # ========================================================
    # INPUT SUMMARY
    # ========================================================

    st.subheader(
        "📋 Submitted Environmental Conditions"
    )


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
                f"{nitrogen:.0f}",
                f"{phosphorus:.0f}",
                f"{potassium:.0f}",
                f"{temperature:.2f}",
                f"{humidity:.2f}",
                f"{ph:.2f}",
                f"{rainfall:.2f}"
            ],

            "Unit": [
                "Dataset nutrient units",
                "Dataset nutrient units",
                "Dataset nutrient units",
                "°C",
                "%",
                "Unitless",
                "mm"
            ],

            "Training Range": [
                "0–140",
                "5–145",
                "5–205",
                "8.83–43.68",
                "14.26–99.98",
                "3.50–9.94",
                "20.21–298.56"
            ]
        }
    )


    st.dataframe(
        summary,
        width="stretch",
        hide_index=True
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()


with st.expander(
    "🧠 About the Artificial Neural Network"
):

    st.markdown(
        """
        ### Model Architecture

        **Input Layer**
        - 7 input variables

        **Hidden Layer 1**
        - 32 neurons
        - ReLU activation

        **Hidden Layer 2**
        - 16 neurons
        - ReLU activation

        **Output Layer**
        - 22 neurons
        - Softmax activation

        ### Input Variables

        1. Nitrogen
        2. Phosphorus
        3. Potassium
        4. Temperature
        5. Relative Humidity
        6. Soil pH
        7. Rainfall

        ### Output

        One recommended crop from 22 crop classes.
        """
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.warning(
    """
    ⚠️ **Academic prototype**

    This application was developed as an Artificial Neural
    Network academic project.

    Recommendations are based on patterns learned from the
    selected dataset and should not replace professional
    agricultural advice, laboratory soil testing, or
    site-specific field assessment.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Artificial Neural Network Crop Recommendation Project"
)

st.caption(
    "Inputs: N • P • K • Temperature • Humidity • pH • Rainfall"
)

st.caption(
    "Output: Recommended Crop from 22 Crop Classes"
)
