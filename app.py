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
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>

    /* ------------------------------------------------------
       MAIN PAGE
    ------------------------------------------------------ */

    .stApp {
        background-color: #f8fbf8 !important;
        color: #1f2937 !important;
    }

    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* ------------------------------------------------------
       STANDARD TEXT
    ------------------------------------------------------ */

    [data-testid="stMarkdownContainer"] p {
        color: #1f2937 !important;
    }

    [data-testid="stMarkdownContainer"] li {
        color: #1f2937 !important;
    }

    [data-testid="stMarkdownContainer"] h1 {
        color: #166534 !important;
    }

    [data-testid="stMarkdownContainer"] h2 {
        color: #166534 !important;
    }

    [data-testid="stMarkdownContainer"] h3 {
        color: #166534 !important;
    }


    /* ------------------------------------------------------
       INPUT LABELS
    ------------------------------------------------------ */

    [data-testid="stWidgetLabel"] p {
        color: #111827 !important;
        font-size: 15px !important;
        font-weight: 700 !important;
    }

    [data-testid="stCaptionContainer"] p {
        color: #4b5563 !important;
        font-size: 13px !important;
    }


    /* ------------------------------------------------------
       NUMBER INPUTS
    ------------------------------------------------------ */

    [data-baseweb="input"] {
        background-color: #ffffff !important;
    }

    [data-baseweb="input"] input {
        color: #111827 !important;
        background-color: #ffffff !important;
        font-weight: 600 !important;
    }


    /* ------------------------------------------------------
       PRIMARY BUTTON
    ------------------------------------------------------ */

    button[kind="primary"] {
        background-color: #2e7d32 !important;
        border-color: #2e7d32 !important;
        color: white !important;
        font-weight: 700 !important;
        min-height: 50px !important;
    }

    button[kind="primary"] p {
        color: white !important;
    }

    button[kind="primary"]:hover {
        background-color: #1b5e20 !important;
        border-color: #1b5e20 !important;
    }


    /* ------------------------------------------------------
       METRIC OUTPUT
    ------------------------------------------------------ */

    [data-testid="stMetric"] {
        background-color: #f1f8e9;
        border: 1px solid #a5d6a7;
        border-radius: 12px;
        padding: 15px;
    }

    [data-testid="stMetricLabel"] p {
        color: #374151 !important;
        font-weight: 600 !important;
    }

    [data-testid="stMetricValue"] {
        color: #166534 !important;
        font-weight: 800 !important;
    }


    /* ------------------------------------------------------
       PROGRESS BARS
    ------------------------------------------------------ */

    [data-testid="stProgress"] > div > div {
        background-color: #43a047 !important;
    }


    /* ------------------------------------------------------
       DATAFRAME / EXPANDER
    ------------------------------------------------------ */

    [data-testid="stExpander"] summary p {
        color: #1f2937 !important;
        font-weight: 600 !important;
    }


    /* ------------------------------------------------------
       FOOTER
    ------------------------------------------------------ */

    .project-footer {
        text-align: center;
        color: #5f6b63 !important;
        font-size: 13px;
        margin-top: 30px;
        padding-top: 18px;
        border-top: 1px solid #d8e3da;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FILE NAMES
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
    file
    for file in required_files
    if not os.path.exists(file)
]

if missing_files:

    st.error(
        "Required model files are missing: "
        + ", ".join(missing_files)
    )

    st.stop()


# ============================================================
# LOAD TRAINED SYSTEM
# ============================================================

@st.cache_resource
def load_system():

    # compile=False is sufficient because the deployed app
    # only performs predictions and does not retrain the ANN.
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

st.title(
    "🌾 Smart Crop Recommendation System"
)

st.markdown(
    "### Artificial Neural Network Based Agricultural Decision Support System"
)

st.success(
    """
    🌱 **How does it work?**

    Enter the soil nutrient and climatic conditions of the
    agricultural area below.

    The trained Artificial Neural Network analyses seven
    environmental parameters and recommends the most suitable
    crop from **22 crop classes**.
    """
)


# ============================================================
# INPUT INFORMATION
# ============================================================

st.subheader(
    "🧪 Soil Nutrient Parameters"
)

st.caption(
    "The ranges shown below correspond to values represented "
    "in the model's training dataset."
)


# ============================================================
# INPUT FORM
# ============================================================

with st.form(
    "crop_recommendation_form"
):

    # --------------------------------------------------------
    # SOIL NUTRIENTS
    # --------------------------------------------------------

    nutrient_col1, nutrient_col2, nutrient_col3 = st.columns(3)


    with nutrient_col1:

        nitrogen = st.number_input(
            "Nitrogen (N) | Range: 0–140 kg/ha",
            min_value=0.0,
            max_value=140.0,
            value=70.0,
            step=1.0,
            format="%.0f",
            help="Training dataset range: 0–140 kg/ha"
        )


    with nutrient_col2:

        phosphorus = st.number_input(
            "Phosphorus (P) | Range: 5–145 kg/ha",
            min_value=5.0,
            max_value=145.0,
            value=50.0,
            step=1.0,
            format="%.0f",
            help="Training dataset range: 5–145 kg/ha"
        )


    with nutrient_col3:

        potassium = st.number_input(
            "Potassium (K) | Range: 5–205 kg/ha",
            min_value=5.0,
            max_value=205.0,
            value=50.0,
            step=1.0,
            format="%.0f",
            help="Training dataset range: 5–205 kg/ha"
        )


    # --------------------------------------------------------
    # CLIMATIC CONDITIONS
    # --------------------------------------------------------

    st.subheader(
        "🌦️ Climatic Conditions"
    )


    climate_col1, climate_col2 = st.columns(2)


    with climate_col1:

        temperature = st.number_input(
            "Temperature | Range: 8.83–43.68 °C",
            min_value=8.83,
            max_value=43.68,
            value=25.00,
            step=0.10,
            format="%.2f",
            help="Training dataset range: 8.83–43.68 °C"
        )


    with climate_col2:

        humidity = st.number_input(
            "Relative Humidity | Range: 14.26–99.98 %",
            min_value=14.26,
            max_value=99.98,
            value=70.00,
            step=0.10,
            format="%.2f",
            help="Training dataset range: 14.26–99.98%"
        )


    # --------------------------------------------------------
    # SOIL pH AND RAINFALL
    # --------------------------------------------------------

    st.subheader(
        "🌍 Soil & Rainfall Conditions"
    )


    environment_col1, environment_col2 = st.columns(2)


    with environment_col1:

        ph = st.number_input(
            "Soil pH | Range: 3.50–9.94",
            min_value=3.50,
            max_value=9.94,
            value=6.50,
            step=0.01,
            format="%.2f",
            help="pH is unitless. Training dataset range: 3.50–9.94"
        )


    with environment_col2:

        rainfall = st.number_input(
            "Rainfall | Range: 20.21–298.56 mm",
            min_value=20.21,
            max_value=298.56,
            value=100.00,
            step=0.10,
            format="%.2f",
            help="Training dataset range: 20.21–298.56 mm"
        )


    # --------------------------------------------------------
    # SUBMIT BUTTON
    # --------------------------------------------------------

    st.write("")

    submitted = st.form_submit_button(
        "🌾 Analyse Conditions & Recommend Crop",
        type="primary",
        use_container_width=True
    )


# ============================================================
# INPUT GUIDANCE
# ============================================================

st.info(
    """
    ℹ️ **Input guidance:** Predictions are restricted to the
    ranges represented in the ANN training dataset.

    The model should not be considered validated for values
    outside these ranges.
    """
)


# ============================================================
# MAKE PREDICTION
# ============================================================

if submitted:

    # --------------------------------------------------------
    # CREATE NEW INPUT RECORD
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        {
            "N": [
                nitrogen
            ],

            "P": [
                phosphorus
            ],

            "K": [
                potassium
            ],

            "temperature": [
                temperature
            ],

            "humidity": [
                humidity
            ],

            "ph": [
                ph
            ],

            "rainfall": [
                rainfall
            ]
        }
    )


    # --------------------------------------------------------
    # SCALE INPUT
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


    predicted_index = int(
        np.argmax(
            probabilities
        )
    )


    predicted_crop = label_encoder.inverse_transform(
        [
            predicted_index
        ]
    )[0]


    confidence = float(
        probabilities[
            predicted_index
        ] * 100
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
    # MAIN RECOMMENDATION
    # ========================================================

    st.divider()

    st.subheader(
        "🎯 ANN Recommendation"
    )


    result_col1, result_col2 = st.columns(
        [
            2,
            1
        ]
    )


    with result_col1:

        st.markdown(
            "### 🌱 Recommended Crop"
        )

        st.markdown(
            f"# {predicted_crop.title()}"
        )

        st.caption(
            "Recommended using the soil nutrient and climatic "
            "conditions entered above."
        )


    with result_col2:

        st.metric(
            label="Model Confidence",
            value=f"{confidence:.2f}%"
        )

        st.markdown(
            f"**{confidence_description}**"
        )


    # ========================================================
    # TOP 3 PREDICTIONS
    # ========================================================

    st.subheader(
        "📊 Top 3 ANN Predictions"
    )


    top3_indices = np.argsort(
        probabilities
    )[-3:][::-1]


    for rank, index in enumerate(
        top3_indices,
        start=1
    ):

        crop_name = label_encoder.inverse_transform(
            [
                int(
                    index
                )
            ]
        )[0]


        crop_probability = float(
            probabilities[
                index
            ]
        )


        probability_percent = (
            crop_probability * 100
        )


        label_col, percentage_col = st.columns(
            [
                4,
                1
            ]
        )


        with label_col:

            st.markdown(
                f"**{rank}. {crop_name.title()}**"
            )


        with percentage_col:

            st.markdown(
                f"**{probability_percent:.2f}%**"
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
    # SUBMITTED INPUT SUMMARY
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

                "Entered Value": [
                    f"{nitrogen:.0f}",
                    f"{phosphorus:.0f}",
                    f"{potassium:.0f}",
                    f"{temperature:.2f}",
                    f"{humidity:.2f}",
                    f"{ph:.2f}",
                    f"{rainfall:.2f}"
                ],

                "Unit": [
                    "kg/ha",
                    "kg/ha",
                    "kg/ha",
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
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

with st.expander(
    "ℹ️ About this ANN model"
):

    st.markdown(
        """
        **Model type:** Feed-forward Artificial Neural Network

        **Input features:** 7

        - Nitrogen
        - Phosphorus
        - Potassium
        - Temperature
        - Relative Humidity
        - Soil pH
        - Rainfall

        **ANN architecture:**

        - Input: 7 features
        - Hidden Layer 1: 32 neurons, ReLU
        - Hidden Layer 2: 16 neurons, ReLU
        - Output Layer: 22 crop classes, Softmax

        **Output:** Recommended crop class
        """
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.warning(
    """
    ⚠️ **Academic prototype**

    This application is an Artificial Neural Network
    demonstration developed for an academic project.

    Recommendations are based on patterns learned from the
    selected dataset and should not replace professional
    agricultural advice, laboratory soil testing, or
    site-specific field assessment.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="project-footer">

    <b>Artificial Neural Network Crop Recommendation Project</b>

    <br><br>

    Inputs: N • P • K • Temperature • Humidity • pH • Rainfall

    <br>

    Output: Recommended Crop from 22 Crop Classes

    </div>
    """,
    unsafe_allow_html=True
)
