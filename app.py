import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Car Industry Cluster Predictor",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Car Industry Cluster Predictor")
st.write(
    "Enter car-industry information below. The saved KNN model "
    "will validate the values and predict the K-Means cluster."
)

# ---------------------------------------------------------
# Load the saved model bundle
# ---------------------------------------------------------
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "car_industry_knn_pipeline.joblib"
)

@st.cache_resource
def load_model_bundle(model_path):
    """Load the trained model and its validation metadata."""
    return joblib.load(model_path)

if not os.path.exists(MODEL_PATH):
    st.error(
        "Model file not found. Place "
        "'car_industry_knn_pipeline.joblib' in the same folder as app.py."
    )
    st.stop()

try:
    model_bundle = load_model_bundle(MODEL_PATH)
except Exception as error:
    st.error(f"Could not load the model file: {error}")
    st.stop()

# Read model metadata from the saved bundle.
model = model_bundle["model"]
required_features = model_bundle["features"]
feature_ranges = model_bundle["feature_ranges"]

# ---------------------------------------------------------
# Input validation function
# ---------------------------------------------------------
def validate_car_data(data, bundle):
    """Validate required columns, numeric values, ranges and price order."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a Pandas DataFrame.")

    required_columns = bundle["features"]
    ranges = bundle["feature_ranges"]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Keep only the model features in the required order.
    validated_data = data[required_columns].copy()

    # Convert all input columns to numeric values.
    for column in required_columns:
        validated_data[column] = pd.to_numeric(
            validated_data[column],
            errors="coerce"
        )

    # Reject blank and non-numeric values.
    if validated_data.isnull().any().any():
        raise ValueError("Please enter valid numeric values for every field.")

    # Reject infinite values.
    if not np.isfinite(validated_data.to_numpy(dtype=float)).all():
        raise ValueError("Input values must be finite numbers.")

    # Check each value against the observed training range.
    for column in required_columns:
        minimum = ranges[column]["min"]
        maximum = ranges[column]["max"]

        if not validated_data[column].between(minimum, maximum).all():
            raise ValueError(
                f"{column} must be between {minimum:g} and {maximum:g}, "
                "based on the model's training data."
            )

    # Check that minimum price is not greater than maximum price.
    if (
        validated_data["IllustrativePriceUSD_Min"]
        > validated_data["IllustrativePriceUSD_Max"]
    ).any():
        raise ValueError(
            "Minimum illustrative price cannot exceed maximum illustrative price."
        )

    return validated_data

# ---------------------------------------------------------
# Input form
# ---------------------------------------------------------
st.subheader("Enter Vehicle and Sales Information")

# Suggested starting values are based on a sample record.
default_values = {
    "ModelYear": 2024,
    "Year": 2023,
    "UnitsSold": 125000,
    "IllustrativePriceUSD_Min": 18000,
    "IllustrativePriceUSD_Max": 30000
}

# Map feature names to readable labels.
field_labels = {
    "ModelYear": "Model Year",
    "Year": "Sales Year",
    "UnitsSold": "Units Sold",
    "IllustrativePriceUSD_Min": "Minimum Illustrative Price (USD)",
    "IllustrativePriceUSD_Max": "Maximum Illustrative Price (USD)"
}

with st.form("car_prediction_form"):
    entered_values = {}

    for feature in required_features:
        minimum = feature_ranges[feature]["min"]
        maximum = feature_ranges[feature]["max"]
        default = default_values[feature]

        # Keep the default inside the observed training range.
        default = min(max(default, minimum), maximum)

        # Year, model year and units sold are whole-number inputs.
        if feature in ["ModelYear", "Year", "UnitsSold"]:
            entered_values[feature] = st.number_input(
                field_labels[feature],
                min_value=int(minimum),
                max_value=int(maximum),
                value=int(default),
                step=1
            )
        else:
            entered_values[feature] = st.number_input(
                field_labels[feature],
                min_value=float(minimum),
                max_value=float(maximum),
                value=float(default),
                step=100.0,
                format="%.2f"
            )

    submitted = st.form_submit_button(
        "Validate and Predict Cluster",
        use_container_width=True
    )

# ---------------------------------------------------------
# Validate input and make prediction
# ---------------------------------------------------------
if submitted:
    # Convert the entered values into a one-row DataFrame.
    input_data = pd.DataFrame([entered_values])

    try:
        # Validate all values before passing them to the model.
        validated_data = validate_car_data(input_data, model_bundle)

        # Predict the K-Means cluster using the saved KNN pipeline.
        prediction = model.predict(validated_data)[0]

        st.success("Input validation successful!")

        st.subheader("Prediction Result")
        st.metric("Predicted K-Means Cluster", str(prediction))

        # Display the validated values used for prediction.
        st.subheader("Validated Input")
        st.dataframe(validated_data, use_container_width=True)

        # Display prediction confidence when supported by the model.
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(validated_data)[0]
            class_labels = model.classes_

            probability_table = pd.DataFrame({
                "Cluster": class_labels,
                "Estimated Probability (%)": probabilities * 100
            }).sort_values(
                by="Estimated Probability (%)",
                ascending=False
            )

            st.subheader("Model Probability Estimates")
            st.dataframe(
                probability_table,
                use_container_width=True,
                hide_index=True
            )

        st.caption(
            "The predicted cluster represents a K-Means-generated group. "
            "It is not an independently verified real-world vehicle category."
        )

    except (TypeError, ValueError) as error:
        st.error(f"Input validation failed: {error}")

    except Exception as error:
        st.error(f"Prediction failed: {error}")

# ---------------------------------------------------------
# Model information
# ---------------------------------------------------------
with st.expander("About this model"):
    st.write(f"**Algorithm:** {model_bundle.get('algorithm', 'K-Nearest Neighbors')}")
    st.write(f"**Target:** {model_bundle.get('target', 'KMeans_Cluster')}")
    st.write(f"**Input features:** {', '.join(required_features)}")
    st.write(
        "**Validation ranges:** These are the minimum and maximum values "
        "observed in the training dataset. They are model-domain checks, "
        "not universal limits for the car industry."
    )
