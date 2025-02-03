import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import tensorflow as tf
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

tf.keras.utils.get_custom_objects()["mse"] = tf.keras.losses.mean_squared_error

st.set_page_config(page_title="Housing Price Prediction - Inference", layout="wide")
st.title("Housing Price Prediction - Inference")

# Check for required files
required_files = [
    "tf_housing_model.h5",
    "scaler_X.pkl",
    "scaler_y.pkl",
    "feature_order.json",
    "Housing.csv",
]
for file in required_files:
    if not os.path.exists(file):
        st.error(
            f"Required file '{file}' not found. Please run the training app first and save the model, scalers, and feature order."
        )
        st.stop()

# Load trained model and scalers
model = tf.keras.models.load_model("tf_housing_model.h5")
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

# Load a list of column names(featured_order)
with open("feature_order.json", "r") as f:
    feature_order = json.load(f)

# Function to encode categorical features in the data.
def encode_data(df):
    df_encoded = df.copy()
    encoding_cols = [
        "furnishingstatus",
        "prefarea",
        "airconditioning",
        "hotwaterheating",
        "basement",
        "guestroom",
        "mainroad",
    ]
    for col in encoding_cols:
        # Ensure data is string type for LabelEncoder
        df_encoded[col] = df_encoded[col].astype(str)
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
    # Drop the original price column since we use price_millions as target.
    df_encoded.drop(columns=["price"], inplace=True)
    return df_encoded

# Option to choose input method
input_method = st.sidebar.radio(
    "Select Input Method:", options=["Manual Input", "Use a Row from Test Set"]
)

if input_method == "Manual Input":
    st.write("## Enter Property Details Manually")
    # Build inputs for the expected features using radio buttons for categorical options.
    input_data = {}
    if "area" in feature_order:
        input_data["area"] = st.number_input(
            "Area (in square feet)",
            min_value=300.0,
            max_value=20000.0,
            value=2000.0,
            step=50.0,
        )
    if "bedrooms" in feature_order:
        input_data["bedrooms"] = st.number_input(
            "Number of Bedrooms", min_value=1, max_value=10, value=3, step=1
        )
    if "bathrooms" in feature_order:
        input_data["bathrooms"] = st.number_input(
            "Number of Bathrooms", min_value=1, max_value=10, value=2, step=1
        )
    if "stories" in feature_order:
        input_data["stories"] = st.number_input(
            "Number of Stories", min_value=1, max_value=5, value=2, step=1
        )
    if "parking" in feature_order:
        input_data["parking"] = st.number_input(
            "Parking Spaces", min_value=0, max_value=5, value=1, step=1
        )
    if "furnishingstatus" in feature_order:
        input_data["furnishingstatus"] = st.radio(
            "Furnishing Status",
            options=[0, 1, 2],
            format_func=lambda x: {
                0: "Unfurnished",
                1: "Semi-Furnished",
                2: "Furnished",
            }[x],
        )
    if "prefarea" in feature_order:
        input_data["prefarea"] = st.radio(
            "Preferred Area",
            options=[0, 1],
            format_func=lambda x: {
                0: "No",
                1: "Yes",
            }[x],
        )
    if "airconditioning" in feature_order:
        input_data["airconditioning"] = st.radio(
            "Air Conditioning",
            options=[0, 1],
            format_func=lambda x: {
                0: "No",
                1: "Yes",
            }[x],
        )
    if "hotwaterheating" in feature_order:
        input_data["hotwaterheating"] = st.radio(
            "Hot Water Heating",
            options=[0, 1],
            format_func=lambda x: {
                0: "No",
                1: "Yes",
            }[x],
        )
    if "basement" in feature_order:
        input_data["basement"] = st.radio(
            "Basement",
            options=[0, 1],
            format_func=lambda x: {
                0: "No",
                1: "Yes",
            }[x],
        )
    if "guestroom" in feature_order:
        input_data["guestroom"] = st.radio(
            "Guestroom",
            options=[0, 1],
            format_func=lambda x: {
                0: "No",
                1: "Yes",
            }[x],
        )
    if "mainroad" in feature_order:
        input_data["mainroad"] = st.radio(
            "Mainroad",
            options=[0, 1],
            format_func=lambda x: {
                0: "No",
                1: "Yes",
            }[x],
        )

    if st.button("Predict Price"):
        # Create a DataFrame with the input data and reorder columns as during training.
        input_df = pd.DataFrame([input_data])
        input_df = input_df[feature_order]

        # Scale the input features
        X_new = scaler_X.transform(input_df)

        # Make prediction (model output is scaled)
        y_pred_scaled = model.predict(X_new)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        # Convert predicted price from millions to dollars.
        predicted_price = y_pred[0][0] * 1e6

        st.success(f"Predicted House Price: ${predicted_price:,.0f}")

else:  # Use a Row from Test Set
    st.write("## Use a Row from the Test Set")
    # Load the original dataset
    df_full = pd.read_csv("Housing.csv")
    df_full["price_millions"] = df_full["price"] / 1e6

    # Display the full original dataset (or a subset) for reference
    if st.checkbox("Show Full Dataset"):
        st.write(df_full)

    # Preprocess the data
    df_encoded = encode_data(df_full)
    # Ensure the columns match the training feature order; add the target for comparison.
    required_cols = feature_order + ["price_millions"]
    try:
        df_encoded = df_encoded[required_cols]
    except Exception as e:
        st.error("Error in reordering columns: " + str(e))
        st.stop()

    # Split into training and test sets using the same random state as in training.
    X = df_encoded.drop(columns=["price_millions"])
    y = df_encoded["price_millions"].values.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=50
    )

    # Let the user select a test row based on the DataFrame index.
    test_index = st.selectbox("Select a test row index", options=list(X_test.index))

    # Display the selected row from the original dataset
    st.write("### Selected Test Row (Original Data)")
    selected_row_original = df_full.loc[[test_index]]
    st.dataframe(selected_row_original)

    # Extract the chosen test row (as a DataFrame with one row) from X_test.
    input_df = X_test.loc[[test_index]]
    # Retrieve actual price from y_test corresponding to the selected row.
    pos = list(X_test.index).index(test_index)
    actual_price_millions = y_test[pos][0]
    actual_price = actual_price_millions * 1e6
    st.write(f"**Actual Price:** ${actual_price:,.0f}")

    if st.button("Predict Price from Test Row"):
        # Make prediction on the selected test row.
        X_new = scaler_X.transform(input_df)
        y_pred_scaled = model.predict(X_new)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        predicted_price = y_pred[0][0] * 1e6

        st.success(f"Predicted House Price: ${predicted_price:,.0f}")

        # Calculate error margin
        error_margin = abs(predicted_price - actual_price)
        percent_error = (error_margin / actual_price) * 100 if actual_price != 0 else 0

        st.write(f"**Error Margin:** ${error_margin:,.0f} ({percent_error:.2f}%)")

        # Visualization: Compare Actual vs. Predicted on the selected row
        fig_compare, ax_compare = plt.subplots(figsize=(4, 3))
        ax_compare.bar(
            ["Actual", "Predicted"],
            [actual_price, predicted_price],
            color=["green", "blue"],
        )
        ax_compare.set_ylabel("Price (in dollars)")
        ax_compare.set_title("Actual vs. Predicted Price")
        st.pyplot(fig_compare, use_container_width=False)