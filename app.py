import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from joblib import load
import datetime

# --- Page Configuration ---
st.set_page_config(page_title="😴 Sleep Pattern Analyzer", page_icon="😴", layout="wide")

# --- Load Models (Alternative Approach: Dictionary for flexibility) ---
@st.cache_resource
def load_prediction_models():
    model_paths = {
        "XGBoost Classifier": "Xgb_model.joblib",
        "Optimized XGBoost": "XGB_tunned_model.joblib",
    }
    loaded_models = {}
    for name, path in model_paths.items():
        try:
            loaded_models[name] = load(path)
            st.success(f"✅ Loaded {name}")
        except FileNotFoundError:
            st.error(f"⚠️ Model file not found: {path}")
        except Exception as e:
            st.error(f"🔥 Error loading {name}: {e}")
    return loaded_models

# --- Feature Engineering (Alternative: Using DBSCAN for anomaly detection as a feature) ---
def create_features(df):
    df_processed = df.copy()
    if 'series_id' not in df_processed.columns:
        df_processed['series_id'] = 0  # For single input

    # Simple rolling statistics
    df_processed['anglez_roll_mean_15'] = df_processed.groupby('series_id')['anglez'].rolling(window=15, min_periods=1).mean().reset_index(level=0, drop=True)
    df_processed['enmo_roll_std_15'] = df_processed.groupby('series_id')['enmo'].rolling(window=15, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)

    # DBSCAN for anomaly detection (potential indicator of unusual movement)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_processed[['anglez', 'enmo']].fillna(0))
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    df_processed['is_anomaly'] = dbscan.fit_predict(scaled_features)

    return df_processed[['anglez', 'enmo', 'anglez_roll_mean_15', 'enmo_roll_std_15', 'is_anomaly']].fillna(0)

# --- Prediction Function (More Explicit Handling) ---
def make_sleep_predictions(model, features):
    if model:
        try:
            predictions = model.predict(features)
            return predictions.astype(int)
        except Exception as e:
            st.error(f"🚨 Prediction error: {e}")
            return None
    else:
        st.warning("⚠️ No model selected or loaded.")
        return None

# --- Visualization Function (Using Plotly for Interactive Plots) ---
def visualize_predictions(df, predictions):
    if predictions is not None:
        plot_df = df.copy()
        plot_df['sleep_state'] = predictions
        fig = px.scatter(plot_df, x=plot_df.index, y='anglez', color='sleep_state',
                         color_continuous_scale=[(0, 'blue'), (1, 'orange')],
                         labels={'anglez': 'Angle Z', 'index': 'Time Step', 'sleep_state': 'Predicted Sleep'})
        fig.update_layout(title='Predicted Sleep States Over Time (vs. Angle Z)')
        st.plotly_chart(fig, use_container_width=True)

        sleep_percentage = np.mean(predictions) * 100
        st.metric("Estimated Sleep Percentage", f"{sleep_percentage:.2f}%")
    else:
        st.warning("No predictions to visualize.")

# --- Main Application ---
st.title("😴 Dynamic Sleep Stage Classifier")
st.markdown("Analyze your movement data to predict sleep patterns.")

loaded_models = load_prediction_models()
model_options = list(loaded_models.keys())
selected_model_name = st.sidebar.selectbox("Choose Prediction Model", model_options)
selected_model = loaded_models.get(selected_model_name)

st.subheader("Input Your Data")
data_source = st.radio("Select data input method:", ["Upload CSV", "Manual Entry", "Sample"])

if data_source == "Upload CSV":
    uploaded_file = st.file_uploader("Upload a CSV file with 'timestamp', 'anglez', and 'enmo' columns", type=["csv"])
    if uploaded_file:
        try:
            input_df = pd.read_csv(uploaded_file)
            if not all(col in input_df.columns for col in ['timestamp', 'anglez', 'enmo']):
                st.error("CSV must contain 'timestamp', 'anglez', and 'enmo' columns.")
            else:
                st.dataframe(input_df.head())
                if st.button("Analyze Uploaded Data"):
                    processed_features = create_features(input_df.copy())
                    predictions = make_sleep_predictions(selected_model, processed_features)
                    if predictions is not None:
                        visualize_predictions(input_df, predictions)
                        st.subheader("Prediction Output")
                        st.write(pd.DataFrame({'Predicted Sleep State': predictions}))
        except Exception as e:
            st.error(f"Error loading or processing CSV: {e}")

elif data_source == "Manual Entry":
    st.info("Enter a few data points for a quick analysis.")
    data_points = []
    with st.form("manual_data_form"):
        anglez_input = st.number_input("Angle Z", value=0.0)
        enmo_input = st.number_input("ENMO", value=0.0)
        submitted = st.form_submit_button("Add Data Point")
        if submitted:
            data_points.append({'anglez': anglez_input, 'enmo': enmo_input})

    if data_points:
        manual_df = pd.DataFrame(data_points)
        st.dataframe(manual_df)
        if st.button("Analyze Manual Data"):
            processed_features = create_features(manual_df.copy())
            predictions = make_sleep_predictions(selected_model, processed_features)
            if predictions is not None:
                visualize_predictions(manual_df, predictions)
                st.subheader("Prediction Output")
                st.write(pd.DataFrame({'Predicted Sleep State': predictions}))

elif data_source == "Sample":
    sample_data = pd.DataFrame({
        'timestamp': pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 00:00:30', '2023-01-01 00:01:00', '2023-01-01 00:01:30', '2023-01-01 00:02:00']),
        'anglez': [10, 12, -30, -40, -35],
        'enmo': [0.1, 0.05, 0.8, 0.9, 0.7]
    })
    st.dataframe(sample_data)
    if st.button("Analyze Sample Data"):
        processed_features = create_features(sample_data.copy())
        predictions = make_sleep_predictions(selected_model, processed_features)
        if predictions is not None:
            visualize_predictions(sample_data, predictions)
            st.subheader("Prediction Output")
            st.write(pd.DataFrame({'Predicted Sleep State': predictions}))

st.sidebar.header("About")
st.sidebar.info("This application analyzes accelerometer data to predict sleep patterns. "
                "It uses machine learning models to classify periods as sleep or awake.")
