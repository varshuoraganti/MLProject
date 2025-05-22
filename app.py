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
st.set_page_config(page_title="😴 Sleep Pattern Insights", page_icon="😴", layout="wide")

# --- Load Models ---
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
            st.sidebar.success(f"✅ Loaded {name}")
        except FileNotFoundError:
            st.sidebar.error(f"⚠️ Model file not found: {path}")
        except Exception as e:
            st.sidebar.error(f"🔥 Error loading {name}: {e}")
    return loaded_models

# --- Feature Engineering ---
def create_features(df):
    df_processed = df.copy()
    if 'series_id' not in df_processed.columns:
        df_processed['series_id'] = 0

    df_processed['anglez_smooth'] = df_processed['anglez'].rolling(window=10, min_periods=1).mean().fillna(df_processed['anglez'])
    df_processed['enmo_derivative'] = df_processed['enmo'].diff().fillna(0)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_processed[['anglez_smooth', 'enmo_derivative']].fillna(0))
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    df_processed['motion_cluster'] = dbscan.fit_predict(scaled_features)

    return df_processed[['anglez_smooth', 'enmo_derivative', 'motion_cluster']].fillna(0)

# --- Prediction Function ---
def make_sleep_predictions(model, features):
    if model:
        try:
            predictions = model.predict(features)
            return predictions.astype(int)
        except Exception as e:
            st.error(f"🚨 Prediction error: {e}")
            return None
    else:
        st.sidebar.warning("⚠️ No model selected or loaded.")
        return None

# --- Visualization Function ---
def visualize_predictions(df, predictions):
    if predictions is not None:
        plot_df = df.copy()
        plot_df['sleep_state'] = predictions
        fig = px.scatter(plot_df, x=plot_df.index, y='anglez', color='sleep_state',
                         color_continuous_scale=[(0, 'lightblue'), (1, 'darkblue')],
                         labels={'anglez': 'Angle Z', 'index': 'Chronological Order', 'sleep_state': 'Sleep Indicator'})
        fig.update_layout(title='Sleep/Awake Timeline (vs. Smoothed Angle Z)')
        st.plotly_chart(fig, use_container_width=True)

        sleep_proportion = np.mean(predictions)
        st.metric("Estimated Sleep Ratio", f"{sleep_proportion:.2f}")
    else:
        st.warning("No predictions to visualize.")

# --- Sidebar for Data Input ---
st.sidebar.header("Data Acquisition")
acquisition_method = st.sidebar.radio(
    "Select your data input:",
    ["Browse File System", "Manual Data Entry", "Example Dataset"]
)

input_df = None

if acquisition_method == "Browse File System":
    uploaded_file = st.sidebar.file_uploader("Load your CSV (with 'timestamp', 'anglez', 'enmo')", type=["csv"])
    if uploaded_file:
        try:
            input_df = pd.read_csv(uploaded_file)
            if not all(col in input_df.columns for col in ['timestamp', 'anglez', 'enmo']):
                st.sidebar.error("CSV must contain 'timestamp', 'anglez', and 'enmo' columns.")
                input_df = None
            else:
                st.sidebar.success("Data loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")
            input_df = None

elif acquisition_method == "Manual Data Entry":
    st.sidebar.info("Provide a few data points.")
    manual_data = []
    with st.sidebar.form("manual_entry_form"):
        anglez_val = st.number_input("Angle Z Value", value=0.0)
        enmo_val = st.number_input("ENMO Value", value=0.0)
        add_point = st.form_submit_button("Add Data Point")
        if add_point:
            manual_data.append({'anglez': anglez_val, 'enmo': enmo_val})

    if manual_data:
        input_df = pd.DataFrame(manual_data)
        st.sidebar.dataframe(input_df)
        if not input_df.empty:
            st.sidebar.info("Click 'Analyze' to see predictions.")

elif acquisition_method == "Example Dataset":
    example_data = pd.DataFrame({
        'timestamp': pd.to_datetime(['2024-01-01 00:00:00', '2024-01-01 00:00:30', '2024-01-01 00:01:00', '2024-01-01 00:01:30', '2024-01-01 00:02:00']),
        'anglez': [5, 7, -25, -35, -30],
        'enmo': [0.05, 0.02, 0.7, 0.8, 0.6]
    })
    input_df = example_data
    st.sidebar.info("Using sample data. Click 'Analyze' to proceed.")
    st.sidebar.dataframe(input_df)

# --- Main Application Area ---
st.title("😴 Sleep Pattern Insights")
st.markdown("Explore your movement data to understand sleep states.")

loaded_models = load_prediction_models()
model_options = list(loaded_models.keys())
selected_model_name = st.sidebar.selectbox("Select Prediction Engine", model_options)
selected_model = loaded_models.get(selected_model_name)

st.subheader("Data Preview")
if input_df is not None:
    st.dataframe(input_df.head())

    if st.button("Initiate Analysis"):
        processed_features = create_features(input_df.copy())
        predictions = make_sleep_predictions(selected_model, processed_features)
        visualize_predictions(input_df, predictions)
        st.subheader("Prediction Outcome")
        if predictions is not None:
            st.write(pd.DataFrame({'Predicted Sleep State': predictions}))
else:
    st.info("Please load or enter your data in the sidebar to begin analysis.")

st.sidebar.header("About This App")
st.sidebar.info("This application processes accelerometer data (Angle Z and ENMO) to predict periods of sleep and wakefulness. Choose your data input method and a prediction model in the sidebar.")
