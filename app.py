import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from sklearn.preprocessing import StandardScaler
import datetime
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Styling ---
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #f0f8ff; /* Light Alice Blue */
        color: #1e3a8a; /* Dark Blue */
    }
    .st-title {
        color: #4a76a8; /* Steel Blue */
    }
    .st-header {
        color: #38761d; /* Forest Green */
    }
    .st-subheader {
        color: #7b68ee; /* Medium Slate Blue */
    }
    .st-button button {
        background-color: #ffa07a !important; /* Light Salmon */
        color: white !important;
        border: none !important;
        border-radius: 5px !important;
        padding: 8px 15px !important;
    }
    .st-button button:hover {
        background-color: #fa8072 !important; /* Salmon */
    }
    .metric-value {
        color: #d2691e !important; /* Chocolate */
    }
    .dataframe {
        border: 1px solid #808080; /* Gray */
        border-radius: 5px;
        padding: 10px;
        background-color: #e0ffff; /* Light Cyan */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Load Models ---
@st.cache_resource
def load_models():
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    models = {}
    model_files = {
        "ZenithBoost": 'Xgb_model.joblib',  # Unique Name
        "ApexTunedBoost": 'XGB_tunned_model.joblib'  # Unique Name
    }

    loaded_models = []
    errors = []

    for model_name, file_path in model_files.items():
        try:
            model = load(file_path)
            models[model_name] = model
            loaded_models.append(model_name)
            st.sidebar.success(f"✅ Loaded {model_name} model")
        except Exception as e:
            errors.append(f"Error loading {model_name}: {str(e)}")
            st.sidebar.error(f"❌ Error loading {model_name}: {str(e)}")

    if not models:
        st.sidebar.error("⚠️ Could not load any models. Using fallback models instead.")
        return create_fallback_models()
    else:
        if errors:
            st.sidebar.warning(f"⚠️ Some models could not be loaded. Only using: {', '.join(loaded_models)}")
        return models

# --- Fallback Model ---
def create_fallback_models():
    class SimpleModel:
        def __init__(self, name, prediction_strategy="pattern"):
            self.name = name
            self.strategy = prediction_strategy

        def predict(self, X):
            if self.strategy == "random":
                return np.random.choice([0, 1], size=X.shape[0], p=[0.7, 0.3])
            elif self.strategy == "pattern":
                n_samples = X.shape[0]
                predictions = np.zeros(n_samples)
                if n_samples > 10:
                    sleep_start = n_samples // 4
                    sleep_end = (n_samples * 3) // 4
                    predictions[sleep_start:sleep_end] = 1
                return predictions
            else:
                return np.zeros(X.shape[0])

    models = {
        "ZenithBoost (Fallback)": SimpleModel("ZenithBoost", "pattern"),
        "ApexTunedBoost (Fallback)": SimpleModel("ApexTunedBoost", "pattern")
    }
    return models

# --- Feature Calculation ---
def calculate_rolling_features(df):
    data = df.copy()
    if 'series_id' not in data.columns:
        data['series_id'] = 1
    data['sd_enmo_1'] = np.nan
    data['sd_anglez_1'] = np.nan
    data['m_enmo_2'] = np.nan
    data['m_anglez_2'] = np.nan

    try:
        data['sd_anglez_1'] = (data.groupby('series_id')['anglez']
                         .rolling(12).std().reset_index(level=0, drop=True))
        data.loc[data['sd_anglez_1'].isna(), 'sd_anglez_1'] = (
            data.groupby('series_id')['anglez']
            .rolling(2).std().reset_index(level=0, drop=True)
        )
        data['sd_enmo_1'] = (data.groupby('series_id')['enmo']
                       .rolling(12).std().reset_index(level=0, drop=True))
        data.loc[data['sd_enmo_1'].isna(), 'sd_enmo_1'] = (
            data.groupby('series_id')['enmo']
            .rolling(2).std().reset_index(level=0, drop=True)
        )
        data['m_enmo_2'] = (data.groupby('series_id')['enmo']
                      .rolling(24).mean().reset_index(level=0, drop=True))
        data.loc[data['m_enmo_2'].isna(), 'm_enmo_2'] = (
            data.groupby('series_id')['enmo']
            .rolling(2).mean().reset_index(level=0, drop=True)
        )
        data['m_anglez_2'] = (data.groupby('series_id')['anglez']
                        .rolling(24).mean().reset_index(level=0, drop=True))
        data.loc[data['m_anglez_2'].isna(), 'm_anglez_2'] = (
            data.groupby('series_id')['anglez']
            .rolling(2).mean().reset_index(level=0, drop=True)
        )
    except Exception as e:
        st.sidebar.warning(f"⚠️ Error calculating rolling features: {e}")

    data['sd_enmo_1'].fillna(0.0, inplace=True)
    data['sd_anglez_1'].fillna(0.0, inplace=True)
    data['m_enmo_2'].fillna(0.0, inplace=True)
    data['m_anglez_2'].fillna(0.0, inplace=True)
    return data

# --- Clustering ---
def perform_clustering(df):
    try:
        kmeans = KMeans(n_clusters=4, algorithm="elkan", random_state=42) # Added random_state for reproducibility
        X = df[['anglez', 'enmo']]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        labels = kmeans.fit_predict(X_scaled)
        return (labels + 1) / 4
    except Exception as e:
        st.sidebar.warning(f"⚠️ Error during clustering: {e}")
        return np.ones(len(df)) * 0.5

# --- Data Processing ---
def process_input_data(df):
    data = df.copy()
    if 'timestamp' in data.columns:
        try:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        except Exception as e:
            st.sidebar.warning(f"⚠️ Could not convert 'timestamp' to datetime: {e}")
    else:
        st.sidebar.warning("⚠️ Timestamp column not found.")

    data['diff_anglez'] = data['anglez'].diff()
    data = data[(data['enmo'] != 0.0) | (data['diff_anglez'] != 0.0)].copy()
    data.drop('diff_anglez', axis=1, inplace=True)

    data = calculate_rolling_features(data)
    data['cluster'] = perform_clustering(data)
    X = data[['sd_anglez_1', 'sd_enmo_1', 'anglez', 'm_anglez_2', 'm_enmo_2', 'enmo', 'cluster']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, data

# --- Prediction ---
def predict_sleep_state(model, X):
    try:
        if hasattr(model, 'predict'):
            X_copy = X.copy()
            predictions = model.predict(X_copy).astype(int)
            return predictions
        else:
            st.sidebar.warning("⚠️ Model doesn't have predict method.")
            return np.zeros(X.shape[0])
    except Exception as e:
        st.sidebar.warning(f"⚠️ Prediction error: {e}")
        if hasattr(model, 'strategy'):
            return model.predict(X)
        return np.zeros(X.shape[0])

# --- Main App ---
st.title("🌈 Dream Weaver: Sleep State Prediction Portal 🌙")
st.markdown("Unveil your sleep patterns with wearable sensor data.")

with st.sidebar:
    st.header("⚙️ Data Input & Model Selection 🧠")
    data_source = st.radio(
        "Choose your data source:",
        ["Upload CSV File 📂", "Enter Values Manually ✍️", "Use Sample Data 🧪"],
        index=2  # Default to sample data
    )

    input_df = None
    if data_source == "Upload CSV File 📂":
        uploaded_file = st.file_uploader("Upload a CSV file (timestamp, anglez, enmo)", type=["csv"])
        if uploaded_file is not None:
            try:
                input_df = pd.read_csv(uploaded_file)
                if not all(col in input_df.columns for col in ['timestamp', 'anglez', 'enmo']):
                    st.sidebar.error("⚠️ CSV must contain 'timestamp', 'anglez', and 'enmo' columns.")
                    input_df = None
                else:
                    st.sidebar.success("✅ File uploaded successfully!")
                    st.sidebar.dataframe(input_df.head())
            except Exception as e:
                st.sidebar.error(f"❌ Error reading file: {e}")
                input_df = None

    elif data_source == "Enter Values Manually ✍️":
        st.sidebar.subheader("Enter Sensor Readings")
        if 'manual_data' not in st.session_state:
            st.session_state.manual_data = pd.DataFrame(columns=['timestamp', 'anglez', 'enmo'])

        with st.sidebar.form("manual_input_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            date_input = col1.date_input("Date", datetime.date.today())
            time_input = col2.time_input("Time", datetime.time(0, 0))
            angle_z = st.number_input("Angle Z (degrees)", step=1.0)
            enmo_val = st.number_input("ENMO (movement)", min_value=0.0, step=0.01)
            submitted = st.form_submit_button("Add Reading")
            if submitted:
                new_row = pd.DataFrame({'timestamp': [datetime.datetime.combine(date_input, time_input)], 'anglez': [angle_z], 'enmo': [enmo_val]})
                st.session_state.manual_data = pd.concat([st.session_state.manual_data, new_row], ignore_index=True)
                st.sidebar.success("✅ Reading added!")

        if not st.session_state.manual_data.empty:
            st.sidebar.subheader("Current Manual Data")
            st.sidebar.dataframe(st.session_state.manual_data)
            if st.sidebar.button("Clear Manual Data"):
                st.session_state.manual_data = pd.DataFrame(columns=['timestamp', 'anglez', 'enmo'])
                st.sidebar.info("🗑️ Manual data cleared.")
            input_df = st.session_state.manual_data.copy()
            if len(input_df) < 3:
                st.sidebar.warning("⚠️ Please add at least 3 readings for prediction.")
        else:
            input_df = None

    elif data_source == "Use Sample Data 🧪":
        try:
            input_df = pd.read_csv("data/sample_data.csv", usecols=["timestamp", "anglez", "enmo"])
            st.sidebar.info("✅ Using sample data.")
            st.sidebar.dataframe(input_df.head())
        except FileNotFoundError:
            st.sidebar.error("❌ Sample data file not found at 'data/sample_data.csv'.")
            input_df = None
        except Exception as e:
            st.sidebar.error(f"❌ Error loading sample data: {e}")
            input_df = None

    st.subheader("🔮 Model Selection")
    models = load_models()
    models_loaded = models is not None and len(models) > 0
    selected_model = None
    if models_loaded:
        selected_model = st.selectbox(
            "Choose your prediction model:",
            list(models.keys()),
            index=0
        )
    else:
        st.warning("⚠️ No models loaded. Predictions will not be available.")

# --- Prediction Area ---
st.header("✨ Sleep State Prediction Output ✨")

if input_df is not None and models_loaded and selected_model:
    if (data_source == "Enter Values Manually ✍️" and not st.session_state.manual_data.empty and len(st.session_state.manual_data) >= 3) or \
       (data_source != "Enter Values Manually ✍️" and not input_df.empty):
        if st.button("✨ Predict Sleep States ✨"):
            with st.spinner("🧠 Analyzing data and predicting sleep states..."):
                try:
                    X_scaled, processed_df = process_input_data(input_df.copy())
                    predictions = predict_sleep_state(models[selected_model], X_scaled)

                    if predictions is not None:
                        processed_df['sleep_prediction'] = predictions
                        processed_df['sleep_state'] = processed_df['sleep_prediction'].map({0: 'Awake ☀️', 1: 'Asleep 🌙'})

                        st.subheader("📊 Prediction Results")
                        st.dataframe(processed_df[['timestamp', 'anglez', 'enmo', 'sleep_state']].style.set_properties(**{'background-color': '#f5f5dc', 'color': 'black'})) # Beige background

                        sleep_percentage = (predictions == 1).mean() * 100
                        awake_percentage = 100 - sleep_percentage

                        col1, col2 = st.columns(2)
                        col1.metric("😴 Sleep Percentage", f"{sleep_percentage:.1f}%", delta=None, delta_color="normal")
                        col2.metric(" জেগে থাকা Awake Percentage", f"{awake_percentage:.1f}%", delta=None, delta_color="normal")

                        fig = go.Figure(data=[go.Scatter(
                            x=processed_df['timestamp'] if 'timestamp' in processed_df.columns else processed_df.index,
                            y=processed_df['sleep_prediction'],
                            mode='lines+markers',
                            name='Sleep State (0=Awake, 1=Asleep)',
                            line=dict(color='#8a2be2'), # Blue Violet
                            marker=dict(color='#9400d3') # Dark Violet
                        )])
                        fig.update_layout(
                            title="🕰️ Sleep Predictions Over Time",
                            xaxis_title="Time",
                            yaxis_title="Sleep State (0=Awake, 1=Asleep)",
                            yaxis=dict(tickvals=[0, 1], ticktext=["Awake", "Asleep"]),
                            plot_bgcolor='#e6e6fa' # Lavender
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        csv = processed_df.to_csv(index=False)
                        st.download_button(
                            label="💾 Download Predictions as CSV",
                            data=csv,
                            file_name="sleep_predictions.csv",
                            mime="text/csv",
                        )

                    else:
                        st.error("⚠️ Prediction failed. Please check your input data and model.")

                except Exception as e:
                    st.error(f"❌ An error occurred during prediction: {e}")

    elif input_df is None:
        st.info("👆 Please upload your data or enter values in the sidebar to begin.")
    elif data_source == "Enter Values Manually ✍"
