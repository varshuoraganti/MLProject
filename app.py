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

# Set page configuration
st.set_page_config(page_title="Sleep State Predictor", page_icon="💤", layout="wide")

# Load XGBoost models only
@st.cache_resource
def load_models():
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    models = {}
    model_files = {
        "XGBoost": 'Xgb_model.joblib',
        "Tuned XGBoost": 'XGB_tunned_model.joblib'
    }

    loaded_models = []
    errors = []

    for model_name, file_path in model_files.items():
        try:
            model = load(file_path)
            models[model_name] = model
            loaded_models.append(model_name)
            st.sidebar.success(f"✓ Loaded {model_name} model")
        except Exception as e:
            errors.append(f"Error loading {model_name}: {str(e)}")
            st.sidebar.error(f"Error loading {model_name}: {str(e)}")

    if not models:
        st.sidebar.error("⚠️ Could not load any models. Using fallback models instead.")
        return create_fallback_models()
    else:
        if errors:
            st.sidebar.warning(f"⚠️ Some models could not be loaded. Only using: {', '.join(loaded_models)}")
        return models

# Fallback model implementation
def create_fallback_models():
    """Creates simple models when joblib models can't be loaded"""
    class SimpleModel:
        def __init__(self, name, prediction_strategy="pattern"):
            self.name = name
            self.strategy = prediction_strategy

        def predict(self, X):
            if self.strategy == "random":
                # Random predictions with sleep ~30% of the time
                return np.random.choice([0, 1], size=X.shape[0], p=[0.7, 0.3])
            elif self.strategy == "pattern":
                # Create a pattern that resembles sleep cycles
                n_samples = X.shape[0]
                predictions = np.zeros(n_samples)

                # If we have enough samples, create a sleep-wake pattern
                if n_samples > 10:
                    # Mark middle portion as sleep for demonstration
                    sleep_start = n_samples // 4
                    sleep_end = (n_samples * 3) // 4
                    predictions[sleep_start:sleep_end] = 1

                return predictions
            else:
                # Default to all awake
                return np.zeros(X.shape[0])

    models = {
        "XGBoost (Fallback)": SimpleModel("XGBoost", "pattern"),
        "Tuned XGBoost (Fallback)": SimpleModel("Tuned XGBoost", "pattern")
    }

    return models

# Function to calculate rolling statistics
def calculate_rolling_features(df):
    # Create a copy to avoid modifying the original
    data = df.copy()

    # Add series_id if it doesn't exist (for groupby operations)
    if 'series_id' not in data.columns:
        data['series_id'] = 1

    # Create rolling features
    data['sd_enmo_1'] = np.nan    # 1 min rolling std: enmo
    data['sd_anglez_1'] = np.nan  # 1 min rolling std: anglez
    data['m_enmo_2'] = np.nan     # 2 min rolling mean: enmo
    data['m_anglez_2'] = np.nan   # 2 min rolling mean: anglez #Changed to mean

    # Calculate rolling statistics with handling of edge cases
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

        # For mean values
        data['m_enmo_2'] = (data.groupby('series_id')['enmo']
                      .rolling(24).mean().reset_index(level=0, drop=True))
        data.loc[data['m_enmo_2'].isna(), 'm_enmo_2'] = (
            data.groupby('series_id')['enmo']
            .rolling(2).mean().reset_index(level=0, drop=True)
        )

        data['m_anglez_2'] = (data.groupby('series_id')['anglez']
                        .rolling(24).mean().reset_index(level=0, drop=True)) #changed to mean
        data.loc[data['m_anglez_2'].isna(), 'm_anglez_2'] = (
            data.groupby('series_id')['anglez']
            .rolling(2).mean().reset_index(level=0, drop=True)
        )
    except Exception as e:
        st.sidebar.warning(f"Error calculating rolling statistics: {e}")

    # Fill remaining NaN values
    data['sd_enmo_1'].fillna(0.0, inplace=True)
    data['sd_anglez_1'].fillna(0.0, inplace=True)
    data['m_enmo_2'].fillna(0.0, inplace=True)
    data['m_anglez_2'].fillna(0.0, inplace=True)

    return data

# Function to perform clustering
def perform_clustering(df):
    try:
        # Use K-means clustering with 4 clusters
        kmeans = KMeans(n_clusters=4, algorithm="elkan")
        # Scale the data for clustering
        X = df[['anglez', 'enmo']]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # Fit the model and get labels
        labels = kmeans.fit_predict(X_scaled)
        # Convert to 0.25, 0.5, 0.75, 1.0 format as in the notebook
        return (labels + 1) / 4
    except Exception as e:
        st.sidebar.warning(f"Error in clustering: {e}")
        # Return a default value if clustering fails
        return np.ones(len(df)) * 0.5

# Function to process input data
def process_input_data(df):
    # Create a copy to avoid modifying the original
    data = df.copy()

    # Ensure 'timestamp' column exists and is the index if needed for plotting
    if 'timestamp' in data.columns:
        try:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        except Exception as e:
            st.sidebar.warning(f"Could not convert 'timestamp' column to datetime: {e}")
            # Fallback to using index for plotting if conversion fails
            pass
    else:
        st.sidebar.warning("Timestamp column not found. Using index for time-based plots.")

    # Remove inactive periods (where both enmo and anglez don't change)
    data['diff_anglez'] = data['anglez'].diff()
    data = data[(data['enmo'] != 0.0) | (data['diff_anglez'] != 0.0)].copy() # Use .copy() to avoid SettingWithCopyWarning
    data.drop('diff_anglez', axis=1, inplace=True)

    # Calculate rolling features
    data = calculate_rolling_features(data)

    # Perform clustering and add as a feature
    data['cluster'] = perform_clustering(data)

    # Extract final features
    X = data[['sd_anglez_1', 'sd_enmo_1', 'anglez', 'm_anglez_2', 'm_enmo_2', 'enmo', 'cluster']]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, data

# Function to make predictions
def predict_sleep_state(model, X):
    try:
        # For XGBoost models
        if hasattr(model, 'predict'):
            # The key fix: Make a copy of the data to prevent modification issues
            X_copy = X.copy()

            # Check the shape of input data
            st.sidebar.write(f"Input data shape for prediction: {X_copy.shape}")

            # Ensure the model is making binary predictions (0 or 1)
            predictions = model.predict(X_copy).astype(int)
            return predictions
        else:
            st.sidebar.warning("Model doesn't have predict method")
            return np.zeros(X.shape[0])
    except Exception as e:
        st.sidebar.warning(f"Prediction error: {e}. Using fallback prediction logic.")

        # For simple fallback models
        if hasattr(model, 'strategy'):
            return model.predict(X)

        # Ultimate fallback - just return zeros (awake)
        return np.zeros(X.shape[0])

# Main title and description
st.title("💤 Sleep State Prediction App")
st.markdown("""
This app predicts sleep states using accelerometer data from wearable devices.
Configure your data and model in the sidebar.
""")

# Sidebar for data input and model selection
with st.sidebar:
    st.header("Data Input")
    input_method = st.radio(
        "Choose input method:",
        ["Upload CSV file", "Enter values manually", "Use sample data"]
    )

    input_df = None
    if input_method == "Upload CSV file":
        uploaded_file = st.file_uploader("Upload a CSV file with columns 'timestamp', 'anglez', and 'enmo'", type=["csv"])
        if uploaded_file is not None:
            try:
                input_df = pd.read_csv(uploaded_file)
                if not all(col in input_df.columns for col in ['timestamp', 'anglez', 'enmo']):
                    st.sidebar.error("CSV must contain columns 'timestamp', 'anglez', and 'enmo'")
                    input_df = None
                else:
                    st.sidebar.success("File uploaded successfully!")
                    st.sidebar.dataframe(input_df.head())
            except Exception as e:
                st.sidebar.error(f"Error reading file: {e}")
                input_df = None

    elif input_method == "Enter values manually":
        st.sidebar.subheader("Enter Accelerometer Data")
        st.sidebar.write("Enter at least a few data points with timestamps for prediction")

        manual_data = []
        with st.sidebar.form("data_form"):
            col1, col2, col3 = st.columns(3)
            date_input = col1.date_input("Date", datetime.datetime.now().date())
            time_input = col1.time_input("Time", datetime.datetime.now().time())
            anglez = col2.number_input("Angle Z (degrees)", value=0.0, step=1.0)
            enmo = col3.number_input("ENMO (movement)", value=0.0, min_value=0.0, step=0.01)
            add_point = st.form_submit_button(label="Add Data Point")

        if add_point:
            timestamp = datetime.datetime.combine(date_input, time_input)
            manual_data.append({'timestamp': timestamp, 'anglez': anglez, 'enmo': enmo})

        if manual_data:
            input_df = pd.DataFrame(manual_data)
            st.sidebar.dataframe(input_df)
            if len(input_df) < 3:
                st.sidebar.warning("Please add at least 3 data points for prediction.")

    elif input_method == "Use sample data":
        @st.cache_data
        def load_sample_data():
            try:
                return pd.read_csv(r"C:\Users\varshini\Downloads\New folder (2)\New folder (2)\data\sample_data.csv", usecols=["timestamp", "anglez", "enmo"])
            except Exception as e:
                st.sidebar.error(f"Error loading sample data: {e}")
                return None

        input_df = load_sample_data()
        if input_df is not None:
            st.sidebar.info("Using sample data.")
            st.sidebar.dataframe(input_df.head())

    st.header("Model Selection")
    models = load_models()
    models_loaded = models is not None and len(models) > 0
    if models_loaded:
        selected_model = st.selectbox(
            "Choose a model for prediction",
            list(models.keys()),
            index=1  # Default to Tuned XGBoost
        )
    else:
        selected_model = None

# Main area for predictions
st.header("Prediction Results")

if input_df is not None and models_loaded and selected_model:
    if (input_method == "Enter values manually" and len(st.session_state.manual_data) >= 3) or (input_method != "Enter values manually" and not input_df.empty):
        if st.button("Make Prediction"):
            with st.spinner("Processing data and making predictions..."):
                X_scaled, processed_df = process_input_data(input_df.copy())
                predictions = predict_sleep_state(models[selected_model], X_scaled)

                if predictions is not None:
                    processed_df['sleep_prediction'] = predictions

                    st.subheader("Predictions")
                    st.dataframe(processed_df[['timestamp', 'anglez', 'enmo', 'sleep_prediction']])

                    # Calculate sleep percentage
                    sleep_percentage = (predictions == 1).mean() * 100
                    awake_percentage = 100 - sleep_percentage
                    st.metric("Sleep Percentage", f"{sleep_percentage:.1f}%")
                    st.metric("Awake Percentage", f"{awake_percentage:.1f}%")

                    # Plot predictions over time
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=processed_df['timestamp'] if 'timestamp' in processed_df.columns else processed_df.index,
                        y=processed_df['sleep_prediction'],
                        mode='lines',
                        name='Sleep State (1=Sleep, 0=Awake)'
                    ))
                    fig.update_layout(
                        title="Sleep Predictions Over Time",
                        xaxis_title="Time",
                        yaxis_title="Sleep State",
                        yaxis=dict(tickvals=[0, 1], ticktext=["Awake", "Sleep"])
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Option to download results
                    csv = processed_df.to_csv(index=False)
                    st.download_button(
                        label="Download predictions as CSV",
                        data=csv,
                        file_name="sleep_predictions.csv",
                        mime="text/csv",
                    )
                else:
                    st.error("Prediction failed.")
    elif input_method == "Enter values manually" and (not hasattr(st.session_state, 'manual_data') or len(st.session_state.manual_data) < 3):
        st.warning("Please enter at least 3 data points for manual prediction.")
    elif input_df is None and input_method == "Use sample data":
        st.info("Loading sample data...")
    else:
        st.info("Please upload your data or enter values manually in the sidebar to see predictions.")
elif not models_loaded:
    st.error("Models could not be loaded. Please ensure the model files are in the correct directory.")
else:
    st.info("Please input your data and select a model in the sidebar to see predictions.")
