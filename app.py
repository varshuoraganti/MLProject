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
            model = load(r"C:\Users\varshini\Downloads\New folder (2)\New folder (2)\data\final_series_df.csv")
            models[model_name] = model
            loaded_models.append(model_name)
            st.success(f"✓ Successfully loaded {model_name} model")
        except Exception as e:
            errors.append(f"Error loading {model_name}: {str(e)}")

    if not models:
        st.error("⚠️ Could not load any models. Using fallback models instead.")
        return create_fallback_models()
    else:
        if errors:
            st.warning(f"⚠️ Some models could not be loaded. Only using: {', '.join(loaded_models)}")
            for err in errors:
                st.error(err)
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
    data['m_anglez_2'] = np.nan   # 2 min rolling std: anglez

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
                        .rolling(24).mean().reset_index(level=0, drop=True))
        data.loc[data['m_anglez_2'].isna(), 'm_anglez_2'] = (
            data.groupby('series_id')['anglez']
            .rolling(2).mean().reset_index(level=0, drop=True)
        )
    except Exception as e:
        st.warning(f"Error calculating rolling statistics: {e}")

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
        st.warning(f"Error in clustering: {e}")
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
            st.warning(f"Could not convert 'timestamp' column to datetime: {e}")
            # Fallback to using index for plotting if conversion fails
            pass
    else:
        st.warning("Timestamp column not found. Using index for time-based plots.")

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
            # Ensure the model is making binary predictions (0 or 1)
            predictions = model.predict(X_copy).astype(int)
            return predictions
        else:
            st.warning("Model doesn't have predict method")
            return np.zeros(X.shape[0])
    except Exception as e:
        st.warning(f"Prediction error: {e}. Using fallback prediction logic.")

        # For simple fallback models
        if hasattr(model, 'strategy'):
            return model.predict(X)

        # Ultimate fallback - just return zeros (awake)
        return np.zeros(X.shape[0])

# Main title and description
st.title("💤 Sleep State Prediction App")
st.markdown("""
This app predicts sleep states using accelerometer data from wearable devices.
Select a model from the dropdown and either upload your data or use the provided sample data.
""")

# Load models
models = load_models()
models_loaded = models is not None and len(models) > 0

# Model selection dropdown
if models_loaded:
    st.sidebar.header("Model Selection")
    selected_model = st.sidebar.selectbox(
        "Choose a model for prediction",
        list(models.keys()),
        index=1  # Default to Tuned XGBoost
    )

    # Model info
    st.sidebar.subheader("Model Information")
    model_description = {
        "XGBoost": "Standard XGBoost model with good performance.",
        "Tuned XGBoost": "XGBoost with optimized hyperparameters for best accuracy.",
        "XGBoost (Fallback)": "Fallback model simulating XGBoost predictions.",
        "Tuned XGBoost (Fallback)": "Fallback model simulating tuned XGBoost predictions."
    }
    st.sidebar.info(model_description.get(selected_model, "Model information not available"))

    # Feature importance
    st.sidebar.subheader("Key Features")
    st.sidebar.write("- sd_anglez_1: Standard deviation of arm angle (1 min window)")
    st.sidebar.write("- sd_enmo_1: Standard deviation of movement (1 min window)")
    st.sidebar.write("- m_anglez_2: Mean arm angle (2 min window)")
    st.sidebar.write("- m_enmo_2: Mean movement (2 min window)")
    st.sidebar.write("- cluster: Movement pattern cluster (0.25-1.0)")
else:
    st.error("No models could be loaded. Please check that the model files exist.")

# Data input section
st.header("Input Data")

input_method = st.radio(
    "Choose input method:",
    ["Upload CSV file", "Enter values manually", "Use sample data"]
)

if input_method == "Upload CSV file":
    uploaded_file = st.file_uploader("Upload a CSV file with columns 'timestamp', 'anglez', and 'enmo'", type=["csv"])
    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            if not all(col in input_df.columns for col in ['timestamp', 'anglez', 'enmo']):
                st.error("CSV must contain columns 'timestamp', 'anglez', and 'enmo'")
            else:
                st.success("File uploaded successfully!")
                st.dataframe(input_df.head())

                # Process data for prediction
                if st.button("Make Prediction"):
                    with st.spinner("Processing data and making predictions..."):
                        X_scaled, processed_df = process_input_data(input_df.copy()) # Use .copy() to avoid potential issues
                        predictions = predict_sleep_state(models[selected_model], X_scaled)

                        if predictions is not None:
                            processed_df['sleep_prediction'] = predictions

                            # Display results
                            st.header("Prediction Results")
                            st.write(f"Model used: {selected_model}")

                            # Calculate sleep percentage
                            sleep_percentage = (predictions == 1).mean() * 100
                            awake_percentage = 100 - sleep_percentage

                            col1, col2 = st.columns(2)

                            # Display metrics
                            col1.metric("Sleep Percentage", f"{sleep_percentage:.1f}%")
                            col2.metric("Awake Percentage", f"{awake_percentage:.1f}%")

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

                            # Show data table with predictions
                            st.subheader("Processed Data with Predictions")
                            st.dataframe(processed_df)

                            # Option to download results
                            csv = processed_df.to_csv(index=False)
                            st.download_button(
                                label="Download predictions as CSV",
                                data=csv,
                                file_name="sleep_predictions.csv",
                                mime="text/csv",
                            )
        except Exception as e:
            st.error(f"Error reading file: {e}")

elif input_method == "Enter values manually":
    st.subheader("Enter Accelerometer Data")
    st.write("Enter at least a few data points with timestamps for prediction")

    # Create empty dataframe
    if 'manual_data' not in st.session_state:
        st.session_state.manual_data = pd.DataFrame(columns=['timestamp', 'anglez', 'enmo'])

    # Form for entering new data points
    with st.form("data_form"):
        col1, col2, col3 = st.columns(3)
        date_input = col1.date_input("Date", datetime.datetime.now().date())
        time_input = col1.time_input("Time", datetime.datetime.now().time())
        anglez = col2.number_input("Angle Z (degrees)", value=0.0, step=1.0)
        enmo = col3.number_input("ENMO (movement)", value=0.0, min_value=0.0, step=0.01)

        # Add the form submit button
        submit_button = st.form_submit_button(label="Add Data Point")

    # Process form submission (outside the form)
    if submit_button:
        # Create timestamp by combining date and time
        timestamp = datetime.datetime.combine(date_input, time_input)
        new_data = pd.DataFrame({
            'timestamp': [timestamp],
            'anglez': [anglez],
            'enmo': [enmo]
        })
        st.session_state.manual_data = pd.concat([st.session_state.manual_data, new_data], ignore_index=True)
        st.success("Data point added!")

    # Display and manage current data
    if not st.session_state.manual_data.empty:
        st.subheader("Current Data")
        st.dataframe(st.session_state.manual_data)

        col1, col2 = st.columns(2)
        if col1.button("Clear All Data"):
            st.session_state.manual_data = pd.DataFrame(columns=['timestamp', 'anglez', 'enmo'])
            st.success("Data cleared!")

        if models_loaded and col2.button("Make Prediction"):
            if len(st.session_state.manual_data) < 3:
                st.warning("Please add at least 3 data points for prediction")
            else:
                with st.spinner("Processing data and making predictions..."):
                    input_df = st.session_state.manual_data.copy()
                    # Calculate the missing features for manual input
                    processed_df_manual = calculate_rolling_features(input_df.copy())
                    processed_df_manual['cluster'] = perform_clustering(input_df.copy())
