# app.py

import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
import os
import io
import sys
from contextlib import contextmanager
from dotenv import load_dotenv # <-- Add this import

# Import your existing modules
from Main import full_data_cleaning_pipeline
from AutoML import run_automl_experiment
from visualizer import generate_visualizations

load_dotenv()

# --- Helper Function to Capture Print Statements ---
@contextmanager
def st_capture_stdout():
    """A context manager to capture stdout and display it in a Streamlit container."""
    old_stdout = sys.stdout
    sys.stdout = output_buffer = io.StringIO()
    yield output_buffer
    sys.stdout = old_stdout

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Automated ML Pipeline", initial_sidebar_state="expanded")

# --- Initialize Session State ---
# This helps store variables across user interactions
if 'pipeline_run' not in st.session_state:
    st.session_state.pipeline_run = False
if 'df_cleaned' not in st.session_state:
    st.session_state.df_cleaned = None
if 'llm_client' not in st.session_state:
    st.session_state.llm_client = None

# --- UI: Sidebar for Inputs ---
with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("Enter your credentials and upload your data to start.")

    # # Input for Groq API Key
    # groq_api_key = st.text_input("Groq API Key", type="password", help="Your API key is not stored.")

    # In with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("Upload your data to start.")

    # Check for the API key from the .env file
    if os.getenv("GROQ_API_KEY"):
        st.success("✅ Groq API Key Loaded.")
    else:
        st.error("GROQ_API_KEY not found in .env file.")
        # Stop the app if the key is missing
        st.stop()


    # File Uploader
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    # Start Pipeline Button
    start_button = st.button("🚀 Start AutoML Pipeline", use_container_width=True)

# --- Main Application Logic ---
st.title("🤖 End-to-End Automated Machine Learning")
st.markdown("This application automates the entire ML workflow: data cleaning, feature engineering, model selection, and training.")

if start_button:
    # In if start_button:
    if uploaded_file is None:
        st.error("❌ Please upload a CSV file.")
    else:
        try:
            # --- 1. Initialize LLM Client ---
            with st.spinner("Initializing LLM Client..."):
                # ChatGroq automatically finds the API key in the environment
                st.session_state.llm_client = ChatGroq(model="llama3-70b-8192", temperature=0)
            st.success("✅ LLM Client initialized successfully.")
    # # --- Validation Checks ---
    # if not groq_api_key:
    #     st.error("❌ Please enter your Groq API Key.")
    # elif uploaded_file is None:
    #     st.error("❌ Please upload a CSV file.")
    # else:
    #     try:
    #         # --- 1. Initialize LLM Client ---
    #         with st.spinner("Initializing LLM Client..."):
    #             os.environ["GROQ_API_KEY"] = groq_api_key
    #             st.session_state.llm_client = ChatGroq(model="llama3-70b-8192", temperature=0)
    #         st.success("✅ LLM Client initialized successfully.")

            # --- 2. Load and Display Raw Data ---
            df_raw = pd.read_csv(uploaded_file)
            st.subheader("Initial Raw Data")
            st.dataframe(df_raw.head(), use_container_width=True)

            # --- 3. Run Data Cleaning Pipeline ---
            st.subheader("🧼 Data Cleaning & Preprocessing")
            with st.spinner("The AI is now cleaning your data. This may take a moment..."):
                with st_capture_stdout() as cleaning_logs:
                    df_cleaned = full_data_cleaning_pipeline(df_raw, st.session_state.llm_client)
                
                # Display the logs captured from your backend scripts
                with st.expander("Show Data Cleaning Logs"):
                    st.text(cleaning_logs.getvalue())

            if df_cleaned is not None and not df_cleaned.empty:
                st.session_state.df_cleaned = df_cleaned
                st.session_state.pipeline_run = True
                st.success("✅ Data cleaning complete!")
                st.subheader("Cleaned & Processed DataFrame")
                st.dataframe(st.session_state.df_cleaned.head(), use_container_width=True)
                st.info(f"Data shape after cleaning: {st.session_state.df_cleaned.shape}")
            else:
                st.error("Data cleaning resulted in an empty DataFrame. Cannot proceed.")
                st.session_state.pipeline_run = False

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.session_state.pipeline_run = False

# --- Post-Cleaning Steps (only show if cleaning is done) ---
if st.session_state.pipeline_run:
    st.markdown("---")
    st.subheader("📊 Data Visualization")
    
    if st.button("Generate Visualizations"):
        if st.session_state.df_cleaned is not None:
            with st.spinner("AI is generating chart suggestions and plotting..."):
                generate_visualizations(st.session_state.df_cleaned, st.session_state.llm_client)
        else:
            st.warning("No cleaned data available to visualize.")

    st.markdown("---")
    st.subheader("🧠 Model Training & AutoML")
    st.markdown("Select your target variable and run the automated machine learning experiment.")

    if st.session_state.df_cleaned is not None:
        # --- Target Variable Selection ---
        target_column = st.selectbox(
            "Select the Target Variable (what you want to predict)",
            options=st.session_state.df_cleaned.columns
        )

        # app.py

        if st.button(f"🤖 Run AutoML for '{target_column}'"):
            with st.spinner("Running AutoML Experiment... This is the longest step. Please wait."):
                with st_capture_stdout() as automl_logs:
                    # Call the function with the target_column from the selectbox
                    run_automl_experiment(st.session_state.df_cleaned, target_column=target_column)

                st.success("✅ AutoML Experiment Complete!")
                
                with st.expander("Show Full AutoML Run Log", expanded=True):
                    st.text(automl_logs.getvalue())
    else:
        st.warning("No cleaned data available for modeling.")

else:
    st.info("⬆️ Upload your data and click 'Start AutoML Pipeline' in the sidebar to begin.")