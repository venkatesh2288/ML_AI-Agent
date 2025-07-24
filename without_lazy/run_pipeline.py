import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from Main import full_data_cleaning_pipeline
from AutoML import run_automl_experiment
from visualizer import generate_visualizations
# --- Application Entry Point ---

def main():
    """
    This is the main function that sets up the environment, loads data,
    initializes the LLM client, and runs the cleaning pipeline.
    """
    # Step 1: Load environment variables from .env file (for GROQ_API_KEY)
    load_dotenv()

    # Step 2: Initialize the LLM Client (This is the "backend" for llm_client)
    # This object holds the connection details and configuration for the Groq API.
    try:
        llm_client = ChatGroq(model="llama3-70b-8192", temperature=0)
        print("LLM Client initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize LLM Client: {e}")
        return

    # Step 3: Load your raw data into a pandas DataFrame
    # Replace 'your_raw_data.csv' with the path to your actual data file.
    # For demonstration, we'll create a sample DataFrame with various issues.
    # Using 'Financials.csv' to match the user's context.
    df_raw = pd.read_csv(r'C:\Users\prabh\OneDrive\desktop\AML\without_lazy\Datasets\bank_data.csv')

    print("\n--- 🚀 Starting Data Cleaning Pipeline with Sample DataFrame ---")
    print("Initial Raw DataFrame:")
    print(df_raw)
    print("\nInitial Raw DataFrame Info:")
    df_raw.info(verbose=False)

    # Step 4: Run the full data cleaning pipeline
    df_cleaned = full_data_cleaning_pipeline(df_raw, llm_client)

    print("\n--- Pipeline Finished ---")
    if df_cleaned is not None and not df_cleaned.empty:
        print("Final Cleaned DataFrame:")
        print(df_cleaned)
        print("\nFinal Cleaned DataFrame Info:")
        df_cleaned.info(verbose=False)        

        # --- New Visualization Step ---
        while True:
            choice = input("\nWould you like to generate data visualizations before modeling? (y/n): ").lower().strip()
            if choice in ['y', 'n']:
                break
            print("Invalid input. Please enter 'y' or 'n'.")
        
        if choice == 'y':
            generate_visualizations(df_cleaned, llm_client)

        # --- Step 5: Run AutoML to find the best model ---
        run_automl_experiment(df_cleaned)
    else:
        print("\n Data cleaning resulted in an empty DataFrame. Skipping AutoML.")


if __name__ == "__main__":
    main()