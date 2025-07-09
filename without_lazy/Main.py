import pandas as pd
from langchain_groq import ChatGroq
from Fix_Duplicates import handle_duplicates
from Handling_dates import handle_dates
from Header_missing import clean_dataframe_headers
from Missing_Values import handling_missing_values
from Unit_Conversion import handle_all_inconsistencies
from Encoding import handle_encoding # Import the new encoding handler

def full_data_cleaning_pipeline(data: pd.DataFrame, llm_client):
    """
    Orchestrates the complete data cleaning pipeline from headers to duplicates.

    Args:
        data (pd.DataFrame): The raw DataFrame to be cleaned.
        llm_client: The initialized LLM client (e.g., ChatGroq instance).

    Returns:
        pd.DataFrame: The fully cleaned DataFrame.
    """
    if not isinstance(data, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame.")
        return None

    print(f"Initial DataFrame shape: {data.shape}")

    #! Step 1: Clean column headers
    print("\n--- Step 1: Cleaning Headers ---")
    data = clean_dataframe_headers(data, llm_client)
    print(f"  Headers cleaned. Current DataFrame shape: {data.shape}")

    #! Step 2: Handle all data inconsistencies (numeric, text, units)
    print("\n--- Step 2: Handling Data Inconsistencies ---")
    initial_rows_step2 = len(data)
    data = handle_all_inconsistencies(data, llm_client)
    print(f"  Inconsistencies handled. Rows changed: {initial_rows_step2 - len(data)}. Current DataFrame shape: {data.shape}")

    #! Step 3: Handle date columns
    print("\n--- Step 3: Handling Date Columns ---")
    initial_rows_step3 = len(data)
    data, _ = handle_dates(data, llm_client, dayfirst=False)
    print(f"  Date columns standardized. Rows changed: {initial_rows_step3 - len(data)}. Current DataFrame shape: {data.shape}")

    #! Step 4: Handle missing values
    print("\n--- Step 4: Handling Missing Values ---")
    initial_rows_step4 = len(data)
    data = handling_missing_values(data, llm_client)
    print(f"  Missing values handled. Rows changed: {initial_rows_step4 - len(data)}. Current DataFrame shape: {data.shape}")

    #! Step 5: Handle duplicate rows
    print("\n--- Step 5: Handling Duplicate Rows ---")
    data = handle_duplicates(data, llm_client)
    print(f"  Duplicate rows handled. Current DataFrame shape: {data.shape}")

    #! Step 6: Handle feature encoding
    print("\n--- Step 6: Handling Feature Encoding ---")
    data = handle_encoding(data, llm_client)
    print(f"  Feature encoding handled. Current DataFrame shape: {data.shape}")

    return data