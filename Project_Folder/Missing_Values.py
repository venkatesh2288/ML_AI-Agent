
#!Simple Missing values===================================================



import pandas as pd
import json
import re
from sklearn.impute import SimpleImputer
import numpy as np

def handling_missing_values(data: pd.DataFrame, llm_client) -> pd.DataFrame:
    """
    Applies a full pipeline of missing value handling steps to a DataFrame in-place.

    This function orchestrates the identification and handling of missing values,
    including primary key validation, row/column removal based on missingness
    thresholds, type classification using an LLM, numeric conversion, and imputation.
    It modifies the input DataFrame directly and returns it.

    Args:
        data (pd.DataFrame): The DataFrame to be processed in-place.

    Returns:
        pd.DataFrame: The same DataFrame, now cleaned. Returns the original
                      DataFrame unmodified if a critical error occurs.
    """
    # --- Nested Helper Functions ---

    def find_primary_key(columns, llm_client):
        prompt = f"""
        You are a highly intelligent database architect and data engineer (IQ 400). Your task is to analyze dataset columns and determine their keys.

        You are given a list of column names:
        {columns}

        Your job is to analyze the column names and return a **valid JSON object** with the following format:
        ```json
        {{
            "column_name_1": "primary_key",
            "column_name_2": "key"
        }}
        ```

        Now return your answer as a **valid lowercase JSON object**."""
        
        try:
            response = llm_client.invoke(prompt).content # Use llm_client
            text = response.strip()
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                json_str = match.group(0)

                if not json_str.strip(): # Check if the extracted string is empty or just whitespace
                    raise ValueError("Extracted JSON string is empty or whitespace only.")

                keys_dict = json.loads(json_str)
                if isinstance(keys_dict, dict):
                    return keys_dict
                else:
                    raise ValueError("LLM output parsed but was not a dictionary.")
            
            raise ValueError("No valid JSON object found in LLM output.")

        except (json.JSONDecodeError, ValueError) as e:
            # Error during JSON parsing or validation in find_primary_key
            return {} # Return an empty dict on error to prevent further issues
        except Exception as e:
            # An unexpected error occurred in find_primary_key
            return {}
    
    def remove_rows_with_missing_pk(df, key_info):
        """
        Identifies primary key columns based on LLM output and drops rows where these keys are missing.
        A primary key should be unique and not null.
        Modifies the DataFrame in-place.
        """
        primary_key_cols = [col for col, key_type in key_info.items() if key_type == "primary_key" and col in df.columns]
    
        if not primary_key_cols:
            return
    
        initial_rows = len(df)
        df.dropna(subset=primary_key_cols, inplace=True) # Modifies df in-place
        rows_dropped = initial_rows - len(df) # Use len(df) after in-place modification
    
        if rows_dropped > 0:
            print(f"Dropped {rows_dropped} rows with missing primary keys.")

    def remove_rows_with_many_missing_features(df: pd.DataFrame, missing_threshold: float = 0.7):
        """
        Removes rows that have a high percentage of missing features.
        
        Args:
            df (pd.DataFrame): The input DataFrame.
            missing_threshold (float): The percentage threshold (0.0 to 1.0) of missing columns
                                    above which a row will be removed. Modifies DataFrame in-place.
        """
        # Calculate the minimum number of non-NA values a row must have to be kept
        min_non_na_values = int(len(df.columns) * (1 - missing_threshold))
        
        initial_rows = len(df)
        df.dropna(thresh=min_non_na_values, inplace=True) # Modifies df in-place
        rows_dropped = initial_rows - len(df) # Use len(df) after in-place modification
        
        if rows_dropped > 0:
            print(f"Dropped {rows_dropped} rows with more than {missing_threshold*100:.0f}% missing features.")

    def remove_columns_with_many_missing(df: pd.DataFrame, missing_threshold: float = 0.7):
        """Removes columns that have a high percentage of missing values in-place."""
        missing_percentage = df.isnull().sum() / len(df)
        cols_to_drop = missing_percentage[missing_percentage > missing_threshold].index.tolist()
        
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
            print(f"Dropped columns with more than {missing_threshold*100:.0f}% missing values: {cols_to_drop}")
        else:
            print("No columns found with missing values above the specified threshold.")

    def nature_of_data(columns, llm_client):
        prompt = f"""
        You are a highly intelligent database architect and data engineer (IQ 400). Your task is to classify dataset columns based on their data nature and semantic meaning.

        You are given a list of column names:
        {columns}

        Your job is to analyze the column names and classify each into **exactly one** of the following lowercase types:

        1. "continuous" → Real numbers (e.g., income, height, weight)
        2. "discrete" → Integer counts (e.g., number of visits, children)
        3. "nominal_categorical" → Categorical with no order (e.g., gender, city, product_type)
        4. "ordinal_categorical" → Categorical with order (e.g., rating, education level)
        5. "binary" → Yes/No or 0/1 style columns (e.g., is_active, approved)
        6. "datetime" → Dates or timestamps (e.g., created_at, dob)
        7. "text" → Freeform text (e.g., name, model_id, feedback)
        return strictly only with these 7 types of datatypes. Should not include any other types. 

        Now return your answer as a **valid lowercase JSON object** of this format:
        ```json
        {{
        "column_name_1": "type",
        "column_name_2": "type"
        }}"""
        try:
            response = llm_client.invoke(prompt).content # Use llm_client
            text = response.strip()
            
            # Use regex to reliably find the JSON object, even if it's wrapped in text
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                json_str = match.group(0)
                keys_dict = json.loads(json_str)
                if isinstance(keys_dict, dict):
                    return keys_dict
            
            raise ValueError("No valid JSON object found in LLM output for column classification.")
        except (json.JSONDecodeError, ValueError) as e:
            # Error parsing JSON for column classification
            return {} # Return empty dict on error to prevent crashes
        except Exception as e:
            # An unexpected error occurred in nature_of_data
            return {}

    def classify_columns(types, all_columns):
        """Organizes columns into lists based on their classified type."""
    
        # Define the valid categories and initialize the result dictionary
        categories = [
            "continuous", "discrete", "nominal_categorical", 
            "ordinal_categorical", "binary", "datetime", "text"
        ]
        column_groups = {f"{cat}_columns": [] for cat in categories}
    
        # Handle any columns the LLM may have missed, defaulting them to 'text'
        classified_cols = set(types.keys())
        unclassified_cols = [col for col in all_columns if col not in classified_cols]
        if unclassified_cols:
            for col in unclassified_cols:
                types[col] = 'text'
    
        # Populate the lists based on the classified type
        for column, dtype in types.items():
            key = f"{dtype}_columns"
            if key in column_groups:
                column_groups[key].append(column)
            else:
                column_groups["text_columns"].append(column)
                
        return column_groups

    def convert_numeric_columns(data: pd.DataFrame, classified_columns: dict, error_threshold: float = 0.5):
        """
        Converts columns identified as 'continuous' or 'discrete' to a numeric type.
        Modifies DataFrame in-place by dropping rows that fail conversion (if below threshold).
        """
        # Combine continuous and discrete columns for numeric conversion
        numeric_cols_to_check = classified_columns.get('continuous_columns', []) + classified_columns.get('discrete_columns', [])
        
        for col in numeric_cols_to_check:
            if col not in data.columns:
                continue

            # Only attempt conversion if the column is not already numeric
            if not pd.api.types.is_numeric_dtype(data[col]):

                # Identify rows that would fail conversion (where original is not null but coerced becomes null)
                # --- NEW: Robustly clean strings before converting to numeric ---
                # Create a temporary series to work with, ensuring it's a string type
                temp_series = data[col].astype(str)
                # Handle parentheses used for negative numbers, e.g., (4,533.75) -> -4,533.75
                temp_series = temp_series.str.replace('(', '-', regex=False)
                # Remove all non-numeric characters except for the decimal point and negative sign
                temp_series = temp_series.str.replace(r'[^\d.-]+', '', regex=True)
                # After cleaning, some strings might be empty or just a hyphen. Treat them as missing.
                temp_series.replace(['', '-'], np.nan, inplace=True)
                # --- END NEW ---
                coerced_series = pd.to_numeric(temp_series, errors='coerce')
                error_mask = data[col].notna() & coerced_series.isna() # Compare original non-nulls to new nulls
                error_indices = data.index[error_mask]
                
                original_non_nulls = data[col].notna().sum()

                # Safety Check: If a high percentage of values fail conversion, it's a likely misclassification.
                if original_non_nulls > 0 and (len(error_indices) / original_non_nulls) > error_threshold:
                    # This column was likely misclassified. To prevent errors in later steps (like imputation),
                    # we will re-classify it as 'text' and skip numeric conversion.
                    if col in classified_columns.get('continuous_columns', []):
                        print(f"  - WARNING: High data loss ({len(error_indices)}/{original_non_nulls} rows) for column '{col}'. Reclassifying as 'text'.")
                        classified_columns['continuous_columns'].remove(col)
                        classified_columns.setdefault('text_columns', []).append(col)
                    if col in classified_columns.get('discrete_columns', []):
                        print(f"  - WARNING: High data loss ({len(error_indices)}/{original_non_nulls} rows) for column '{col}'. Reclassifying as 'text'.")
                        classified_columns['discrete_columns'].remove(col)
                        classified_columns.setdefault('text_columns', []).append(col)
                    continue # Skip numeric conversion for this column
                
                # If there are problematic rows below the threshold, remove them.
                if not error_indices.empty:
                    data.drop(error_indices, inplace=True)
                    print(f"  - Dropped {len(error_indices)} rows with non-numeric values in column '{col}'.")
                
                # Apply the cleaned and coerced series to the dataframe
                data[col] = pd.to_numeric(temp_series, errors='coerce')

    def impute_missing_values(data, column_types):
        """
        Imputes missing values for all column types based on the provided classification in-place.
        """
        imputation_strategies = {
            'continuous': 'mean',
            'discrete': 'median',
            'nominal_categorical': 'most_frequent',
            'ordinal_categorical': 'most_frequent',
            'binary': 'most_frequent',
            'datetime': 'most_frequent',
            'text': 'most_frequent'
        }

        for col_type, strategy in imputation_strategies.items():
            # The key in column_types dictionary is like 'continuous_columns'
            columns_to_impute = column_types.get(f"{col_type}_columns", [])
            
            if not columns_to_impute:
                continue

            # Filter for columns that actually exist in the dataframe and have missing values
            cols_with_missing = [col for col in columns_to_impute if col in data.columns and data[col].isnull().any()]

            if not cols_with_missing:
                print(f"No missing values in {col_type} columns. Skipping imputation.")
                continue

            imputer = SimpleImputer(strategy=strategy)
            data[cols_with_missing] = imputer.fit_transform(data[cols_with_missing])

    # --- Main Pipeline Logic ---

    # Step 1: Find primary key using LLM
    print("\n--- Starting Missing Value Handling Pipeline ---")
    print("  Step 1: Identifying primary keys...")
    current_columns = data.columns.tolist()
    key_info_dict = find_primary_key(current_columns, llm_client)
    print(f"Identified primary key info: {key_info_dict}")

    # Step 2: Remove rows with missing primary keys
    print("  Step 2: Removing rows with missing primary keys...")
    remove_rows_with_missing_pk(data, key_info_dict)

    # Step 3: Remove rows with many missing features
    print("  Step 3: Removing rows with many missing features...")
    remove_rows_with_many_missing_features(data)

    # Step 4: Remove columns with many missing values
    print("  Step 4: Removing columns with many missing values...")
    remove_columns_with_many_missing(data)

    # Step 5: Classify nature of data and columns (LLM calls)
    print("  Step 5: Classifying column data types...")
    current_columns = data.columns.tolist() # Update columns list after potential column removal
    column_nature_dict = nature_of_data(current_columns, llm_client)
    column_types = classify_columns(column_nature_dict, current_columns)
    print(f"Column types classified: {column_types}")

    # Step 6: Convert numeric columns
    print("  Step 6: Converting numeric columns...")
    convert_numeric_columns(data, column_types)

    # Step 7: Impute missing values
    print("  Step 7: Imputing remaining missing values...")
    impute_missing_values(data, column_types)

    return data