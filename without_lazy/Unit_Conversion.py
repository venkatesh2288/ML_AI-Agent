import re
import pandas as pd
import json # Keep json import as it's used in nested functions
import numpy as np
import unicodedata # Keep unicodedata import as it's used in clean_text_columns

# Assuming llama_run and llm are imported from a central utility or passed as arguments
# For this file, we'll assume llm is passed, and llama_run is available if needed by sub-functions.

# --- Basic Inconsistency Handling ---

def numeric_format_inconsistency(data, columns):
    """
    Cleans and fixes mixed numeric data types in specified columns. This is a robust
    version that handles currency symbols ($), commas, parentheses for negatives, and whitespace.
    It converts columns that are mostly numeric to a numeric type.
    """
    # Work on a copy to avoid modifying the original DataFrame in place
    data_cleaned = data.copy()
    modified_cols = []

    # We only want to try this on non-numeric (object) columns
    for col in data_cleaned.select_dtypes(include=['object']).columns:
        if col not in columns: # If a subset of columns was passed, respect it
            continue

        original_series = data_cleaned[col]
        
        # Use robust cleaning logic to handle currency, commas, and parentheses
        # This logic is designed to turn things like " $(1,234.50) " into -1234.50
        temp_series = original_series.astype(str)
        temp_series = temp_series.str.replace('(', '-', regex=False)
        temp_series = temp_series.str.replace(r'[^\d.-]+', '', regex=True)
        temp_series.replace(['', '-'], np.nan, inplace=True)
        
        numeric_series = pd.to_numeric(temp_series, errors='coerce')

        # Heuristic: If conversion is successful for a high percentage of non-null values,
        # we commit the change. Otherwise, we assume it's a text column with some numbers.
        original_non_nulls = original_series.notna().sum()
        converted_non_nulls = numeric_series.notna().sum()

        # Use a threshold (e.g., 80%) to decide if the column is fundamentally numeric
        if original_non_nulls > 0 and (converted_non_nulls / original_non_nulls) > 0.8:
            if not numeric_series.equals(original_series):
                data_cleaned[col] = numeric_series
                modified_cols.append(col)
                print(f"    - Robustly converted column '{col}' to numeric.")

    return data_cleaned, modified_cols

def handle_unit_conversion(df, llm):
    """
    Detects, groups, and standardizes columns with varying units for the same concept.
    This function orchestrates the unit standardization process and returns a new DataFrame.
    """

    def detect_and_group_unit_columns(df_inner, llm_inner):
        """
        Uses an LLM to identify and group columns with unit inconsistencies.
        """
        columns = df_inner.columns.tolist()
        prompt = f"""
        You are an expert data scientist. Analyze the following list of column names and group them by the concept they measure if they seem to have different units (e.g., weight, distance, currency).

        Column Names: {columns}

        Return a JSON object where each key is a common concept (e.g., "weight") and the value is a list of the column names that belong to that concept.

        Example Output:
        {{
            "weight": ["weight_kg", "weight_lbs"],
            "price": ["price_in_usd", "price_in_eur"],
            "distance": ["distance_km", "distance_miles"]
        }}

        If no such groups are found, return an empty JSON object {{}}.
        Provide only the JSON object and nothing else.
        """
        try:
            response = llm_inner.invoke(prompt)
            content = response.content
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            print("Warning: Could not find a valid JSON object in the LLM response for unit grouping.")
            return {}
        except (json.JSONDecodeError, Exception) as e:
            print(f"An error occurred during unit grouping detection: {e}")
            return {}

    def standardize_units(df_inner, column_groups, llm_inner):
        """
        Standardizes units by analyzing the data within columns, generating conversion
        rules with an LLM, and applying them to create a single standardized column.
        """
        df_standardized = df_inner.copy()

        for concept, columns in column_groups.items():
            if len(columns) < 2:
                continue

            print(f"\n--- Standardizing units for concept: '{concept}' ---")

            sample_values = []
            for col in columns:
                if col in df_standardized.columns:
                    if pd.api.types.is_object_dtype(df_standardized[col]) or pd.api.types.is_string_dtype(df_standardized[col]):
                        unique_vals = df_standardized[col].dropna().unique()
                        sample_values.extend(unique_vals[:50])

            sample_values = list(set(sample_values))
            if not sample_values:
                print(f"Warning: No string data found to analyze for concept '{concept}'. Skipping.")
                continue

            prompt = f"""
            You are an expert data scientist and physicist specializing in parsing and converting units of measurement from raw text.
            Analyze the following sample values from columns related to the concept '{concept}':
            Sample Values: {sample_values}

            Your task is to create a set of rules to parse these values and convert them to a single, standard scientific unit (e.g., 'kg' for mass, 'm' for distance, 'USD' for currency).

            Please return a single, valid JSON object with the following structure:
            {{
                "standard_unit": "your_chosen_standard_unit",
                "conversion_rules": [
                    {{
                        "unit_name": "e.g., kilograms",
                        "regex_pattern": "A Python regex to extract the numeric value. It MUST contain one capturing group for the number.",
                        "conversion_factor": "The number to multiply the extracted value by to get the standard unit."
                    }}
                ]
            }}

            - The regex pattern must be a valid Python regex string (use double backslashes for escaping).
            - The regex must contain exactly one capturing group `()` for the numeric part.
            - Provide only the JSON object, with no surrounding text.
            """
            try:
                response = llm_inner.invoke(prompt)
                content = response.content
                match = re.search(r'\{.*\}', content, re.DOTALL)
                if not match:
                    print(f"Warning: Could not get conversion rules from LLM for concept '{concept}'. Skipping.")
                    continue
                
                conversion_data = json.loads(match.group(0))
                rules = conversion_data.get("conversion_rules", [])
                standard_unit = conversion_data.get("standard_unit", concept)

                if not rules:
                    print(f"Warning: LLM did not provide any conversion rules for '{concept}'. Skipping.")
                    continue

                print(f"Standardizing to '{standard_unit}'. Applying {len(rules)} conversion rules.")

                def apply_conversion(value):
                    if pd.isna(value): return None
                    str_value = str(value).lower().strip()
                    for rule in rules:
                        try:
                            match = re.search(rule['regex_pattern'], str_value)
                            if match:
                                numeric_part = float(match.group(1))
                                factor = float(rule['conversion_factor'])
                                return numeric_part * factor
                        except (re.error, IndexError, TypeError, KeyError): continue
                    try: return float(str_value)
                    except (ValueError, TypeError): return None

                new_col_name = f"{concept}_{standard_unit}"
                converted_series_list = [df_standardized[col].apply(apply_conversion) for col in columns if col in df_standardized.columns]

                if converted_series_list:
                    final_series = converted_series_list[0]
                    for next_series in converted_series_list[1:]:
                        final_series = final_series.fillna(next_series)
                    df_standardized[new_col_name] = final_series
                    # Instead of dropping NaNs, fill with a placeholder or 0, or consider imputation
                    df_standardized[new_col_name] = df_standardized[new_col_name].fillna(0)  # Example: Fill with 0
                
                cols_to_drop = [col for col in columns if col in df_standardized.columns]
                df_standardized.drop(columns=cols_to_drop, inplace=True)
                print(f"Created standardized column '{new_col_name}' and dropped original columns: {cols_to_drop}")

            except (json.JSONDecodeError, Exception) as e:
                print(f"An error occurred while standardizing units for '{concept}': {e}. Skipping this concept.")
                continue
                
        return df_standardized

    # --- Main logic for handle_unit_conversion ---
    # This function creates a copy internally, as per its current implementation.
    print("\n--- Detecting and Standardizing Units ---")
    unit_groups = detect_and_group_unit_columns(df, llm)
    if unit_groups:
        print(f"Found potential unit groups: {unit_groups}")
        cleaned_df = standardize_units(df, unit_groups, llm)
        return cleaned_df
    else:
        print("No unit groups detected for standardization.")
        return df.copy()

# --- Text Cleaning ---

def clean_text_columns(df):
    """
    Cleans and standardizes all text-based columns (object or category types).
    This function does not modify numeric or datetime columns.
    """
    cleaned_df = df.copy()

    for col in cleaned_df.select_dtypes(include=["object", "category"]).columns:
        def clean_text(x):
            if pd.isna(x):  # Handle NaN values directly
                return x
            original_x = x  # Keep original value to return on error
            try:
                # Convert to string
                x = str(x)

                # Fix encoding issues (best effort)
                x = x.encode('latin1', errors='ignore').decode('utf-8', errors='ignore')
                # Normalize unicode (e.g., é, ñ)
                x = unicodedata.normalize("NFKD", x)
                # Remove emojis / non-ASCII characters
                x = re.sub(r'[^\x00-\x7F]+', '', x)
                # Remove leading/trailing whitespace
                x = x.strip()
                # Collapse multiple spaces into one
                x = re.sub(r'\s+', ' ', x)
                # Normalize case
                x = x.lower()

                return x
            except Exception as e:
                # Log the error for debugging, but return original value to avoid dropping all rows
                # print(f"Warning: Failed to clean text '{original_x}' in column '{col}' due to error: {e}. Keeping original value.") # Uncomment for debugging
                return original_x  # Return original value instead of pd.NA

        cleaned_df[col] = cleaned_df[col].apply(clean_text)

    return cleaned_df

# --- Orchestrator for all Inconsistency Handling ---

def handle_all_inconsistencies(df: pd.DataFrame, llm) -> pd.DataFrame:
    """
    Orchestrates all inconsistency handling steps within the DataFrame.
    This includes numeric format cleaning, text cleaning, and unit standardization.

    Args:
        df (pd.DataFrame): The input DataFrame to be processed.
        llm: The LLM instance required for unit conversion.

    Returns:
        pd.DataFrame: A new, cleaned DataFrame.
    """
    print("\n--- Starting Inconsistency Handling Pipeline ---")
    # Work on a copy to avoid side effects
    df_processed = df.copy()

    # Step 1: Clean numeric formats.
    print("  - Cleaning numeric formats...")
    df_processed, _ = numeric_format_inconsistency(df_processed, df_processed.columns.tolist())

    # Step 2: Clean text columns.
    print("  - Cleaning text columns...")
    df_processed = clean_text_columns(df_processed)

    # Step 3: Standardize units.
    print("  - Standardizing units...")
    df_processed = handle_unit_conversion(df_processed, llm)
    
    print("--- Inconsistency Handling Pipeline Completed ---")
    return df_processed
