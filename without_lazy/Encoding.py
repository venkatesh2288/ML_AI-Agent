import pandas as pd
import json
import re
from sklearn.preprocessing import LabelEncoder

def get_encoding_strategy(data: pd.DataFrame, llm_client):
    """
    Uses an LLM to determine the appropriate encoding strategy for categorical columns.

    Args:
        data (pd.DataFrame): The input DataFrame.
        llm_client: The initialized LLM client.

    Returns:
        dict: A dictionary specifying which columns to encode with which method.
    """
    # Select only non-numeric columns for the LLM to analyze
    candidate_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

    if not candidate_cols:
        print("    - No categorical columns found to analyze for encoding.")
        return {}

    # Create a detailed summary of each candidate column for the LLM
    col_summary = []
    for col in candidate_cols:
        unique_count = data[col].nunique()
        examples = data[col].dropna().unique()[:5].tolist()
        col_summary.append(f"- `{col}` (Cardinality: {unique_count}, Examples: {examples})")
    summary_str = "\n".join(col_summary)

    sample_data = data.head().to_string()

    prompt = f"""
    You are an expert data scientist specializing in feature engineering with an IQ of 999.
    Your task is to determine the best encoding strategy for categorical columns in a dataset.

    Here is a summary of the categorical columns that may need encoding:
    {summary_str}

    Here is a sample of the full dataset to provide additional context:
    {sample_data}

    Based on the column names and their data, classify which columns should be encoded and what strategy to use.
    Choose from the following encoding strategies:
    - "label_encoding": Use for ordinal categorical variables or binary variables.
    - "one_hot_encoding": Use for nominal categorical variables with low to medium cardinality (e.g., less than 15 unique values).
    - "frequency_encoding": Use for nominal categorical variables with high cardinality.

    Only include columns in your response that are suitable for one of these encoding types. Do not include columns that are already numeric, are unique IDs, or represent free text that should not be encoded.

    Return your answer as a single, valid JSON object where keys are the encoding type and values are lists of column names.
    Example format:
    ```json
    {{
      "label_encoding": ["education_level", "is_active"],
      "one_hot_encoding": ["city", "department"],
      "frequency_encoding": ["product_id"]
    }}
    ```
    If no columns require encoding, return a JSON object with empty lists.
    Provide only the JSON object in your response.
    """

    try:
        response_text = llm_client.invoke(prompt).content
        # Use regex to reliably find the JSON object
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            encoding_plan = json.loads(match.group(0))
            # Basic validation of the plan structure
            if isinstance(encoding_plan, dict):
                return encoding_plan

        print("    - Warning: Could not parse a valid JSON encoding plan from the LLM. No encoding will be applied.")
        return {}
    except (json.JSONDecodeError, Exception) as e:
        print(f"    - An error occurred while getting the encoding strategy: {e}")
        return {}

def apply_encoding(data: pd.DataFrame, encoding_plan: dict) -> pd.DataFrame:
    """
    Applies encoding to the DataFrame based on the provided plan.

    Args:
        data (pd.DataFrame): The DataFrame to encode.
        encoding_plan (dict): The plan from the LLM.

    Returns:
        pd.DataFrame: The encoded DataFrame.
    """
    df_encoded = data.copy()

    # 1. Label Encoding
    label_cols = encoding_plan.get("label_encoding", [])
    if label_cols:
        print(f"  - Applying Label Encoding to: {label_cols}")
        le = LabelEncoder()
        for col in label_cols:
            if col in df_encoded.columns:
                # Ensure column is treated as string/object to handle mixed types before encoding
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            else:
                print(f"    - Warning: Column '{col}' not found for label encoding.")

    # 2. Frequency Encoding
    frequency_cols = encoding_plan.get("frequency_encoding", [])
    if frequency_cols:
        print(f"  - Applying Frequency Encoding to: {frequency_cols}")
        for col in frequency_cols:
            if col in df_encoded.columns:
                frequency_map = df_encoded[col].value_counts(normalize=True)
                df_encoded[col] = df_encoded[col].map(frequency_map)
                df_encoded[col].fillna(0, inplace=True)
            else:
                print(f"    - Warning: Column '{col}' not found for frequency encoding.")

    # 3. One-Hot Encoding
    one_hot_cols = encoding_plan.get("one_hot_encoding", [])
    one_hot_cols_exist = [col for col in one_hot_cols if col in df_encoded.columns]
    if one_hot_cols_exist:
        print(f"  - Applying One-Hot Encoding to: {one_hot_cols_exist}")
        df_encoded = pd.get_dummies(df_encoded, columns=one_hot_cols_exist, drop_first=True, dtype=int)
    
    if not any([label_cols, frequency_cols, one_hot_cols]):
        print("  - No columns identified for encoding.")

    return df_encoded

def handle_encoding(data: pd.DataFrame, llm_client) -> pd.DataFrame:
    """Orchestrates the feature encoding pipeline."""
    print("  - Identifying and applying feature encoding...")
    encoding_plan = get_encoding_strategy(data, llm_client)
    if not encoding_plan:
        print("    - No encoding plan received. Skipping encoding.")
        return data
    print(f"    - Received encoding plan from LLM: {encoding_plan}")
    encoded_data = apply_encoding(data, encoding_plan)
    return encoded_data
