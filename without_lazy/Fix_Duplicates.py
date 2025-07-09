import pandas as pd
import re
import json
import ast

def find_and_clean_semantic_duplicates(column_list, llm_client):
    """
    Identifies semantic duplicate columns using an LLM, parses the response robustly,
    and returns a cleaned list of unique duplicate pairs.
    """
    prompt = f'''
You are an expert in data analysis with a very high IQ. You are given a list of column names and your task is to find semantic duplicates (e.g., 'nation' and 'country' are the same).

You must find only the semantic duplicate **pairs** and return them in a **Python list of tuples**.

List of columns:
{column_list}

Return only the list of duplicate pairs, like this:
[("column1", "column2"), ("column3", "column4")]

Do NOT include any explanation or extra text. Only return the list.
'''
    try:
        response_text = llm_client.invoke(prompt).content
        match = re.search(r'\[.*\]', response_text, re.DOTALL)

        if not match:
            print("Warning: Could not find a list structure `[...]` in the LLM response for semantic duplicates.")
            return []

        raw_duplicates = ast.literal_eval(match.group(0))

        if not isinstance(raw_duplicates, list):
            print(f"Warning: LLM output was not a list. Type: {type(raw_duplicates)}")
            return []

        cleaned_duplicates = []
        seen_pairs = set()
        for item in raw_duplicates:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                pair = tuple(sorted(item))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    cleaned_duplicates.append(pair)
        return cleaned_duplicates

    except (ValueError, SyntaxError) as e:
        print(f"Error parsing LLM response for semantic duplicates: {e}")
        print(f"  - Raw response snippet: {response_text[:200] if 'response_text' in locals() else 'N/A'}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred in find_and_clean_semantic_duplicates: {e}")
        return []

def remove_semantic_duplicates(data, duplicates):
    """Merges and removes semantic duplicate columns."""
    for col1, col2 in duplicates:
        if col1 in data.columns and col2 in data.columns:
            print(f"    Merging semantic duplicates: '{col1}' and '{col2}' -> keeping '{col1}'")
            #! Fill missing values in the first column with values from the second
            data[col1] = data[col1].fillna(data[col2])
            #! Drop the second column
            data = data.drop(columns=[col2])
    return data

def find_redundant_columns(data, llm_client):
    """Uses an LLM to identify columns that are derived from others."""
    column_list = data.columns.tolist()
    sample_data = data.head(5).to_string()
    prompt = f'''
You are a highly intelligent data analyst specializing in feature engineering and redundancy detection.
Your task is to analyze the provided columns and sample data to identify redundant columns.

A column is considered redundant if it is directly derived from one or more other columns through a simple mathematical or string operation.
For example:
- If a 'grade' column exists and a 'score' column is simply 'grade / 10', then 'score' is redundant and should be removed.
- If a 'full_name' column is just a concatenation of 'first_name' and 'last_name', then 'full_name' might be considered redundant.

Based on the columns and data below, identify which columns should be removed.

Columns: {column_list}

Sample Data:
{sample_data}

Return your answer as a JSON array of strings, where each string is the name of a column to be removed.
For example: ["score", "redundant_column_2"]
If no redundant columns are found, return an empty list `[]`.
Provide only the JSON array and nothing else.
'''
    try:
        #! Use the passed llm_client
        response_text = llm_client.invoke(prompt).content
        match = re.search(r'\[.*?\]', response_text, re.DOTALL)
        if match:
            columns_to_remove = json.loads(match.group(0))
            if isinstance(columns_to_remove, list):
                return columns_to_remove
        print("Warning: No valid JSON array found in LLM response for redundant columns.")
        return []
    except (json.JSONDecodeError, Exception) as e:
        print(f"An error occurred during redundant column detection: {e}")
        return []


def handle_duplicates(data: pd.DataFrame, llm_client) -> pd.DataFrame:
    """
    Orchestrates the duplicate handling pipeline for columns.
    This includes semantic and redundant duplicate column removal.
    Row-based duplicate removal has been disabled.
    """
    cleaned_data = data.copy() # Work on a copy

    #! 1. Semantic duplicate column removal
    print("  - Detecting and removing semantic duplicate columns...")
    columns = cleaned_data.columns.tolist()
    semantic_duplicates = find_and_clean_semantic_duplicates(columns, llm_client)
    if semantic_duplicates:
        cleaned_data = remove_semantic_duplicates(cleaned_data, semantic_duplicates)
    else:
        print("    No semantic duplicates found.")

    #! 2. Redundant (derived) column removal
    print("  - Detecting and removing redundant (derived) columns...")
    redundant_cols = find_redundant_columns(cleaned_data, llm_client)
    if redundant_cols:
        cols_to_drop = [col for col in redundant_cols if col in cleaned_data.columns]
        if cols_to_drop:
            print(f"    LLM identified redundant columns for removal: {cols_to_drop}")
            cleaned_data = cleaned_data.drop(columns=cols_to_drop)
    else:
        print("    No redundant columns found.")

    print("  - Row-based duplicate removal is disabled.")

    return cleaned_data
