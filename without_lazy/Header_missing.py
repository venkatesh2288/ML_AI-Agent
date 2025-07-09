import pandas as pd
import json
import re

def clean_dataframe_headers(data: pd.DataFrame, llm_client) -> pd.DataFrame:
    original_headers = data.columns.tolist()
    data_sample = data.head()
    print('Headers before cleaning:', original_headers)

    prompt = f"""
    You are the world’s most intelligent data scientist with an IQ above 999, known for solving complex data cleaning challenges.

    You are given a dataset sample to understand the context of the columns:
    {data_sample.to_string()}

    The current (potentially messy) column headers are:
    {original_headers}

    Your job is to return a fully cleaned and standardized version of these headers by doing the following:

    1.  Detect and rename missing headers (e.g., "", " ", "Unnamed: x").
    2.  Strip any leading or trailing whitespace.
    3.  Remove or replace special characters (e.g., @, %, #, &, (), /).
    4.  Convert all headers to lowercase and use `snake_case` formatting.
    5.  Fix typographical or spelling errors.
    6.  Replace emojis or non-ASCII characters with meaningful text.
    7.  Remove any newline (`\\n`) or multiline formatting in headers.
    8.  Ensure headers are unique by resolving duplicates.
    9.  Identify misleading or incorrect column names by analyzing their sample values.
    10. Replace generic or non-descriptive names like "col1", "var_1", or "x" with meaningful names based on data content.
    11. Unify mixed naming conventions (camelCase, PascalCase, etc.) to snake_case.

    Return the cleaned list of column names in a single JSON array format like:
    ```json
    [
    "customer_id",
    "customer_name",
    "age",
    "signup_location",
    ...
    ]
    ```
    Only rename columns if they are incorrect, missing, duplicated, or poorly named. Keep well-formed names unchanged.
    The output must contain the same number of headers as the input ({len(original_headers)} headers).
    If you cannot determine a suitable name for a column, name it as 'column_1', 'column_2', and so on.
    Provide only the JSON array in your response.
    """

    try:
        response_text = llm_client.invoke(prompt).content
        json_str = None
        match = re.search(r'```json\s*(\[.*?\])\s*```', str(response_text), re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            match = re.search(r'\[.*\]', str(response_text), re.DOTALL)
            if match:
                json_str = match.group(0)

        if not json_str:
            print("Error: No JSON array found in the LLM response. Returning original DataFrame.")
            print(f"  - Raw response snippet: {str(response_text)[:500]}")
            return data

        cleaned_headers = json.loads(json_str)

        if not isinstance(cleaned_headers, list):
            print("Error: LLM output was not a valid JSON list. Returning original DataFrame.")
            print(f"  - Parsed data type: {type(cleaned_headers)}")
            return data

        if len(cleaned_headers) != len(original_headers):
            print(f"Error: Mismatch in number of headers. Expected {len(original_headers)}, but got {len(cleaned_headers)}. Returning original DataFrame.")
            print(f"  - Received headers: {cleaned_headers}")
            return data

        print('Headers after cleaning:', cleaned_headers)
        data.columns = cleaned_headers
        return data

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from LLM response: {e}. Returning original DataFrame.")
        if 'json_str' in locals() and json_str:
            print(f"  - Problematic JSON string: {json_str}")
        return data
    except Exception as e:
        print(f"An unexpected error occurred while cleaning headers: {e}. Returning original DataFrame.")
        return data