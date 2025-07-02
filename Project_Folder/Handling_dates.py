import re
import pandas as pd
import json
import unicodedata

def handle_dates(df, llm, dayfirst=False):
    """
    Detects and fixes date/time format inconsistencies across columns using LLM for identification.
    Converts all identified date columns into separate _year, _month, _day columns.
    Handles ambiguous date formats (e.g., MM/DD vs DD/MM) via the `dayfirst` parameter.
    Imputes missing month/day with middle values (June 15th) if only year is available or date is unparseable.
    This function modifies the DataFrame in-place.
    """
    modified_cols = []
    cols_to_drop = []


    def identify_date_columns_with_llm(df, llm):
        """
        Uses an LLM to identify single and multi-part date columns from a DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.
            llm: The initialized LangChain LLM object.

        Returns:
            dict: A dictionary with identified date columns, or an empty dict on failure.
        """
        columns = df.columns.tolist()
        prompt = f"""
        You are an expert data analyst specializing in identifying date-related features in datasets.
        Analyze the following list of column names and identify which columns represent dates.

        Column Names: {columns}

        Your task is to return a JSON object that categorizes these columns. The JSON should have two main keys:
        1.  `"single_column_dates"`: A list of column names where each column by itself represents a full date or datetime (e.g., "order_date", "timestamp").
        2.  `"multi_column_dates"`: A dictionary where each key is a descriptive name for a new combined date column (e.g., "transaction_date"), and the value is another dictionary mapping date components ('year', 'month', 'day') to their corresponding column names from the input list.
            If 'month' or 'day' are missing in the input columns for a multi-column date, assume they are not provided and the LLM should still identify the 'year' if present.

        Example Input Columns: ['order_id', 'sale_date', 'customer_name', 'tx_year', 'tx_month', 'tx_day', 'price', 'event_year']

        Example Output:
        {{
            "single_column_dates": ["sale_date"],
            "multi_column_dates": {{
                "transaction_date": {{
                    "year": "tx_year",
                    "month": "tx_month",
                    "day": "tx_day"
                }},
                "event_date": {{
                    "year": "event_year"
                }}
            }}
        }}

        If no date-related columns are found, return an empty JSON object with empty lists/dictionaries.
        Provide only the JSON object in your response, with no surrounding text.
        """
        try:
            response = llm.invoke(prompt)
            content = response.content
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            else:
                print("Warning: LLM response did not contain a valid JSON object for date identification.")
                return {"single_column_dates": [], "multi_column_dates": {}}
        except (json.JSONDecodeError, Exception) as e:
            print(f"An error occurred during LLM interaction or JSON parsing for date identification: {e}")
            return {"single_column_dates": [], "multi_column_dates": {}}

    def parse_and_impute_date(series, dayfirst_param=False):
        def clean_string(text):
            text = str(text)
            text = unicodedata.normalize("NFKD", text)
            return re.sub(r'[^\x00-\x7F]+', '', text).strip()

        cleaned_series = series.apply(clean_string)

        parsed = pd.to_datetime(cleaned_series, errors='coerce', dayfirst=dayfirst_param)

    
        nat_mask = parsed.isna()
        if nat_mask.any():
            imputed_years = cleaned_series[nat_mask].str.extract(r'(?<!\d)\b(\d{4})\b(?!\d)')[0]

            valid_imputed_mask = imputed_years.notna()
            if valid_imputed_mask.any():
                indices_to_update = imputed_years[valid_imputed_mask].index
                years_to_update = imputed_years[valid_imputed_mask].astype(int)
                # ...create a new Timestamp with a default month/day and update the parsed series
                parsed.loc[indices_to_update] = years_to_update.apply(lambda y: pd.Timestamp(y, 6, 15))
        return parsed


    #! 1. Identify date columns using LLM
    date_info = identify_date_columns_with_llm(df, llm)
    single_cols = date_info.get("single_column_dates", [])
    multi_col_groups = date_info.get("multi_column_dates", {})

    #! 2. Process single-column dates
    for col in single_cols:
        if col in df.columns:
            print(f"Processing single date column: {col}")
            temp_date_series = parse_and_impute_date(df[col], dayfirst_param=dayfirst)

            df[f"{col}_year"] = temp_date_series.dt.year
            df[f"{col}_month"] = temp_date_series.dt.month
            df[f"{col}_day"] = temp_date_series.dt.day
            
            modified_cols.extend([f"{col}_year", f"{col}_month", f"{col}_day"])
            cols_to_drop.append(col)

    #! 3. Process multi-column dates
    for new_col_name, components in multi_col_groups.items():
        if 'year' not in components or not components.get('year'):
            print(f"Warning: Skipping multi-column date group '{new_col_name}' because 'year' component is missing from LLM response.")
            continue

        #! Ensure all required component columns exist
        component_cols_exist = all(c in df.columns for c in components.values())
        if not component_cols_exist:
            print(f"Warning: Skipping multi-column date group '{new_col_name}' due to missing component columns: {components.values()}")
            continue

        print(f"Processing multi-column date group: {new_col_name} from {components}")
        
    
        temp_df_for_parsing = pd.DataFrame(index=df.index)
        
        # Default values for missing components
        default_year = 1900 # A placeholder year if not provided, will be imputed later if needed
        default_month = 6   # Middle month
        default_day = 15    # Middle day

        # Helper to get column as a Series, even if it doesn't exist (fills with default)
        def get_series_or_default(df_source, col_name_key, default_val):
            col_name = components.get(col_name_key)
            if col_name and col_name in df_source.columns:
                return df_source[col_name]
            else:
                # Create a Series of default values with the same index as the DataFrame
                return pd.Series(default_val, index=df_source.index)

        # Extract components, ensuring they are Series and handling non-numeric values
        year_series = get_series_or_default(df, 'year', default_year)
        temp_df_for_parsing['year'] = pd.to_numeric(year_series, errors='coerce').fillna(default_year).astype(int)

        month_series = get_series_or_default(df, 'month', default_month)
        # Handle month names (jan, feb) if they are in a separate column, before numeric conversion
        # This part needs to be careful if month_series itself is already numeric.
        month_col = month_series # Use the series obtained from get_series_or_default
        if month_col is not None:
            # Convert month names to numbers if they are strings
            month_map = {
                'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
                'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6, 'jul': 7, 'july': 7,
                'aug': 8, 'august': 8, 'sep': 9, 'september': 9, 'oct': 10, 'october': 10,
                'nov': 11, 'november': 11, 'dec': 12, 'december': 12
            }
            # Ensure month column is string, then map, then convert to numeric
            # Only apply map if it's an object/string dtype, otherwise directly convert to numeric
            temp_month_series = month_col.astype(str).str.lower().map(month_map).fillna(pd.to_numeric(month_col, errors='coerce')) if pd.api.types.is_object_dtype(month_col) or pd.api.types.is_string_dtype(month_col) else pd.to_numeric(month_col, errors='coerce')
            temp_df_for_parsing['month'] = temp_month_series.fillna(default_month).astype(int)
        else:
            temp_df_for_parsing['month'] = default_month

        day_series = get_series_or_default(df, 'day', default_day)
        temp_df_for_parsing['day'] = pd.to_numeric(day_series, errors='coerce').fillna(default_day).astype(int)

        # Combine into a single datetime series
        # Use apply to handle potential invalid dates after imputation (e.g., Feb 30)
        combined_date_series = temp_df_for_parsing.apply(
            lambda row: pd.to_datetime(f"{row['year']}-{row['month']}-{row['day']}", errors='coerce'),
            axis=1
        )
        
        # For any remaining NaT (e.g., Feb 30 became NaT), impute again with default day/month
        for idx, val in combined_date_series.items():
            if pd.isna(val):
                year = temp_df_for_parsing.loc[idx, 'year']
                # Ensure year is valid before creating timestamp
                if pd.notna(year) and year > 0: # Basic year validation
                    combined_date_series.loc[idx] = pd.Timestamp(year, default_month, default_day)
                else:
                    # If year itself is invalid, keep as NaT for later dropping
                    pass 

        # Extract year, month, day into new columns
        df[f"{new_col_name}_year"] = combined_date_series.dt.year
        df[f"{new_col_name}_month"] = combined_date_series.dt.month
        df[f"{new_col_name}_day"] = combined_date_series.dt.day
        
        modified_cols.extend([f"{new_col_name}_year", f"{new_col_name}_month", f"{new_col_name}_day"])
        cols_to_drop.extend(components.values())

    # Drop original columns that have been processed
    if cols_to_drop:
        df.drop(columns=list(set(cols_to_drop)), inplace=True)
        print(f"Dropped original date columns: {list(set(cols_to_drop))}")

    # Final cleanup: remove any rows where date parsing ultimately failed for any of the new columns
    # Instead of dropping, let's fill with pd.NA to indicate unparseable dates,
    # allowing downstream missing value handling to decide.
    if modified_cols:
        initial_rows = len(df)
        for col in modified_cols:
            # Fill NaNs in newly created date components with pd.NA or a suitable placeholder
            # This preserves the row, allowing other columns to remain.
            df[col] = df[col].fillna(pd.NA)
        # No rows are explicitly dropped here anymore.
        print(f"Filled unparseable date components with pd.NA in columns: {modified_cols}. Rows preserved.")

    return df, modified_cols
