import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import re

def get_chart_suggestions_from_llm(df: pd.DataFrame, llm_client):
    """
    Uses an LLM to generate insightful chart suggestions based on the data.
    """
    columns = df.columns.tolist()
    data_sample = df.head(10).to_string()
    
    # Identify potential numeric and categorical columns for better suggestions
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    prompt = f"""
    You are an expert data visualization specialist with an IQ of 999. Your task is to suggest insightful visualizations for a given dataset.

    Here are the available columns:
    {columns}

    Here is a breakdown of column types:
    - Numeric Columns: {numeric_cols}
    - Categorical Columns: {categorical_cols}

    Here is a sample of the data:
    {data_sample}

    Based on this information, suggest a list of up to 5 insightful charts.
    For each chart, provide the chart type, the columns for the x and y axes, an optional 'hue' for segmentation, and a brief description of the insight it might reveal.

    Choose from the following chart types: "scatterplot", "countplot", "boxplot", "barplot", "violinplot", "heatmap".
    - For "countplot", the 'y' axis is not needed.
    - For "barplot", 'x' should be categorical and 'y' should be numeric.
    - For "heatmap", 'x' and 'y' are not needed; it will be calculated on all numeric columns.
    - IMPORTANT: You MUST only use column names from the "Available columns" list provided above. Do not invent column names.

    Return your answer as a single, valid JSON object which is a list of dictionaries, enclosed in a markdown code block.
    Example format:
    ```json
    [
      {{
        "chart_type": "scatterplot",
        "x": "age",
        "y": "fare",
        "hue": "survived",
        "description": "Relationship between age and fare, segmented by survival status."
      }},
      {{
        "chart_type": "countplot",
        "x": "pclass",
        "hue": "sex",
        "description": "Distribution of passengers across classes, segmented by gender."
      }},
      {{
        "chart_type": "heatmap",
        "description": "Correlation matrix of all numeric features."
      }}
    ]
    ```
    Provide only the JSON object in your response. Ensure all suggested column names exist in the provided list of columns.
    """
    try:
        response_text = llm_client.invoke(prompt).content
        json_str = None
        # First, try to find JSON inside a markdown block
        match = re.search(r'```json\s*(\[.*?\])\s*```', response_text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # As a fallback, find the first occurrence of a list
            match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if match:
                json_str = match.group(0)

        if json_str:
            suggestions = json.loads(json_str)
            if isinstance(suggestions, list):
                return suggestions
        print("    - Warning: Could not parse a valid JSON list of chart suggestions from the LLM.")
        print(f"      - LLM Response Snippet: {response_text[:200]}")
        return []
    except (json.JSONDecodeError, Exception) as e:
        print(f"    - An error occurred while getting chart suggestions: {e}")
        return []

def generate_visualizations(df: pd.DataFrame, llm_client):
    """
    Generates and displays data visualizations based on LLM suggestions.
    """
    print("\n--- 🤖 Generating Visualization Suggestions with LLM ---")
    suggestions = get_chart_suggestions_from_llm(df, llm_client)

    if not suggestions:
        print("No visualization suggestions were generated. Skipping plotting.")
        return

    print(f"Received {len(suggestions)} visualization suggestions. Now plotting...")
    
    for i, chart_info in enumerate(suggestions):
        plt.figure(figsize=(10, 6))
        chart_type = chart_info.get("chart_type")
        description = chart_info.get("description", "LLM Suggested Chart")
        
        try:
            print(f"  - Plotting '{description}'...")
            
            if chart_type == "heatmap":
                numeric_df = df.select_dtypes(include=['number'])
                if not numeric_df.empty:
                    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
                else:
                    print("    - Skipping heatmap: No numeric columns found.")
                    plt.close()
                    continue
            else:
                # --- VALIDATION STEP ---
                # Check if all suggested columns actually exist in the DataFrame
                required_cols = [chart_info.get('x'), chart_info.get('y'), chart_info.get('hue')]
                missing_cols = [col for col in required_cols if col and col not in df.columns]
                if missing_cols:
                    print(f"    - Skipping chart: LLM suggested non-existent column(s): {missing_cols}")
                    plt.close()
                    continue
                # --- END VALIDATION ---
                plot_params = {'data': df, 'x': chart_info.get("x"), 'y': chart_info.get("y"), 'hue': chart_info.get("hue")}
                # Remove None values for seaborn compatibility
                plot_params = {k: v for k, v in plot_params.items() if v is not None}

                # Dynamically call the seaborn function
                plot_func = getattr(sns, chart_type, None)
                if plot_func:
                    plot_func(**plot_params)
                else:
                    print(f"    - Skipping chart: Unknown chart type '{chart_type}'")
                    plt.close()
                    continue

            plt.title(description)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"    - Failed to generate plot for '{description}': {e}")
            plt.close() # Close the figure if an error occurs