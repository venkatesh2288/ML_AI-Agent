import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from feature_selection_ga import GeneticFeatureSelector
from sklearn.linear_model import LogisticRegression, LinearRegression
from hardcoded_model_selection import find_best_model

def select_target_variable(df: pd.DataFrame) -> str:
    """
    Prompts the user to select the target variable from a list of columns.

    Args:
        df (pd.DataFrame): The cleaned DataFrame.

    Returns:
        str: The name of the selected target column.
    """
    print("\n--- Please Select the Target Variable for Modeling ---")
    columns = df.columns.tolist()
    for i, col in enumerate(columns):
        print(f"  [{i+1}] {col}")

    while True:
        try:
            choice = int(input(f"\nEnter the number of your target column (1-{len(columns)}): "))
            if 1 <= choice <= len(columns):
                target_column = columns[choice - 1]
                print(f"\n You have selected '{target_column}' as the target variable.")
                return target_column
            else:
                print(f" Invalid input. Please enter a number between 1 and {len(columns)}.")
        except (ValueError, IndexError):
            print(" Invalid input. Please enter a valid number corresponding to a column.")

def run_automl_experiment(df: pd.DataFrame):
    """
    Runs an AutoML experiment using PyCaret to find the best model.

    This function will:
    1. Ask the user to select a target variable.
    2. Automatically determine if the problem is classification or regression.
    3. Set up the PyCaret environment.
    4. Compare all available models and rank them by performance.
    5. Print the results leaderboard.

    Args:
        df (pd.DataFrame): The fully cleaned DataFrame ready for modeling.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        print(" AutoML step skipped: The provided DataFrame is empty or invalid.")
        return

    # Work on a copy to avoid modifying the original DataFrame passed to the function
    df_processed = df.copy()

    target_column = select_target_variable(df_processed)

    # --- Prepare data for modeling ---
    # Drop rows where the target variable is NaN, as it cannot be used for training/evaluation
    df_processed.dropna(subset=[target_column], inplace=True)
    X = df_processed.drop(columns=[target_column])
    y = df_processed[target_column]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    print(f"\nData split into training ({X_train.shape[0]} rows) and testing ({X_test.shape[0]} rows) sets.")

    # --- Scale numerical features ---
    print("\n--- Scaling Numerical Features ---")
    # Identify numeric columns, excluding binary columns (0/1) which don't need scaling.
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    # Filter out columns that are already binary (e.g., from one-hot encoding)
    numeric_cols_to_scale = [col for col in numeric_cols if X_train[col].nunique() > 2]

    if numeric_cols_to_scale:
        scaler = StandardScaler()
        # Fit the scaler ONLY on the training data to learn the mean and std dev
        scaler.fit(X_train[numeric_cols_to_scale])

        # Transform both the training and test data using the fitted scaler
        X_train.loc[:, numeric_cols_to_scale] = scaler.transform(X_train[numeric_cols_to_scale])
        X_test.loc[:, numeric_cols_to_scale] = scaler.transform(X_test[numeric_cols_to_scale])
        print(f"  - Applied StandardScaler to {len(numeric_cols_to_scale)} columns: {numeric_cols_to_scale}")
    else:
        print("  - No numerical columns with more than 2 unique values found to scale.")

    # --- Automatically determine the problem type (Classification vs. Regression) ---
    target_data = df_processed[target_column]
    unique_values = target_data.nunique()

    # Heuristic: If the target is non-numeric, or numeric with few unique values, it's a classification problem.
    # We'll consider <= 20 unique values as a threshold for classification.
    if pd.api.types.is_numeric_dtype(target_data) and unique_values > 20:
        problem_type = "Regression"
        print(f"Problem type detected as: {problem_type} (target is numeric with {unique_values} unique values).")
        ga_estimator = LinearRegression()
        ga_scoring = 'r2'
    else:
        problem_type = "Classification"
        print(f"Problem type detected as: {problem_type} (target is non-numeric or has {unique_values} unique values).")
        ga_estimator = LogisticRegression(random_state=123, max_iter=1000)
        ga_scoring = 'accuracy'

    # --- Feature Selection with Genetic Algorithm ---
    print("\n--- Running Feature Selection with Genetic Algorithm ---")
    # We use the training data to find the best features
    # The selector uses a simple model to evaluate feature subsets quickly.
    ga_selector = GeneticFeatureSelector(
        model=ga_estimator,
        X=X_train,
        y=y_train,
        scoring=ga_scoring,
        n_generations=10, # Keep low for speed, can be increased
        population_size=20, # Keep low for speed, can be increased
        cv=3, # Number of cross-validation folds for feature selection
        random_state=123 
    )
    
    # The run method should be part of your GeneticFeatureSelector class
    best_features = ga_selector.run() 
    
    print(f"  - Genetic algorithm selected {len(best_features)} features: {best_features}")
    
    # Update X_train and X_test to use only the selected features
    X_train = X_train[best_features]
    X_test = X_test[best_features]

    # --- Compare Models with Hardcoded Selection ---
    # This function will train, evaluate, print results, and perform a final
    # evaluation on the best model found.
    find_best_model(X_train, y_train, X_test, y_test, problem_type)

    print("\nModel selection process complete.")