import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from context_tree_selector import ContextTreeFeatureSelector  # Import Context Tree
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

def run_automl_experiment(df: pd.DataFrame, target_column: str):
    """
    Runs an AutoML experiment with Context Tree + Genetic Algorithm feature selection.

    This function will:
    1. Ask the user to select a target variable.
    2. Automatically determine if the problem is classification or regression.
    3. Set up the data preprocessing.
    4. Run Context Tree feature selection to pre-select relevant features.
    5. Run Genetic Algorithm on the Context Tree selected features.
    6. Compare all available models and rank them by performance.
    7. Print the results leaderboard.

    Args:
        df (pd.DataFrame): The fully cleaned DataFrame ready for modeling.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        print(" AutoML step skipped: The provided DataFrame is empty or invalid.")
        return

    # Work on a copy to avoid modifying the original DataFrame passed to the function
    df_processed = df.copy()

    # target_column = select_target_variable(df_processed)
    print(f"\nTarget variable '{target_column}' was selected in the UI.")


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

    # --- Step 1: Context Tree Feature Selection ---
    print("\n=== PHASE 1: Context Tree Feature Selection ===")
    context_tree_selector = ContextTreeFeatureSelector(
        X=X_train,
        y=y_train,
        problem_type=problem_type.lower(),
        max_features_ratio=0.7,  # Select up to 70% of features
        min_importance_threshold=0.05,  # Minimum importance threshold
        correlation_threshold=0.8,  # Correlation threshold for feature grouping
        random_state=123
    )
    
    # Run Context Tree feature selection
    context_selected_features, context_feature_importance = context_tree_selector.run()
    
    # Display Context Tree summary
    print(f"\n{context_tree_selector.get_tree_summary()}")
    
    # Update X_train and X_test to use only Context Tree selected features
    X_train_context = X_train[context_selected_features]
    X_test_context = X_test[context_selected_features]
    
    print(f"\nContext Tree reduced feature space from {X_train.shape[1]} to {X_train_context.shape[1]} features.")

    # --- Step 2: Genetic Algorithm Feature Selection on Context Tree Features ---
    print("\n=== PHASE 2: Genetic Algorithm Feature Selection ===")
    print("Running Genetic Algorithm on Context Tree selected features...")
    
    ga_selector = GeneticFeatureSelector(
        model=ga_estimator,
        X=X_train_context,  # Use Context Tree selected features
        y=y_train,
        scoring=ga_scoring,
        n_generations=10,  # Keep low for speed, can be increased
        population_size=20,  # Keep low for speed, can be increased
        cv=3,  # Number of cross-validation folds for feature selection
        random_state=123 
    )
    
    # The run method returns the final selected features from GA
    final_selected_features = ga_selector.run() 
    
    print(f"\nGenetic Algorithm selected {len(final_selected_features)} features from {len(context_selected_features)} Context Tree features.")
    print(f"Final selected features: {final_selected_features}")
    
    # --- Feature Selection Pipeline Summary ---
    print(f"\n=== FEATURE SELECTION PIPELINE SUMMARY ===")
    print(f"Original features: {X_train.shape[1]}")
    print(f"Context Tree selected: {len(context_selected_features)} ({len(context_selected_features)/X_train.shape[1]:.1%})")
    print(f"Genetic Algorithm selected: {len(final_selected_features)} ({len(final_selected_features)/X_train.shape[1]:.1%})")
    print(f"Total reduction: {X_train.shape[1] - len(final_selected_features)} features removed")
    
    # Display feature importance from Context Tree for final selected features
    print(f"\nFinal Selected Features with Context Tree Importance:")
    for feature in final_selected_features:
        if feature in context_feature_importance:
            print(f"  - {feature}: {context_feature_importance[feature]:.4f}")
    
    # Update X_train and X_test to use only the final selected features
    X_train_final = X_train_context[final_selected_features]
    X_test_final = X_test_context[final_selected_features]

    # --- Step 3: Model Selection and Training ---
    print(f"\n=== PHASE 3: Model Selection and Training ===")
    print(f"Training models with {len(final_selected_features)} optimally selected features...")
    
    # This function will train, evaluate, print results, and perform a final
    # evaluation on the best model found.
    find_best_model(X_train_final, y_train, X_test_final, y_test, problem_type)

    print(f"\n=== AutoML Pipeline Complete ===")
    print("✓ Context Tree feature pre-selection")
    print("✓ Genetic Algorithm feature optimization") 
    print("✓ Model selection and evaluation")
    print("✓ Final model training and testing")
    print("\nModel selection process complete.")

