import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lazypredict.Supervised import LazyClassifier, LazyRegressor, CLASSIFIERS, REGRESSORS
from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_squared_error
from feature_selection_ga import GeneticFeatureSelector
from sklearn.linear_model import LogisticRegression, LinearRegression
import numpy as np

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

    df_processed = df.copy()

    target_column = select_target_variable(df_processed)

    df_processed.dropna(subset=[target_column], inplace=True)
    X = df_processed.drop(columns=[target_column])
    y = df_processed[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    print(f"\nData split into training ({X_train.shape[0]} rows) and testing ({X_test.shape[0]} rows) sets.")

    print("\n--- Scaling Numerical Features ---")
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_cols_to_scale = [col for col in numeric_cols if X_train[col].nunique() > 2]

    if numeric_cols_to_scale:
        scaler = StandardScaler()
        scaler.fit(X_train[numeric_cols_to_scale])

        X_train.loc[:, numeric_cols_to_scale] = scaler.transform(X_train[numeric_cols_to_scale])
        X_test.loc[:, numeric_cols_to_scale] = scaler.transform(X_test[numeric_cols_to_scale])
        print(f"  - Applied StandardScaler to {len(numeric_cols_to_scale)} columns: {numeric_cols_to_scale}")
    else:
        print("  - No numerical columns with more than 2 unique values found to scale.")

    target_data = df_processed[target_column]
    unique_values = target_data.nunique()


    if pd.api.types.is_numeric_dtype(target_data) and unique_values > 20:
        problem_type = "Regression"
        print(f" Problem type detected as: {problem_type} (target is numeric with {unique_values} unique values).")
        automl_module = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
        ga_estimator = LinearRegression()
        ga_scoring = 'r2'
    else:
        problem_type = "Classification"
        print(f" Problem type detected as: {problem_type} (target is non-numeric or has {unique_values} unique values).")
        automl_module = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
        ga_estimator = LogisticRegression(random_state=123, max_iter=1000)
        ga_scoring = 'accuracy'

    print("\n--- Running Feature Selection with Genetic Algorithm ---")
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
    
    best_features = ga_selector.run() 
    
    print(f"  - Genetic algorithm selected {len(best_features)} features: {best_features}")
    
    X_train = X_train[best_features]
    X_test = X_test[best_features]

    print("\n---  Comparing Models with LazyPredict on Selected Features ---")
    print("This may take a moment...")
    
    models, predictions = automl_module.fit(X_train, X_test, y_train, y_test)

    print("\n--- AutoML Results ---")
    print(models)

    print("\nAutoML process complete. The table above shows the performance of all models.")

    if not models.empty:
        best_model_name = models.index[0]
        print(f"\n--- Training and Evaluating the Best Model: {best_model_name} ---")

        model_registry = {}
        if problem_type == "Regression":
            model_registry = {name: model for name, model in REGRESSORS}
        else:
            model_registry = {name: model for name, model in CLASSIFIERS}

        best_model_class = model_registry.get(best_model_name)

        if best_model_class:
            try:
                best_model_instance = best_model_class()
                best_model_instance.fit(X_train, y_train)
                
                y_pred = best_model_instance.predict(X_test)

                print(f"\n--- Best Model Performance: {best_model_name} ---")
                if problem_type == "Regression":
                    r2 = r2_score(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    print(f"  - R-squared (R²): {r2:.4f}")
                    print(f"  - Mean Squared Error (MSE): {mse:.4f}")
                    print(f"  - Root Mean Squared Error (RMSE): {rmse:.4f}")
                else: # Classification
                    accuracy = accuracy_score(y_test, y_pred)
                    print(f"  - Accuracy: {accuracy:.4f}")
                    print("  - Classification Report:")
                    print(classification_report(y_test, y_pred, zero_division=0))

                # Ask user if they want to save the model
                while True:
                    save_choice = input(f"\nWould you like to save the best model ('{best_model_name}') to a file? (y/n): ").lower().strip()
                    if save_choice in ['y', 'n']:
                        break
                    print("Invalid input. Please enter 'y' or 'n'.")

                if save_choice == 'y':
                    filename = "best_model.pkl"
                    with open(filename, 'wb') as f:
                        pickle.dump(best_model_instance, f)
                    print(f"  - Model saved successfully to '{filename}'")
            except Exception as e:
                print(f" An error occurred while training/evaluating the best model '{best_model_name}': {e}")
        else:
            print(f" Could not find the model class for '{best_model_name}' in the LazyPredict registry.")