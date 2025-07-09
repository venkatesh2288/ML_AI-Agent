import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, r2_score, classification_report, mean_squared_error

# --- Model Imports ---
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet, 
    RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier, 
    BayesianRidge, SGDRegressor, PassiveAggressiveRegressor, HuberRegressor
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier,
    RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
)
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR, LinearSVC, NuSVC, LinearSVR, NuSVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# --- Optional Imports for XGBoost and LightGBM ---
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGB_INSTALLED = True
except ImportError:
    XGB_INSTALLED = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LGBM_INSTALLED = True
except ImportError:
    LGBM_INSTALLED = False

def find_best_model(X_train, y_train, X_test, y_test, problem_type):
    """
    Trains, evaluates, and selects the best model from a hardcoded suite.

    This function encapsulates the entire process:
    1. Defines a list of candidate models based on the problem type.
    2. Trains and evaluates each model with default parameters.
    3. Selects the best model type based on the primary metric.
    4. Performs hyperparameter tuning on the best model using GridSearchCV.
    5. Prints a final, detailed evaluation of the tuned model.

    Args:
        X_train: Training feature data.
        y_train: Training target data.
        X_test: Testing feature data.
        y_test: Testing target data.
        problem_type (str): "Classification" or "Regression".

    Returns:
        tuple: A tuple containing the best trained model instance and its name.
               Returns (None, None) if the problem type is invalid or no models succeed.
    """
    print(f"\n--- Comparing Hardcoded Models for {problem_type} ---")

    model_candidates = {}
    results = {}

    # --- Define Parameter Grids for Hyperparameter Tuning ---
    # These grids are more comprehensive for better tuning.
    param_grids = {
        "Classification": {
            "Logistic Regression": {'C': [0.01, 0.1, 1.0, 10.0], 'solver': ['liblinear'], 'penalty': ['l1', 'l2']},
            "Random Forest": {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]},
            "Gradient Boosting": {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]},
            "AdaBoost": {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0]},
            "SVM": {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto'], 'kernel': ['rbf', 'poly']},
            "Linear SVC": {'C': [0.1, 1, 10, 100], 'loss': ['hinge', 'squared_hinge']},
            "NuSVC": {'nu': [0.25, 0.5, 0.75], 'kernel': ['rbf', 'poly'], 'gamma': ['scale', 'auto']},
            "K-Nearest Neighbors": {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance'], 'metric': ['minkowski', 'euclidean']},
            "Decision Tree": {'max_depth': [10, 20, None], 'min_samples_split': [2, 5, 10], 'criterion': ['gini', 'entropy']},
            "Extra Trees": {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5, 10]},
            "Ridge Classifier": {'alpha': [0.1, 1.0, 10.0, 100.0]},
            "SGD Classifier": {'loss': ['hinge', 'log_loss', 'modified_huber'], 'penalty': ['l2', 'l1'], 'alpha': [0.0001, 0.001, 0.01]},
            "Passive Aggressive Classifier": {'C': [0.1, 1.0, 10.0], 'loss': ['hinge', 'squared_hinge']},
            "XGBoost": {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5, 7], 'subsample': [0.7, 1.0]},
            "LightGBM": {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'num_leaves': [31, 50, 70]},
        },
        "Regression": {
            "Ridge Regression": {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]},
            "Lasso Regression": {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]},
            "ElasticNet": {'alpha': [0.01, 0.1, 1.0, 10.0], 'l1_ratio': [0.1, 0.5, 0.9]},
            "Random Forest": {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]},
            "Gradient Boosting": {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]},
            "AdaBoost": {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0]},
            "SVR": {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto'], 'kernel': ['rbf', 'poly']},
            "Linear SVR": {'C': [0.1, 1, 10, 100], 'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive']},
            "NuSVR": {'nu': [0.25, 0.5, 0.75], 'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']},
            "K-Nearest Neighbors": {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']},
            "Decision Tree": {'max_depth': [10, 20, None], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]},
            "Extra Trees": {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5, 10]},
            "SGD Regressor": {'loss': ['squared_error', 'huber'], 'penalty': ['l2', 'l1'], 'alpha': [0.0001, 0.001, 0.01]},
            "Passive Aggressive Regressor": {'C': [0.1, 1.0, 10.0], 'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive']},
            "Huber Regressor": {'epsilon': [1.35, 1.5, 1.75], 'alpha': [0.0001, 0.001, 0.01]},
            "XGBoost": {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5, 7], 'subsample': [0.7, 1.0]},
            "LightGBM": {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'num_leaves': [31, 50, 70]},
        }
    }

    # --- Define models for each type ---
    if problem_type == "Classification":
        model_candidates = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=123),
            "Random Forest": RandomForestClassifier(random_state=123),
            "Gradient Boosting": GradientBoostingClassifier(random_state=123),
            "AdaBoost": AdaBoostClassifier(random_state=123),
            "SVM": SVC(random_state=123, probability=True),
            "Linear SVC": LinearSVC(random_state=123, max_iter=5000, dual=True),
            "NuSVC": NuSVC(random_state=123, probability=True),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(random_state=123),
            "Naive Bayes": GaussianNB(),
            "Extra Trees": ExtraTreesClassifier(random_state=123),
            "Ridge Classifier": RidgeClassifier(random_state=123),
            "SGD Classifier": SGDClassifier(random_state=123),
            "Passive Aggressive Classifier": PassiveAggressiveClassifier(random_state=123),
        }
        metric_name = "Accuracy"
        metric_func = accuracy_score
        is_classification = True
    elif problem_type == "Regression":
        model_candidates = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(random_state=123),
            "Lasso Regression": Lasso(random_state=123),
            "ElasticNet": ElasticNet(random_state=123),
            "Random Forest": RandomForestRegressor(random_state=123),
            "Gradient Boosting": GradientBoostingRegressor(random_state=123),
            "AdaBoost": AdaBoostRegressor(random_state=123),
            "SVR": SVR(),
            "Linear SVR": LinearSVR(random_state=123, max_iter=5000),
            "NuSVR": NuSVR(),
            "K-Nearest Neighbors": KNeighborsRegressor(),
            "Decision Tree": DecisionTreeRegressor(random_state=123),
            "Extra Trees": ExtraTreesRegressor(random_state=123),
            "BayesianRidge": BayesianRidge(),
            "SGD Regressor": SGDRegressor(random_state=123),
            "Passive Aggressive Regressor": PassiveAggressiveRegressor(random_state=123),
            "Huber Regressor": HuberRegressor(),
        }
        metric_name = "R²"
        metric_func = r2_score
        is_classification = False
    else:
        print(f" Invalid problem_type: {problem_type}. Please use 'Classification' or 'Regression'.")
        return None, None

    # --- Add optional models if they are installed ---
    if XGB_INSTALLED:
        if is_classification:
            model_candidates["XGBoost"] = XGBClassifier(random_state=123, use_label_encoder=False, eval_metric='logloss')
        else:
            model_candidates["XGBoost"] = XGBRegressor(random_state=123)
    if LGBM_INSTALLED:
        if is_classification:
            model_candidates["LightGBM"] = LGBMClassifier(random_state=123)
        else:
            model_candidates["LightGBM"] = LGBMRegressor(random_state=123)

    # --- Train and Evaluate Each Model ---
    for name, model in model_candidates.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = metric_func(y_test, y_pred)
            results[name] = score
            print(f" {name}: {metric_name} = {score:.4f}")
        except Exception as e:
            print(f" {name} failed: {e}")

    if not results:
        print("No models were successfully trained and evaluated.")
        return None, None

    # --- Select Best Model from Initial Run ---
    best_model_name = max(results, key=results.get)
    print(f"\n🏆 Best Model Selected (based on initial run): {best_model_name}")

    # --- Hyperparameter Tuning on Best Model ---
    print(f"\n--- Hyperparameter Tuning for {best_model_name} ---")
    
    # Get the base model instance (untrained) and the parameter grid
    base_model = model_candidates[best_model_name]
    grid_for_model = param_grids.get(problem_type, {}).get(best_model_name)
    
    if grid_for_model:
        print(f"  - Searching parameter grid: {grid_for_model}")
        scoring_metric = 'accuracy' if is_classification else 'r2'
        
        # n_jobs=-1 uses all available CPU cores to speed up the search
        grid_search = GridSearchCV(estimator=base_model, param_grid=grid_for_model, cv=3, scoring=scoring_metric, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        
        best_model_instance = grid_search.best_estimator_
        print(f"  - Tuning Complete. Best Parameters: {grid_search.best_params_}")
        print(f"  - Best cross-validation score from tuning: {grid_search.best_score_:.4f}")
    else:
        print(f"  - No hyperparameter grid defined for {best_model_name}. Using default model.")
        best_model_instance = model_candidates[best_model_name] # Use the already trained model

    # --- Final Evaluation on Best (and Tuned) Model ---
    print("\n--- Final Evaluation ---")
    y_pred_best = best_model_instance.predict(X_test)

    if is_classification:
        print(f"  - Accuracy: {accuracy_score(y_test, y_pred_best):.4f}")
        print("  - Classification Report:")
        print(classification_report(y_test, y_pred_best, zero_division=0))
    else:  # Regression
        print(f"  - R²: {r2_score(y_test, y_pred_best):.4f}")
        print(f"  - MSE: {mean_squared_error(y_test, y_pred_best):.4f}")
        print(f"  - RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_best)):.4f}")

    return best_model_instance, best_model_name


def main():
    """
    Main function to demonstrate the model selection process for both problem types.
    This block is for demonstration purposes and can be removed when integrating.
    """
    # --- Classification Example ---
    print("========================================")
    print("=      CLASSIFICATION EXAMPLE        =")
    print("========================================")
    X_c, y_c = make_classification(n_samples=1000, n_features=20, n_informative=5, n_redundant=0, random_state=123)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_c, y_c, test_size=0.2, random_state=123)
    find_best_model(X_train_c, y_train_c, X_test_c, y_test_c, "Classification")

    # --- Regression Example ---
    print("\n========================================")
    print("=        REGRESSION EXAMPLE          =")
    print("========================================")
    X_r, y_r = make_regression(n_samples=1000, n_features=20, n_informative=10, random_state=123)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_r, y_r, test_size=0.2, random_state=123)
    find_best_model(X_train_r, y_train_r, X_test_r, y_test_r, "Regression")


if __name__ == "__main__":
    main()