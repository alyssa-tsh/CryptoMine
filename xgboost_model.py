#MODULE: XGBOOST 
#MODULE : MODEL LIBRARIES
# all libraries
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from xgboost import XGBRegressor
from skopt import BayesSearchCV
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import VarianceThreshold
from skopt.space import Real, Integer
import numpy as np
import matplotlib.pyplot as plt
import optuna
import pandas as pd


def random_search(X_train, X_val, X_test, y_train, y_val, y_test, target_var):
    # Combine train + val for final training later
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train[target_var], y_val[target_var]])

    # Perform Rolling Expanding Validation  
    tscv = TimeSeriesSplit(n_splits=10)  
    model = XGBRegressor(random_state=42)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05],
        'max_depth': [3, 5],
        'min_child_weight': [5, 10],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0.1, 1.0],
        'reg_lambda': [1.0, 2.0],
    }

    # Perform Random Search with TimeSeriesSplit cross-validation
    random_search = RandomizedSearchCV(
        model, 
        param_grid, 
        n_iter=50, 
        cv=tscv,  
        n_jobs=-1,
        random_state=42,
        scoring='neg_mean_squared_error'
    )

    print("Running Hyperparameter Tuning...")
    random_search.fit(X_train, y_train[target_var])  
    best_params = random_search.best_params_
    print(f'Best parameters: {best_params}')

    best_model = XGBRegressor(**best_params, random_state=42)
    best_model.fit(X_train, y_train[target_var])

    y_val_pred = best_model.predict(X_val)

    print("\n=== Validation Performance (Jan-Feb 2025) ===")
    print(f"MSE: {mean_squared_error(y_val[target_var], y_val_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_val[target_var], y_val_pred):.4f}")
    print(f"R²: {r2_score(y_val[target_var], y_val_pred):.4f}")

    # Extract feature importances
    feature_importances = pd.Series(best_model.feature_importances_, index=X_train_val.columns)
    feature_importances_sorted = feature_importances.sort_values(ascending=False)

    # Plot Feature Importance
    feature_importances_sorted[:10].plot(kind='barh', color='skyblue')
    plt.title("Top 10 Feature Importances")
    plt.show()

    # Select features based on importance (top 10)
    selected_features = feature_importances_sorted.head(10).index
    print("Selected Features based on Importance:", selected_features)

    X_train_val_selected = X_train_val[selected_features]

    # Train Final Model with Only Selected Features 
    X_train_selected = X_train[selected_features]
    X_val_selected = X_val[selected_features]
    X_test_selected = X_test[selected_features]

    final_model = XGBRegressor(**best_params, random_state=42)
    final_model.fit(X_train_val_selected, y_train_val)  # Train on selected features

    # Evaluate on Train & Validation
    y_train_pred = final_model.predict(X_train_selected)
    y_val_pred = final_model.predict(X_val_selected)
    # Evaluate on Test Data (March–April 2025) 
    y_test_pred = final_model.predict(X_test_selected)

    print("\n=== Train Performance (Jan-Dec 2024) ===")
    print(f"MSE: {mean_squared_error(y_train[target_var], y_train_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_train[target_var], y_train_pred):.4f}")
    print(f"R²: {r2_score(y_train[target_var], y_train_pred):.4f}")

    print("\n=== Validation Performance (Jan-Feb 2025) ===")
    print(f"MSE: {mean_squared_error(y_val[target_var], y_val_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_val[target_var], y_val_pred):.4f}")
    print(f"R²: {r2_score(y_val[target_var], y_val_pred):.4f}")


    print("\n=== Test Performance (March-April 2025) ===")
    print(f"MSE: {mean_squared_error(y_test[target_var], y_test_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_test[target_var], y_test_pred):.4f}")
    print(f"R²: {r2_score(y_test[target_var], y_test_pred):.4f}")

    return final_model, y_train_pred, y_val_pred, y_test_pred





#MODULE: XGBOOST: Bayes Optimization
def bayes_xg_boost(X_train, X_test, X_val, y_train, y_val, y_test, target_var):

    print("=== Phase 1: Feature Selection ===")
    
    # A. Remove low-variance features
    variance_selector = VarianceThreshold(threshold=0.01)
    X_train_filtered = variance_selector.fit_transform(X_train)
    
    
    # # Get remaining feature names
    remaining_features = X_train.columns[variance_selector.get_support()]
    print(remaining_features)
    print(f"Removed {X_train.shape[1] - len(remaining_features)} low-variance features")

    # #B: Feature Correlation Analysis - removes almost all SMA & EMA features
    # corr_matrix = pd.DataFrame(X_train_filtered, columns=remaining_features).corr().abs()
    # upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]  # 0.8 threshold
    # print(to_drop)
    # X_train_filtered = pd.DataFrame(X_train_filtered, columns=remaining_features).drop(to_drop, axis=1)

    # Define the initial model
    initial_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    initial_model.fit(X_train[remaining_features], y_train[target_var])
        
    # C: Get feature importances & select top 10
    feature_importances = pd.Series(initial_model.feature_importances_, index=remaining_features)
    feature_importances = feature_importances.sort_values(ascending=False)
    
    print(f'Feature importances:\n{feature_importances}')
    feature_importances[:10].plot(kind='barh', color='skyblue')
    plt.title("Top 10 Feature Importances")
    plt.show()

    selected_features = feature_importances.nlargest(10).index.tolist()
    print(f"\nSelected {len(selected_features)} features:\n{selected_features}")

    # Create train/val/test sets with selected features
    X_train_selected = X_train[selected_features]
    X_val_selected = X_val[selected_features]
    X_test_selected = X_test[selected_features]

    print("\n=== Phase 2: Hyperparameter Optimization ===")
    tscv = TimeSeriesSplit(n_splits=5)
    model = XGBRegressor(objective='reg:squarederror', random_state=42)

    # Define the search space for hyperparameters 
    search_space = {
        'learning_rate': Real(0.01, 0.7, prior='log-uniform'), 
        'max_depth': Integer(5, 14), 
        'gamma': Real(0.4, 6.4), 
        'reg_alpha': Real(0.4, 6.4), 
        'reg_lambda': Real(0.4, 6.4), 
        'n_estimators': Integer(50, 200)
    }

    # Step 1: Optimize hyperparameters using BayesSearchCV
    opt = BayesSearchCV(
        model,
        search_space,
        cv=tscv,
        n_jobs=-1,
        scoring='neg_mean_squared_error',
        random_state=42
    )

    opt.fit(X_train_selected, y_train[target_var])

    # Best parameters from BayesSearchCV
    best_params_bayescv = opt.best_params_
    y_test_predcv = opt.predict(X_val_selected)
    mse_bayescv = mean_squared_error(y_val[target_var], y_test_predcv)
    print(f"Prediction MSE (BayesSearchCV): {mse_bayescv}")

    # Step 2: Optimize hyperparameters using Optuna
    def objective_optuna(trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 5, 7),
            'gamma': trial.suggest_float('gamma', 0, 1.6),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.4, 6.4),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.4, 6.4), 
            'n_estimators': trial.suggest_int('n_estimators', 50, 200)
        }
        model = XGBRegressor(**params, objective='reg:squarederror', random_state=42)
        scores = cross_val_score(model, X_train_selected, y_train[target_var], cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
        return -np.mean(scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective_optuna, n_trials=25)

    best_params_optuna = study.best_params
    model_optuna = XGBRegressor(**best_params_optuna, objective='reg:squarederror', random_state=42)
    model_optuna.fit(X_train_selected, y_train[target_var])
    test_prediction_optuna = model_optuna.predict(X_val_selected)
    mse_optuna = mean_squared_error(y_val[target_var], test_prediction_optuna)
    print(f"Prediction MSE (Optuna): {mse_optuna}")

    # Step 3: Compare results and select the best model
    results = {
        'BayesSearchCV': {'params': best_params_bayescv, 'mse': mse_bayescv},
        'Optuna': {'params': best_params_optuna, 'mse': mse_optuna}
    }

    best_method = min(results, key=lambda k: results[k]['mse'])
    best_params_final = results[best_method]['params']
    print(f"Best method: {best_method}")
    print(f"Best parameters: {best_params_final}")

    # Step 4: Train the final model with best params on all available data
    final_model = XGBRegressor(**best_params_final, objective='reg:squarederror', random_state=42)
    X_train_val_selected = pd.concat([X_train_selected, X_val_selected])
    y_train_val = pd.concat([y_train, y_val])
    final_model.fit(X_train_val_selected, y_train_val[target_var])

    # Evaluate model on train, val, and test sets
    y_train_pred = final_model.predict(X_train_selected)
    y_val_pred = final_model.predict(X_val_selected)
    y_test_pred = final_model.predict(X_test_selected)

    print("\n=== Train Performance ===")
    print(f"MSE: {mean_squared_error(y_train[target_var], y_train_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_train[target_var], y_train_pred):.4f}")
    print(f"R²: {r2_score(y_train[target_var], y_train_pred):.4f}")

    print("\n=== Validation Performance ===")
    print(f"MSE: {mean_squared_error(y_val[target_var], y_val_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_val[target_var], y_val_pred):.4f}")
    print(f"R²: {r2_score(y_val[target_var], y_val_pred):.4f}")

    print("\n=== Test Performance ===")
    print(f"MSE: {mean_squared_error(y_test[target_var], y_test_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_test[target_var], y_test_pred):.4f}")
    print(f"R²: {r2_score(y_test[target_var], y_test_pred):.4f}")

    return final_model, best_params_final, y_train_pred, y_val_pred, y_test_pred