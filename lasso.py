from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Lasso
from sklearn.feature_selection import VarianceThreshold
import numpy as np

def train_lasso_model(X_train, X_val, X_test, y_train, y_val, y_test, target_var):

    print("=== Phase 1: Initial Feature Preprocessing ===")
    # Step 1: Remove low-variance features
    selector = VarianceThreshold(threshold=0.01)
    X_train_filtered = selector.fit_transform(X_train)
    selected_feature_indices = selector.get_support(indices=True)
    selected_features = X_train.columns[selected_feature_indices]
    print(f"Removed {X_train.shape[1] - len(selected_features)} low-variance features")
    
    # Apply to all datasets
    X_train_reduced = X_train[selected_features]
    X_val_reduced = X_val[selected_features]
    X_test_reduced = X_test[selected_features]
    
    # Step 2: Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_reduced)
    X_val_scaled = scaler.transform(X_val_reduced)
    X_test_scaled = scaler.transform(X_test_reduced)
    
    # Prepare target variables
    y_train_target = y_train[target_var].values.ravel()
    y_val_target = y_val[target_var].values.ravel()

    print("\n=== Phase 2: Hyperparameter Tuning ===")
    # Time series cross-validation for alpha selection
    tscv = TimeSeriesSplit(n_splits=5)
    alphas = np.logspace(-4, 2, 50)  # Wider range of alphas
    
    lasso_grid = GridSearchCV(
        Lasso(max_iter=10000, tol=1e-4),
        param_grid={'alpha': alphas},
        cv=tscv,
        scoring='neg_mean_squared_error'
    )
    lasso_grid.fit(X_train_scaled, y_train_target)
    
    best_alpha = lasso_grid.best_params_['alpha']
    print(f"Best alpha from CV: {best_alpha:.4f}")

    print("\n=== Phase 3: Feature Selection ===")
    # Train with best alpha to get coefficients
    lasso = Lasso(alpha=best_alpha, max_iter=10000)
    lasso.fit(X_train_scaled, y_train_target)
    
    # Select non-zero coefficient features
    selected_mask = lasso.coef_ != 0
    selected_features = selected_features[selected_mask]
    print(f'Selected features:\n{selected_features}\n{len(selected_features)} features')
    
    
    # Filter datasets to only selected features
    X_train_final = X_train_reduced[selected_features]
    X_val_final = X_val_reduced[selected_features]
    X_test_final = X_test_reduced[selected_features]
    
    # Re-scale the selected features
    final_scaler = StandardScaler()
    X_train_final_scaled = final_scaler.fit_transform(X_train_final)
    X_val_final_scaled = final_scaler.transform(X_val_final)
    
    print("\n=== Phase 4: Final Model Training ===")
    # Combine train + val for final model
    X_train_val = np.vstack([X_train_final_scaled, X_val_final_scaled])
    y_train_val = np.concatenate([y_train_target, y_val_target])
    
    final_model = Lasso(alpha=best_alpha, max_iter=10000)
    final_model.fit(X_train_val, y_train_val)

    
    print("\n=== Phase 5: Evaluation ===")
    # Evaluate model on train, val, and test sets
    y_train_pred = lasso.predict(X_train_scaled)
    y_val_pred = lasso.predict(X_val_scaled)
    y_test_pred = lasso.predict(X_test_scaled)

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
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train_pred, y_val_pred, y_test_pred, lasso