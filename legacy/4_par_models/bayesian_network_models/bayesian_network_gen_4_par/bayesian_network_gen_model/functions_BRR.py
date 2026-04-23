import numpy as np
import itertools
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- 1. Custom Metric Functions ---
def get_mse(y, y_pred):
    """Mean Square Error (MSE)"""
    return np.mean((y - y_pred) ** 2)

def get_rmse(y, y_pred):
    """Root Mean Squared Error (RMSE)"""
    return np.sqrt(get_mse(y, y_pred))

def get_mae(y, y_pred):
    """Mean Absolute Error (MAE)"""
    return np.mean(np.abs(y - y_pred))

def get_r_squared(y, y_pred):
    """R-squared (coefficient of determination)."""
    return 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

# --- 2. Manual grid search for Bayesian Ridge Regression ---
def get_best_params(max_iter_values, alpha_1_values, alpha_2_values, lambda_1_values, lambda_2_values,
                    metric='rmse', X_train=None, y_train=None, X_test=None, y_test=None):
    """
    Perform a manual grid search to find the best hyperparameters
    for a BayesianRidge model according to a chosen metric.
    """

    # Safety check: all datasets must be provided
    if X_train is None or y_train is None or X_test is None or y_test is None:
        raise ValueError("You must provide X_train, y_train, X_test, and y_test")

    # Generate all possible combinations of hyperparameters
    combinations = itertools.product(max_iter_values, alpha_1_values, alpha_2_values, lambda_1_values, lambda_2_values)

    # Variables to store the best result
    best_params = None
    best_value = None

    # Loop over all hyperparameter combinations
    for n_iter, alpha_1, alpha_2, lambda_1, lambda_2 in combinations:

        # Initialize the Bayesian Ridge model
        model = BayesianRidge(max_iter=n_iter,
                              alpha_1=alpha_1,
                              alpha_2=alpha_2,
                              lambda_1=lambda_1,
                              lambda_2=lambda_2)

        # Fit the model on the training data
        model.fit(X_train, y_train)

        # Predict on the test set
        y_pred = model.predict(X_test)

        # Metric selection and evaluation
        if metric == 'rmse':
            value = np.sqrt(mean_squared_error(y_test, y_pred))
            better = (best_value is None) or (value < best_value)

        elif metric == 'mse':
            value = mean_squared_error(y_test, y_pred)
            better = (best_value is None) or (value < best_value)

        elif metric == 'mae':
            value = mean_absolute_error(y_test, y_pred)
            better = (best_value is None) or (value < best_value)

        elif metric == 'r_squared':
            value = r2_score(y_test, y_pred)
            better = (best_value is None) or (value > best_value)

        elif metric == 'uncertainty':

            """
            Uncertainty measure:
            - sigma_ is the posterior covariance matrix of the weights
            - the trace represents the total variance
            """
            sigma = model.sigma_
            value = np.sqrt(np.trace(sigma))
            better = (best_value is None) or (value < best_value)

        else:
            raise ValueError(f"Metric '{metric}' is not supported")

        # Update best parameters if current model is better
        if better:
            best_value = value
            best_params = {
                'n_iter': n_iter,
                'alpha_1': alpha_1,
                'alpha_2': alpha_2,
                'lambda_1': lambda_1,
                'lambda_2': lambda_2
            }
    # Return the best hyperparameter set and its metric value
    return best_params, best_value