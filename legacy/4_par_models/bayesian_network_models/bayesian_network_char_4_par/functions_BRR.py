import numpy as np
import itertools
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Metriche
def get_mse(y, y_pred):
    return np.mean((y - y_pred) ** 2)

def get_rmse(y, y_pred):
    return np.sqrt(get_mse(y, y_pred))

def get_mae(y, y_pred):
    return np.mean(np.abs(y - y_pred))

def get_r_squared(y, y_pred):
    return 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)


def get_best_params(max_iter_values, alpha_1_values, alpha_2_values, lambda_1_values, lambda_2_values,
                    metric='rmse', X_train=None, y_train=None, X_test=None, y_test=None):
    if X_train is None or y_train is None or X_test is None or y_test is None:
        raise ValueError("Devi fornire X_train, y_train, X_test e y_test")

    combinations = itertools.product(max_iter_values, alpha_1_values, alpha_2_values, lambda_1_values, lambda_2_values)

    best_params = None
    best_value = None

    for n_iter, alpha_1, alpha_2, lambda_1, lambda_2 in combinations:
        # Crea e allena il modello
        model = BayesianRidge(max_iter=n_iter,
                              alpha_1=alpha_1,
                              alpha_2=alpha_2,
                              lambda_1=lambda_1,
                              lambda_2=lambda_2)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # Calcola la metrica richiesta
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
            # misura incertezza come radice della traccia di sigma_
            sigma = model.sigma_
            value = np.sqrt(np.trace(sigma))
            better = (best_value is None) or (value < best_value)

        else:
            raise ValueError(f"Metrica '{metric}' non supportata")

        if better:
            best_value = value
            best_params = {
                'n_iter': n_iter,
                'alpha_1': alpha_1,
                'alpha_2': alpha_2,
                'lambda_1': lambda_1,
                'lambda_2': lambda_2
            }

    return best_params, best_value