import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge
from functions_BRR import get_mse, get_rmse, get_mae, get_r_squared, get_best_params
from data_corruption_v2 import apply_corruption

# Data Loading
df_params = pd.read_csv(r"C:\Projects\MyPythonProject\Data Augmentation - Models\bayesian network\lista_amEXPSOE.csv",
                        sep=';')
light_curves = np.load(
    r"C:\Projects\MyPythonProject\Data Augmentation - Models\bayesian network\analyticModelEXPSOE_Run1_20230328_07-55-00.npy")

print(f"CSV shape: {df_params.shape}")
print(f"NPY shape: {light_curves.shape}")

assert df_params.shape[0] == light_curves.shape[0], "Number of rows in CSV and .npy file must match"

# Data Preparation
X = light_curves
y = df_params[['Radius', 'Mass', 'Energy', 'Nichel']]

# Corruption and interpolation of light curves
noisy_X, nan_X, interp_X = apply_corruption(X, noise=0.1, missing_days=90)

# Remove rows with NaNs
mask = ~y.isnull().any(axis=1)
# X_clean = X[mask]
X_clean = interp_X[mask] # Use interpolated curves
y_clean = y[mask]

print(f"Valid samples: {len(X_clean)}")

# Log transformation for Nickel (for stabilization)
# y_clean['Nichel'] = np.log1p(y_clean['Nichel'])
y_clean = y[mask].copy()  # Avoid SettingWithCopy
# Apply log1p transformation to all strictly positive parameters
pos_cols = ['Radius', 'Mass', 'Energy', 'Nichel']
y_log = y_clean.copy()
y_log[pos_cols] = np.log1p(y_log[pos_cols])

# Scaling physical parameters (in log space)
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y_log)

# Scaling light curves
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X_clean)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.15, random_state=7)

# Bayesian Ridge model training with max_iter=300
print("Training Bayesian Ridge models with early stopping...")
models = {}
metrics = {}
histories = {}
param_names = ['Radius', 'Mass', 'Energy', 'Nichel']
all_rmse_per_param = []

for i, param in enumerate(param_names):
    print(f"\nTraining for parameter: {param}")

    model = BayesianRidge(
        max_iter=300,
        tol=1e-6,
        compute_score=True,
        verbose=False
    )

    model.fit(X_train, y_train[:, i])
    histories[param] = model.scores_

    y_pred = np.float64(model.predict(X_test))
    mse = get_mse(y_test[:, i], y_pred)
    rmse = get_rmse(y_test[:, i], y_pred)
    mae = get_mae(y_test[:, i], y_pred)
    r2 = get_r_squared(y_test[:, i], y_pred)

    models[param] = model
    metrics[param] = {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}
    all_rmse_per_param.append(rmse)

    print(f"Final performance for {param}:")
    print(f"R²: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Effective number of iterations: {model.n_iter_}")

    plt.figure(figsize=(10, 5))
    plt.plot(histories[param], label='Marginal Log-Likelihood')
    plt.title(f'Convergence for {param}')
    plt.xlabel('Iterations')
    plt.ylabel('Marginal Log-Likelihood')
    plt.legend()
    plt.grid(True)
    plt.show()

print("\nTraining completed!")
all_r2 = [metrics[param]['r2'] for param in param_names]
print(f"\nMean R²: {np.mean(all_r2):.4f}")
print(f"Min R²: {np.min(all_r2):.4f}")
print(f"Max R²: {np.max(all_r2):.4f}")
print(f"Mean RMSE: {np.mean(all_rmse_per_param):.4f}")

# Visualization of performance per parameter
plt.figure(figsize=(10, 6))
plt.bar(param_names, all_r2, color='skyblue')
plt.xlabel('Physical Parameter')
plt.ylabel('R² Score')
plt.title('Model Performance per Physical Parameter')
plt.ylim(0, 1)
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.show()

# Function to characterize new curves
def characterize_light_curves(models_dict, scaler_X, scaler_y, new_curves):
    if isinstance(new_curves, pd.DataFrame):
        curves_array = new_curves.values
    else:
        curves_array = new_curves
    curves_scaled = scaler_X.transform(curves_array)
    params_scaled = np.zeros((len(curves_array), len(param_names)))
    for i, param in enumerate(param_names):
        params_scaled[:, i] = models_dict[param].predict(curves_scaled)

    # Inverse transform from StandardScaler (back to log space)
    params_log = scaler_y.inverse_transform(params_scaled)

    # Inverse of log1p to return to the original positive space
    params_unscaled = np.expm1(params_log)

    # Numerical safety: clip any residual negative values due to numerical noise
    params_unscaled = np.clip(params_unscaled, a_min=0.0, a_max=None)

    return pd.DataFrame(params_unscaled, columns=param_names)

# Characterization test
print("\nTesting characterization on test samples...")
test_params_pred = characterize_light_curves(models, scaler_X, scaler_y, X_test)

# Prepare "Ground Truth" in physical space
y_test_log_inv = scaler_y.inverse_transform(y_test)      # Back to log space
y_test_phys     = np.expm1(y_test_log_inv)               # Back to physical space (≥0)



fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()
for i, param in enumerate(param_names):
    axes[i].scatter(y_test_phys[:, i], test_params_pred[param], alpha=0.6)
    lo, hi = y_test_phys[:, i].min(), y_test_phys[:, i].max()
    axes[i].plot([lo, hi], [lo, hi], 'r--')
    axes[i].set_xlabel(f'Real Value {param}')
    axes[i].set_ylabel(f'Predicted Value {param}')
    axes[i].set_title(f'Comparison for {param}')
    axes[i].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\nReal parameters (for comparison):")
print(pd.DataFrame(y_test_phys[:3], columns=param_names))

# Example characterization of new curves
print("\nExample: Characterizing new light curves...")
example_curves = X_test[:3]
characterized_params = characterize_light_curves(models, scaler_X, scaler_y, example_curves)
print("\nCharacterized parameters:")
print(characterized_params)

# Helper function for inverse_transform of a single feature
def inverse_single_feature(scaler, values, feature_idx, n_features):
    temp = np.zeros((len(values), n_features))
    temp[:, feature_idx] = values
    out_log = scaler.inverse_transform(temp)[:, feature_idx]  # Log space
    return np.expm1(out_log)                                  # Physical space

# Uncertainty analysis for a specific parameter
sample_param = 'Mass'
feature_idx  = param_names.index(sample_param)
model_unc    = models[sample_param]

# X Scaling (using the already scaled X_test)
# Note: If X_test is already scaled, ensure transform is not called twice.

# Predictions and std in the STANDARDIZED target space
y_pred_unc, std_unc = model_unc.predict(X_test, return_std=True)

# Sort by real value (still in standardized space)
order        = np.argsort(y_test[:, feature_idx])
y_test_sorted  = y_test[order, feature_idx]
y_pred_sorted  = y_pred_unc[order]
std_sorted     = std_unc[order]
lower_bound    = y_pred_sorted - 1.96 * std_sorted
upper_bound    = y_pred_sorted + 1.96 * std_sorted

# Inverse transform: scaler_y -> log space -> expm1 -> physical space
y_test_inv = inverse_single_feature(scaler_y, y_test_sorted, feature_idx, len(param_names))
y_pred_inv = inverse_single_feature(scaler_y, y_pred_sorted, feature_idx, len(param_names))
lower_inv  = inverse_single_feature(scaler_y, lower_bound, feature_idx, len(param_names))
upper_inv  = inverse_single_feature(scaler_y, upper_bound, feature_idx, len(param_names))

# Safety clamp: values must be ≥ 0
y_test_inv = np.clip(y_test_inv, 0, None)
y_pred_inv = np.clip(y_pred_inv, 0, None)
lower_inv  = np.clip(lower_inv, 0, None)
upper_inv  = np.clip(upper_inv, 0, None)



plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label=f'Real values ({sample_param})', linewidth=2)
plt.plot(y_pred_inv, label=f'Predictions ({sample_param})', linewidth=2)
plt.fill_between(range(len(y_pred_inv)), lower_inv, upper_inv, alpha=0.3, label='95% Confidence Interval')
plt.xlabel("Sorted Samples"); plt.ylabel(sample_param)
plt.title(f"Bayesian Ridge: Uncertainty for parameter {sample_param}")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

# Hyperparameter tuning
print("\nHyperparameter tuning...")
n_iter_values = [1, 3, 10, 30, 100]
alpha_1_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
alpha_2_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
lambda_1_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
lambda_2_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

for param in param_names:
    print(f"\nTuning for parameter {param}:")
    y_train_single = y_train[:, param_names.index(param)]
    y_test_single = y_test[:, param_names.index(param)]
    best_params = get_best_params(
        max_iter_values=n_iter_values,
        alpha_1_values=alpha_1_values,
        alpha_2_values=alpha_2_values,
        lambda_1_values=lambda_1_values,
        lambda_2_values=lambda_2_values,
        X_train=X_train,  # Training on scaled features
        y_train=y_train_single,  # Target in standardized space
        X_test=X_test, 
        y_test=y_test_single,
        metric='rmse'
    )

    print(f"Best parameters: {best_params}")