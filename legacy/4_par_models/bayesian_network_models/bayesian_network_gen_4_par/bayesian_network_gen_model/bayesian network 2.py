import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge
from functions_BRR import get_mse, get_rmse, get_mae, get_r_squared, get_best_params

sym_lum_path = r"C:\Users\39320\Desktop\ASTRAI\four parameter synthetic dataset\analyticModelEXPSOE_Run1_20230328_07-55-00.npy"
attributes_path = r"C:\Users\39320\Desktop\ASTRAI\four parameter synthetic dataset\lista_amEXPSOE.csv"

# --- 1. Data Loading ---
# Load physical parameters (Inputs: Radius, Mass, Energy, Nickel)
attributes = pd.read_csv(attributes_path,
                        sep=';')
# Load light curves (Targets: Luminosity over time)
sym_lums = np.load(sym_lum_path)

print(f"CSV shape: {attributes.shape}")
print(f"NPY shape: {sym_lums.shape}")

# Ensure consistency between inputs and targets
assert attributes.shape[0] == sym_lums.shape[0], "Numero di righe nel CSV e nel file .npy devono corrispondere"

# --- 2. Data Preparation ---
# Select specific physical attributes as features
X = attributes[['Radius', 'Mass', 'Energy', 'Nichel']]
y = sym_lums

# Remove rows containing NaN values to ensure data quality
mask = ~X.isnull().any(axis=1)
X_clean = X[mask]
y_clean = y[mask]

print(f"Valid samples: {len(X_clean)}")

# Standardize features (scale to mean=0, variance=1)
# Bayesian Ridge is sensitive to feature scaling because of the regularization priors.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

# Split into Training (85%) and Test (15%) sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_clean, test_size=0.15, random_state=7)

# --- 3. Model Training ---
# Instead of one big model, we train 421 separate Bayesian Ridge models,
# one for each discrete time step in the light curve.
print("Training Bayesian Ridge models...")
models = {}
metrics = {}
all_rmse_per_timepoint = [] # List to store RMSE for each time step

# Iterate through all 421 time points
for t in range(y_train.shape[1]):
    if t % 50 == 0:  # Print progress every 50 steps
        print(f"Punto temporale {t}/{y_train.shape[1]}")

    # Initialize and train Bayesian Ridge for time point t
    model = BayesianRidge()
    model.fit(X_train, y_train[:, t])

    # Evaluate on test set for this specific time point
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = get_mse(y_test[:, t], y_pred)
    rmse = get_rmse(y_test[:, t], y_pred)
    mae = get_mae(y_test[:, t], y_pred)
    r2 = get_r_squared(y_test[:, t], y_pred)

    # Store model and metrics
    models[t] = model
    metrics[t] = {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}
    all_rmse_per_timepoint.append(rmse)

print("Training completed!")

# --- 4. Global Evaluation ---
# Aggregate metrics across all time points
all_r2 = [metrics[t]['r2'] for t in range(421)]
#all_rmse = [metrics[t]['rmse'] for t in range(421)]

print(f"R² medio: {np.mean(all_r2):.4f}")
print(f"R² min: {np.min(all_r2):.4f}")
print(f"R² max: {np.max(all_r2):.4f}")
#print(f"RMSE medio: {np.mean(all_rmse):.4f}")
print(f"RMSE medio: {np.mean(all_rmse_per_timepoint):.4f}")

# Plot RMSE distribution over time
# This helps identify which phases of the supernova (early rise vs. late tail) are hardest to predict.
plt.figure(figsize=(14, 7))
time_axis = np.arange(y_train.shape[1]) # X-axis represents time points (0 to 420)

plt.plot(time_axis, all_rmse_per_timepoint, 'o-', color='darkgreen', markersize=4, linewidth=1.5)
plt.xlabel('Time Point')
plt.ylabel('RMSE (Root Mean Squared Error)')
plt.title('RMSE Variation along Light Curve Time Points')
plt.grid(True, linestyle='--', alpha=0.6)
# Add average line
plt.axhline(y=np.mean(all_rmse_per_timepoint), color='red', linestyle='--', label=f'RMSE Medio: {np.mean(all_rmse_per_timepoint):.4f}')
plt.legend()
plt.tight_layout()
plt.show()

print("\nRMSE plot for all time points generated.")

# --- 5. Full Curve Prediction ---
def predict_sym_lums(models_dict, scaler, new_params):
    """
    Reconstructs complete light curves from physical parameters by
    querying the 421 separate models sequentially.
    """
    # Extract values if input is a DataFrame
    if isinstance(new_params, pd.DataFrame):
        params_array = new_params[['Radius', 'Mass', 'Energy', 'Nichel']].values
    else:
        params_array = new_params

    # Scale input parameters using the training scaler
    params_scaled = scaler.transform(params_array)
    curves = np.zeros((len(params_array), 421))

    # Predict each time point individually
    for t in range(421):
        curves[:, t] = models_dict[t].predict(params_scaled)

    return curves


# Test prediction on test set samples
print("\nTesting prediction on test samples...")
# Inverse transform X_test because predict_sym_lums expects raw physical values (and scales them internally)
test_curves_pred = predict_sym_lums(models, scaler, scaler.inverse_transform(X_test))

# Visualize: Compare Real vs Predicted curves for 4 random samples
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()

for i in range(4):
    time_axis = np.arange(421)
    axes[i].plot(time_axis, y_test[i], label='Real', color='blue', linewidth=2)
    axes[i].plot(time_axis, test_curves_pred[i], label='Predicted', color='red', linewidth=2, linestyle='--')
    axes[i].set_xlabel('Time')
    axes[i].set_ylabel('Luminosity')
    axes[i].set_title(f'Sample {i + 1}')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --- 6. Performance Analysis ---
# Plot R2 score evolution over time
plt.figure(figsize=(12, 6))
time_points = list(range(0, 421, 10)) # Sample every 10th point
r2_values = [all_r2[t] for t in time_points]

plt.plot(time_points, r2_values, 'bo-', linewidth=2, markersize=4)
plt.xlabel('Time Point')
plt.ylabel('R² Score')
plt.title('Bayesian Ridge Performance over Time')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- 7. Uncertainty Analysis ---
# Bayesian Ridge provides variance estimates (uncertainty) for predictions.
sample_time = 200 # Analyze a specific time point (e.g., peak or tail)
model_unc = models[sample_time]
# predict(return_std=True) returns both mean prediction and standard deviation
y_pred_unc, std_unc = model_unc.predict(X_test, return_std=True)

# Sort data for cleaner visualization
order = np.argsort(y_test[:, sample_time])
y_test_sorted = y_test[order, sample_time]
y_pred_sorted = y_pred_unc[order]
std_sorted = std_unc[order]

# Calculate 95% Confidence Interval (± 1.96 * std)
lower_bound = y_pred_sorted - 1.96 * std_sorted
upper_bound = y_pred_sorted + 1.96 * std_sorted

plt.figure(figsize=(12, 6))
plt.plot(y_test_sorted, label=f'Real Values (t={sample_time})', color='blue', linewidth=2)
plt.plot(y_pred_sorted, label=f'Predictions (t={sample_time})', color='red', linewidth=2)
plt.fill_between(range(len(y_pred_sorted)), lower_bound, upper_bound,
                 color='orange', alpha=0.3, label='95% Confidence Interval')
plt.xlabel("Sorted Samples")
plt.ylabel("Luminosity")
plt.title(f"Bayesian Ridge: Uncertainty at Time Point {sample_time}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- 8. Generating Novel Curves ---
# Define new synthetic physical parameters
new_params_example = pd.DataFrame({
    'Radius': [10.0, 15.0, 20.0],
    'Mass': [1.4, 2.0, 2.5],
    'Energy': [1e51, 2e51, 3e51],
    'Nichel': [0.1, 0.15, 0.2]
})

print("\nGenerating curves from new parameters:")
print(new_params_example)

# Generate curves using the trained ensemble of models
generated_curves = predict_sym_lums(models, scaler, new_params_example)

plt.figure(figsize=(12, 6))
time_axis = np.arange(421)

for i in range(len(generated_curves)):
    plt.plot(time_axis, generated_curves[i],
             label=f'R={new_params_example.iloc[i]["Radius"]:.1f}, M={new_params_example.iloc[i]["Mass"]:.1f}',
             linewidth=2)

plt.xlabel('Time')
plt.ylabel('Luminosity')
plt.title('Light Curves Generated with Bayesian Ridge')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- 9. Hyperparameter Tuning ---
# Perform grid search for optimal Bayesian priors at key moments in the curve.
# Different phases (explosion, peak, decay) might require different regularization.
print("\nHyperparameter tuning...")
key_times = [0, 100, 200, 300, 420] # Key phases: Start, Rise, Peak/Plateau, Decay, Tail

# Grid of hyperparameters for Bayesian Ridge
# alpha: shape/scale for Gamma distribution over alpha (precision of weights)
# lambda: shape/scale for Gamma distribution over lambda (precision of noise)
n_iter_values = [100, 200, 300, 400, 500]
alpha_1_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
alpha_2_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
lambda_1_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
lambda_2_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

for t in key_times:
    if t < y_train.shape[1]:
        print(f"\nTuning for time point {t}:")

        y_train_single = y_train[:, t]
        y_test_single = y_test[:, t]

        best_params = get_best_params(max_iter_values=n_iter_values, # Changed n_iter_values to max_iter_values
                                      alpha_1_values=alpha_1_values,
                                      alpha_2_values=alpha_2_values,
                                      lambda_1_values=lambda_1_values,
                                      lambda_2_values=lambda_2_values,
                                      X_train=X_train, y_train=y_train_single,
                                      X_test=X_test, y_test=y_test_single,
                                      metric='rmse')
        print(f"Best parameters: {best_params}")


print(f"\nBayesian Ridge models trained for all {y_train.shape[1]} time points!")
print(f"Average Performance: R² = {np.mean(all_r2):.3f}, RMSE = {np.mean(all_rmse_per_timepoint):.4f}")