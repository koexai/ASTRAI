from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

sym_lums_path = r"C:\Users\39320\Desktop\ASTRAI\four parameter synthetic dataset\analyticModelEXPSOE_Run1_20230328_07-55-00.npy"
attributes_path = r"C:\Users\39320\Desktop\ASTRAI\four parameter synthetic dataset\lista_amEXPSOE.csv"

# --- 1. Data Loading ---
# Load synthetic light curve data (Targets)
sym_lums = np.load(sym_lums_path)
# Load physical attributes (Features)
attributes = pd.read_csv(attributes_path, sep=";")

# --- 2. Data Preprocessing ---

# Initialize scaler for the target variable (Light Curves)
yscaler = MinMaxScaler()
Y = yscaler.fit_transform(sym_lums)  # Normalize targets to [0, 1] range

# Initialize scaler for input features (Physical Attributes)
xscaler = MinMaxScaler()
# Skip the first column (Index/ID) and use the rest as input features
X = xscaler.fit_transform(np.array(attributes[attributes.columns[1:]]))

# Split data into training and testing sets (50/50 split)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=42)

# --- 3. Hyperparameter Tuning (Grid Search) ---

# Define the search space for hyperparameters
param_grid = {
    'n_estimators': [50, 100, 200],        # Number of trees in the forest
    'max_depth': [5, 10, 15, None],        # Max depth of trees (None = unlimited)
    'min_samples_split': [2, 4, 6],        # Min samples to split a node
    'min_samples_leaf': [1, 2, 4],         # Min samples required at a leaf node
    'max_features': ['sqrt', 'log2', None] # Number of features to consider for best split
}

# Configure GridSearchCV
grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    scoring='neg_root_mean_squared_error',  # Metric: Negative RMSE (higher is better for sklearn)
    cv=5, # 5-fold Cross-Validation
    n_jobs=-1,  # Use all available CPU cores
    verbose=1   # Print progress updates
)

# Run the search
print("Starting Grid Search...")
grid_search.fit(X_train, y_train)
print("Grid Search completed!")

# Display the best hyperparameters found
print("\nGrid Search Results:")
print("Best parameters:", grid_search.best_params_)
print("Best RMSE (CV):", -grid_search.best_score_)  # Flip sign to view positive RMSE

# --- 4. Model Comparison (Baseline vs Optimized) ---
print("\n=== MODEL COMPARISON ===")

# Define a baseline "Original" model with fixed parameters
regressor_original = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    random_state=42
)

# Train the baseline model
regressor_original.fit(X_train, y_train)
# Calculate R^2 Score on test set
score_original = regressor_original.score(X_test, y_test)
print("Original Model - R² score:", score_original)

# Get the optimized model from Grid Search
best_model = grid_search.best_estimator_
# Calculate R^2 Score for optimized model
score_best = best_model.score(X_test, y_test)
print("Best Model - R² score:", score_best)

# --- 5. Final Evaluation (RMSE) ---
# Generate predictions on the test set
y_pred_original = regressor_original.predict(X_test)
y_pred_best = best_model.predict(X_test)

# Calculate RMSE for both models
rmse_original = np.sqrt(mean_squared_error(y_test, y_pred_original))
rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))

print("Original Model - RMSE test:", rmse_original)
print("Best Model - RMSE test:", rmse_best)

if rmse_best < rmse_original:
    print(f"RMSE Improvement: {rmse_original - rmse_best:.4f}")
else:
    print(f"RMSE Worsened: {rmse_best - rmse_original:.4f}")