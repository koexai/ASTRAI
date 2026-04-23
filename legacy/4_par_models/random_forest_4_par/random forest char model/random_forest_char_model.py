from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

# --- Data Loading ---
# LIGHT CURVES: These will be our features (X)
lums = np.load("analyticModelEXPSOE_Run1_20230328_07-55-00.npy")  
# PHYSICAL PARAMETERS: These will be our targets (Y)
attributes = pd.read_csv("lista_amEXPSOE.csv", sep=";")            

# --- Data Scaling ---
xscaler = MinMaxScaler()
X = xscaler.fit_transform(lums)  # Light curves are now inputs (X)

yscaler = MinMaxScaler()
# Parameters are now outputs (Y). Assuming first column is an ID/index, using [1:]
Y = yscaler.fit_transform(np.array(attributes[attributes.columns[1:]]))  

# --- Train/Test Split ---
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=42)

# --- Parameter Grid for Optimization ---
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# --- GridSearchCV Setup ---
# GridSearchCV automates the process of testing different combinations of hyperparameters
grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    scoring='neg_root_mean_squared_error',  # Negative RMSE for compatibility with GridSearch
    cv=5,
    n_jobs=-1,
    verbose=1
)



print(" Starting Grid Search...")
grid_search.fit(X_train, Y_train)
print("Grid Search completed!")

# Best parameters found
print("Best parameters:", grid_search.best_params_)
print("Best RMSE (CV):", -grid_search.best_score_)

# --- Model Comparison ---
print("\n=== MODEL COMPARISON ===")

# Baseline Model (without optimization)
regressor_original = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42
)
regressor_original.fit(X_train, Y_train)
Y_pred_original = regressor_original.predict(X_test)
rmse_original = np.sqrt(mean_squared_error(Y_test, Y_pred_original))
r2_original = r2_score(Y_test, Y_pred_original)

print(f"Original Model - Test RMSE: {rmse_original:.4f}")
print(f"Original Model - R² score: {r2_original:.4f}")

# Best Model from Grid Search
best_model = grid_search.best_estimator_
Y_pred_best = best_model.predict(X_test)
rmse_best = np.sqrt(mean_squared_error(Y_test, Y_pred_best))
r2_best = r2_score(Y_test, Y_pred_best)

print(f"Best Model - Test RMSE: {rmse_best:.4f}")
print(f"Best Model - R² score: {r2_best:.4f}")

# Improvement check
if rmse_best < rmse_original:
    print(f" RMSE Improvement: {rmse_original - rmse_best:.4f}")
else:
    print(f" RMSE Degradation: {rmse_best - rmse_original:.4f}")