import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import os

# Importiamo le tue funzioni custom
from functions_BRR import get_rmse, get_mae, get_r_squared, get_rrmse
from data_corruption_v2 import apply_corruption

# --- 1. Caricamento dati ---
base_path = r"/home/andrea/repo_koexai/ASTRAI/seven parameters dataset/ASTRAI DATASET/dataset_preprocessed.csv"
print(f"Caricamento del dataset: {base_path}...")
df = pd.read_csv(base_path)

param_names = ['Raggio', 'Massa', 'Energia', 'Nickel', 'Mcsm', 'Rcsm', 'Slope']
curve_cols = [str(i) for i in range(1601)]

X_raw = df[curve_cols].values
y_raw = df[param_names].copy()

print(f" Dataset caricato: {X_raw.shape[0]} campioni.")

# --- 2. Pre-processing ---
# Log-trasformazione dei parametri target
for col in param_names:
    y_raw[col] = np.log1p(y_raw[col])

# Scaling dei Target (Y)
y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y_raw)

# --- 3. 10-Fold Cross-Validation ---
n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
all_fold_results = []

print(f"\n Inizio {n_splits}-Fold CV con Random Forest (Parallel Mode)...")

for fold_idx, (train_index, test_index) in enumerate(kf.split(X_raw), 1):
    X_train_raw, X_test_raw = X_raw[train_index], X_raw[test_index]
    y_train, y_test = y_scaled[train_index], y_scaled[test_index]
    
    # --- CORRUZIONE INTERNA (Solo Training) ---
    # Per il training usiamo i dati corrotti/interpolati
    _, _, X_train_interp = apply_corruption(X_train_raw, noise=0.1, missing_days=90)
    
    # Per il test set, simuliamo la corruzione per testare la robustezza reale
    _, _, X_test_interp = apply_corruption(X_test_raw, noise=0.1, missing_days=90)
    
    # Scaling Input (X)
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train_interp)
    X_test_scaled = x_scaler.transform(X_test_interp)
    
    # --- MODEL TRAINING ---
    # n_jobs=-1 usa tutti i core assegnati dal job SLURM
    model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=15, 
        n_jobs=-1, 
        random_state=42,
        verbose=0
    )
    model.fit(X_train_scaled, y_train)
    
    # Predizione
    y_pred = model.predict(X_test_scaled)
    
    # Metriche del fold
    fold_metrics = {}
    for i, param in enumerate(param_names):
        fold_metrics[param] = {
            'RMSE': get_rmse(y_test[:, i], y_pred[:, i]),
            'RRMSE': get_rrmse(y_test[:, i], y_pred[:, i]),
            'MAE': get_mae(y_test[:, i], y_pred[:, i]),
            'R2': get_r_squared(y_test[:, i], y_pred[:, i])
        }
    
    all_fold_results.append(fold_metrics)
    print(f"✔️ Fold {fold_idx} completato.")

# --- 4. Calcolo Medie e Salvataggio Risultati ---
results_list = []
print("\n" + "="*85)
print(f"{'Parametro':12} | {'Avg RMSE':10} | {'Avg RRMSE':10} | {'Avg MAE':10} | {'Avg R2':10}")
print("="*85)

for param in param_names:
    p_rmse = np.mean([f[param]['RMSE'] for f in all_fold_results])
    p_rrmse = np.mean([f[param]['RRMSE'] for f in all_fold_results])
    p_mae = np.mean([f[param]['MAE'] for f in all_fold_results])
    p_r2 = np.mean([f[param]['R2'] for f in all_fold_results])
    
    print(f"{param:12} | {p_rmse:<10.4f} | {p_rrmse:<10.4f} | {p_mae:<10.4f} | {p_r2:<10.4f}")
    results_list.append([param, p_rmse, p_rrmse, p_mae, p_r2])

# Salva metriche in CSV per sicurezza
res_df = pd.DataFrame(results_list, columns=['Parametro', 'RMSE', 'RRMSE', 'MAE', 'R2'])
res_df.to_csv("risultati_finali_RF.csv", index=False)
print(f"\nRisultati salvati in 'risultati_finali_RF.csv'")

# --- 5. Grafico (Salvataggio in PNG) ---
plt.figure(figsize=(10, 6))
plt.bar(param_names, res_df['R2'], color='darkblue', alpha=0.7)
plt.ylabel("R-Squared Medio")
plt.title("Performance Random Forest - 10-Fold CV")
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.savefig("performance_RF.png") # Salvataggio su disco
print("Grafico salvato come 'performance_RF.png'")