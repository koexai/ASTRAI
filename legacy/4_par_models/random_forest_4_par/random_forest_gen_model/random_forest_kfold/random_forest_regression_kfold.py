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

# INPUT (X): I 7 parametri fisici
param_names = ['Raggio', 'Massa', 'Energia', 'Nickel', 'Mcsm', 'Rcsm', 'Slope']
# OUTPUT (y): Le 1601 magnitudini della curva di luce
curve_cols = [str(i) for i in range(1601)]

X_raw = df[param_names].copy()
y_raw = df[curve_cols].values

print(f"Dataset caricato. Righe: {len(df)}")
print(f"Esempio Input (X): {param_names}")

# --- 2. Pre-processing ---
# Log-trasformazione degli input fisici per stabilizzare il range
for col in param_names:
    X_raw[col] = np.log1p(X_raw[col])

# Scaling degli Input (X) e Target (y)
x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_scaled = x_scaler.fit_transform(X_raw)
y_scaled = y_scaler.fit_transform(y_raw)

# --- 3. 10-Fold Cross-Validation ---
n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
all_fold_results = []

print(f"\n Inizio {n_splits}-Fold CV Generativo (Parametri -> Curve)...")

for fold_idx, (train_index, test_index) in enumerate(kf.split(X_scaled), 1):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    # In questo modello, le curve sono il target (y)
    y_train_clean, y_test_clean = y_scaled[train_index], y_scaled[test_index]
    
    # --- CORRUZIONE INTERNA (Sui Target del Training) ---
    # Invertiamo momentaneamente lo scaling per applicare la corruzione fisica
    y_train_phys = y_scaler.inverse_transform(y_train_clean)
    _, _, y_train_interp_phys = apply_corruption(y_train_phys, noise=0.1, missing_days=90)
    
    # Ri-applichiamo lo scaling ai dati corrotti/interpolati per il training
    y_train_interp_scaled = y_scaler.transform(y_train_interp_phys)
    
    # --- MODEL TRAINING ---
    # Random Forest gestisce nativamente output multipli (le 1601 colonne)
    model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=15, 
        n_jobs=-1,  # Sfrutta tutti i core assegnati su Leonardo
        random_state=42
    )
    model.fit(X_train, y_train_interp_scaled)
    
    # --- GENERAZIONE (Predizione) ---
    y_gen_scaled = model.predict(X_test)
    
    # --- CALCOLO METRICHE (Confronto tra Generato e Verità Fisica del Test Set) ---
    # Calcoliamo le metriche medie sull'intera curva (1601 punti)
    f_rmse = get_rmse(y_test_clean, y_gen_scaled)
    f_rrmse = get_rrmse(y_test_clean, y_gen_scaled)
    f_mae = get_mae(y_test_clean, y_gen_scaled)
    f_r2 = get_r_squared(y_test_clean, y_gen_scaled)
    
    all_fold_results.append({
        'fold': fold_idx,
        'RMSE': f_rmse,
        'RRMSE': f_rrmse,
        'MAE': f_mae,
        'R2': f_r2
    })
    
    print(f"✔️ Fold {fold_idx} completato. R2 medio sulla curva: {f_r2:.4f}")

# --- 4. Risultati Finali e Media ---
print("\n" + "="*85)
print(f"{'Fold':12} | {'RMSE':10} | {'RRMSE':10} | {'MAE':10} | {'R2':10}")
print("="*85)

for m in all_fold_results:
    print(f"Fold {m['fold']:<7} | "
          f"{m['RMSE']:<10.4f} | "
          f"{m['RRMSE']:<10.4f} | "
          f"{m['MAE']:<10.4f} | "
          f"{m['R2']:<10.4f}")

# Calcolo MEDIA FINALE
final_rmse = np.mean([x['RMSE'] for x in all_fold_results])
final_rrmse = np.mean([x['RRMSE'] for x in all_fold_results])
final_mae = np.mean([x['MAE'] for x in all_fold_results])
final_r2 = np.mean([x['R2'] for x in all_fold_results])

print("-" * 85)
print(f"{'MEDIA TOTALE':12} | {final_rmse:<10.4f} | {final_rrmse:<10.4f} | {final_mae:<10.4f} | {final_r2:<10.4f}")
print("="*85)

# Salvataggio CSV per analisi post-job
res_df = pd.DataFrame(all_fold_results)
res_df.to_csv("risultati_generativi_RF.csv", index=False)

# --- 5. Grafico di Stabilità ---
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), [x['R2'] for x in all_fold_results], 'o-', color='orange', label='R2 per Fold')
plt.axhline(y=final_r2, color='red', linestyle='--', label='Media Globale')
plt.title("Capacità Generativa Random Forest (Parametri -> Curve)")
plt.xlabel("Fold Index")
plt.ylabel("R-Squared Medio")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("performance_generativa_RF.png")
print("\n Risultati e grafici salvati con successo.")