import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim

from data_corruption import apply_corruption

sym_lum_path = r"C:\Users\39320\Desktop\ASTRAI\four parameter synthetic dataset\analyticModelEXPSOE_Run1_20230328_07-55-00.npy"
attributes_path = r"C:\Users\39320\Desktop\ASTRAI\four parameter synthetic dataset\lista_amEXPSOE.csv"

class ResidualBlock(nn.Module):
    """
    A standard Residual Block component.
    It helps train deeper networks by allowing gradients to flow through
    the 'skip connection' (x += residual).
    Structure: Input -> BN -> LeakyReLU -> Dropout -> Linear -> Add Input
    """
    def __init__(self, in_features, dropout):
        super(ResidualBlock, self).__init__()
        # LoRA (Low-Rank Adaptation) optimization lines are defined but not currently used in forward()
        self.fc1a = nn.Linear(in_features, in_features // 8)
        self.fc1b = nn.Linear(in_features // 8, in_features)
        self.leaky = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(in_features) # Normalizes input to stabilize learning
        self.dropout = dropout
        if self.dropout:
            self.drop = nn.Dropout()

    def forward(self, x):
        residual = x # Save input for skip connection

        # Pre-activation architecture: BN -> Activation -> Dropout -> Linear
        x = self.bn1(x)
        x = self.leaky(x)
        if self.dropout:
            x = self.drop(x)

        # Note: The code here actually uses the LoRA-style split layers (fc1a -> fc1b)
        # instead of a single dense layer. This creates a bottleneck structure.
        x = self.fc1a(x)
        x = self.fc1b(x)
        x += residual # Add the original input back
        return x


class MLPWithResiduals(nn.Module):
    """
    Main Neural Network class.
    Constructs a Multi-Layer Perceptron using a stack of Residual Blocks.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, depth, dropout):
        super(MLPWithResiduals, self).__init__()
        # Initial projection layer
        layers = [nn.Linear(input_dim, hidden_dim)]

        # Stack 'depth' number of residual blocks
        for _ in range(depth):
            layers.append(ResidualBlock(hidden_dim, dropout))

        # Final projection layer to output dimension
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.res_blocks = nn.Sequential(*layers)

    def forward(self, x):
        return self.res_blocks(x)

    def fit(self, X, y, epochs=1000, lr=1e-3, verbose=True, device='cpu', corruption=False):
        """
        Training loop with optional Data Augmentation (corruption).
        """
        self.to(device)
        self.train()

        # If not using dynamic corruption, load X onto GPU once
        if not corruption:
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

        # Ensure target shape is correct (batch, features)
        if len(y_tensor.shape) == 1:
            y_tensor = y_tensor.unsqueeze(1)

        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            # If corruption is enabled, generate fresh noisy data every epoch (Denoising approach)
            if corruption:
                X_corr = apply_corruption(X)
                X_tensor = torch.tensor(X_corr, dtype=torch.float32).to(device)

            optimizer.zero_grad()
            outputs = self.forward(X_tensor)
            loss = criterion(outputs, y_tensor)

            # --- Regularization Placeholder ---
            # If the output is a time-series (length > 100), we might want to add smoothness constraints
            if outputs.shape[1] > 100:
                pass # Logic for derivative regularization could go here

            loss.backward()
            optimizer.step()

            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    def predict(self, X, device='cpu', corruption=False):
        """Performs inference, optionally applying corruption first for testing robustness."""
        if corruption:
            X_corr = apply_corruption(X)
            X_tensor = torch.tensor(X_corr, dtype=torch.float32).to(device)
        else:
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

        self.to(device)
        self.eval()

        with torch.no_grad():
            outputs = self.forward(X_tensor)
        return outputs.cpu().numpy()

    def score(self, X, y, device='cpu', corruption=False):
        y_pred = self.predict(X, device, corruption)
        return r2_score(y, y_pred)


def train_test_char(depth, width, n_epochs, dropout, corruption=False, verbose=False):
    """
    Trains and Tests the 'Characterization' model (Input: Light Curve -> Output: Params).
    Evaluates the model's robustness against different noise levels (Gold, Silver, Bronze).
    """

    # 421 input points (Light Curve) -> 4 output parameters
    regr = MLPWithResiduals(421, width, 4, depth, dropout)
    regr.fit(X_train, y_train, verbose=verbose, epochs=n_epochs, corruption=corruption)

    # Predict on clean data
    y_pred_sym = regr.predict(X_test, corruption=False)

    # Predict on corrupted data (simulation of real-world bad observation conditions)
    # Gold: Low noise, small gaps
    y_pred_gold = regr.predict(apply_corruption(X_test, noise=0.03, missing_days=30), corruption=False)
    # Silver: Medium noise
    y_pred_silver = regr.predict(apply_corruption(X_test, noise=0.05, missing_days=60), corruption=False)
    # Bronze: High noise, large gaps
    y_pred_bronze = regr.predict(apply_corruption(X_test, noise=0.1, missing_days=90), corruption=False)

    # # Calculate RMSE for each scenario
    medals = []
    for y_pred in [y_pred_sym, y_pred_gold, y_pred_silver, y_pred_bronze]:
        medals.append(np.sqrt(mean_squared_error(y_test, y_pred)))

    if verbose:
        print(f"RMSE levels: {medals}, depth: {depth}, width: {width}, epochs: {n_epochs}, dropout: {dropout}")

    # Return the average error across all quality levels as the fitness score
    return np.mean(medals)


def train_test_gen(depth, width, n_epochs, dropout, verbose=False):
    """
    Trains and Tests the 'Generative' model (Input: Params -> Output: Light Curve).
    """
    # 4 input parameters -> 421 output points (Light Curve)
    lcgen = MLPWithResiduals(4, width, 421, depth, dropout)

    # Note: X and y are swapped here because we are reversing the mapping
    lcgen.fit(y_train, X_train, verbose=verbose, epochs=n_epochs)
    X_pred = lcgen.predict(y_test)

    # Calculate reconstruction error
    gen_rmse = np.sqrt(mean_squared_error(X_test, X_pred, multioutput='raw_values'))

    # Scale RMSE back to original time units (optional normalization step)
    time_rmse = gen_rmse * xscaler.data_range_

    if verbose:
        print(f"Gen RMSE: {np.mean(gen_rmse):.4f}, Time RMSE: {np.mean(time_rmse):.4f}")
        print(f"Params - depth: {depth}, width: {width}, epochs: {n_epochs}, dropout: {dropout}")

    return np.mean(gen_rmse)


def hyperparam_search(model_type='char', exp_name='experiment'):
    """
        Executes a Grid Search to find the optimal hyperparameters.

        Args:
            model_type (str): 'char' for Characterization model, 'gen' for Generative model.
            exp_name (str): Suffix for saving files.

        Returns:
            dict: The best parameters found.
        """

    # --- Hyperparameter Grid ---
    depths = [1, 2, 3, 4]  # Network depth (num residual blocks)
    widths = [8, 16, 32, 64, 128]  # Layer width (neurons per layer)
    ns_epochs = [50, 100, 200]  # Training duration
    drops = [0, 0.1, 0.2, 0.3, 0.4, 0.5]  # Dropout rate

    nas_grid = []

    print(f"\nAvvio ricerca iperparametri per modello {model_type}")
    print("=" * 60)

    total_combinations = len(depths) * len(widths) * len(ns_epochs) * len(drops)
    current_combination = 0

    # Iterate through every combination
    for dropout in drops:
        for depth in depths:
            for width in widths:
                for n_epochs in ns_epochs:
                    current_combination += 1
                    print(f"Combinazione {current_combination}/{total_combinations}: "
                          f"depth={depth}, width={width}, epochs={n_epochs}, dropout={dropout}")

                    # Run training based on model type
                    if model_type == 'char':
                        # Uses corruption=True for robustness
                        rmse = train_test_char(depth, width, n_epochs, dropout,
                                               corruption=True, verbose=False)
                    else:  # model_type == 'gen'
                        rmse = train_test_gen(depth, width, n_epochs, dropout, verbose=False)

                    # Store results
                    nas_grid.append({
                        'rmse': rmse,
                        'depth': depth,
                        'width': width,
                        'n_epochs': n_epochs,
                        'dropout': dropout
                    })

                    print(f"RMSE: {rmse:.4f}")
                    print("-" * 40)

    # Save all results to DataFrame
    nas_df = pd.DataFrame(nas_grid)

    # Find best configuration (lowest RMSE)
    best_idx = nas_df['rmse'].idxmin()
    best_params = nas_df.loc[best_idx]

    print(f"\n MIGLIORI PARAMETRI TROVATI ({model_type.upper()}):")
    print("=" * 50)
    print(f"  Depth: {int(best_params['depth'])}")
    print(f"  Width: {int(best_params['width'])}")
    print(f"  N_epochs: {int(best_params['n_epochs'])}")
    print(f"  Dropout: {best_params['dropout']}")
    print(f"  RMSE: {best_params['rmse']:.4f}")
    print("=" * 50)

    # Save to CSV
    nas_df.to_csv(f"nas_results_{model_type}_{exp_name}.csv", index=False)

    # Generate analysis plots
    plot_hyperparameter_study(nas_df, model_type, exp_name)

    # Restituisce i migliori parametri come dizionario
    return {
        'depth': int(best_params['depth']),
        'width': int(best_params['width']),
        'n_epochs': int(best_params['n_epochs']),
        'dropout': best_params['dropout'],
        'rmse': best_params['rmse']
    }


def plot_hyperparameter_study(nas_df, model_type, exp_name):
    """
        Creates a 2x2 plot visualizing how RMSE changes with each hyperparameter.
        Shows mean error with standard deviation bars.
        """

    fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharey=True)
    fig.suptitle(f"RMSE vs Hyperparameters ({model_type.upper()} Model)", fontsize=16)

    params = ['depth', 'width', 'n_epochs', 'dropout']

    for idx, (ax, param) in enumerate(zip(axs.flatten(), params)):
        # Group by the specific parameter to see its isolated impact
        grouped = nas_df.groupby(param)['rmse'].agg(['mean', 'std', 'min', 'max']).reset_index()

        # Plot error bars
        ax.errorbar(grouped[param], grouped['mean'], yerr=grouped['std'],
                    marker='o', capsize=5, linewidth=2, markersize=6)

        # Highlight the global minimum (Best Model)
        min_idx = grouped['mean'].idxmin()
        ax.scatter(grouped.loc[min_idx, param], grouped.loc[min_idx, 'mean'],
                   color='red', s=100, marker='*', zorder=5, label='Best')

        ax.set_xlabel(param.replace('_', ' ').title())
        if idx % 2 == 0:
            ax.set_ylabel("RMSE")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"NAS_{model_type}_{exp_name}.png", dpi=300, bbox_inches='tight')
    plt.show()


def run_complete_nas(exp_name='experiment'):
    """
    Wrapper to run the full search for both Generative and Characterization models.
    """
    print("STARTING COMPLETE NEURAL ARCHITECTURE SEARCH")
    print("=" * 60)

    # # 1. Search for best Characterization model
    best_char_params = hyperparam_search('char', exp_name)

    # 2. Search for best Generative model
    best_gen_params = hyperparam_search('gen', exp_name)

    # 3. Final Verification Run
    print("\n FINAL VERIFICATION WITH BEST PARAMETERS")
    print("=" * 60)

    print("\nCharacterization Model:")
    final_char_rmse = train_test_char(
        best_char_params['depth'],
        best_char_params['width'],
        best_char_params['n_epochs'],
        best_char_params['dropout'],
        corruption=True,
        verbose=True
    )

    print("\nGenerative Model:")
    final_gen_rmse = train_test_gen(
        best_gen_params['depth'],
        best_gen_params['width'],
        best_gen_params['n_epochs'],
        best_gen_params['dropout'],
        verbose=True
    )

    return best_char_params, best_gen_params


# --- MAIN EXECUTION ---

# Load synthetic light curve data (luminosities)
sym_lums = np.load(sym_lum_path)
# Load physical attributes (parameters)
attributes = pd.read_csv(attributes_path, sep=";")

# Data Normalization (Scaling to 0-1 range)
xscaler = MinMaxScaler()
X = xscaler.fit_transform(sym_lums)
yscaler = MinMaxScaler()
# Skip first column (likely ID) and scale physical parameters
Y = yscaler.fit_transform(np.array(attributes[attributes.columns[1:]]))

# Split into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=42)

# Execution Entry Point
if __name__ == "__main__":
    # Runs the full search pipeline
    best_char, best_gen = run_complete_nas('light_curve_analysis')

    print("\n SEARCH COMPLETED")
    print(f"Best parameters saved and plots generated.")
