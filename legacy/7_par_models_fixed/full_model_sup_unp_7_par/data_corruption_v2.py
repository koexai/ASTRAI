import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C

# Functions extracted from Augmentation.py
# (Sporcation) funzione di augmentation per aggiungere rumore
# ---

# ---
# Original Functions of this .py (by Vincenzo)

#def add_gaussian_noise(X, noise_std):
 #   """
  #  Add Gaussian noise to each series in the batch.
   # X: numpy array of shape (batch_size, series_length)
    #noise_std: standard deviation of the noise
    #"""
    #fast_pseudo_rands = np.tile(np.random.randn((len(X))), len(X[0])) #Return a sample (or samples) from the "standard normal" distribution.
    #return X + noise_std * fast_pseudo_rands.reshape(*X.shape)  # np.random.randn(*X.shape)*X.shape

def add_gaussian_noise_v2(X, noise_std):
    """
    Aggiunge rumore gaussiano con deviazione standard `noise_std`
    a ogni elemento dell'array `X` (shape: [batch_size, series_length]).
    """
    noise = np.random.randn(*X.shape)  # genera rumore N(0, 1)
    return X + noise_std * noise       # scala per la std e lo somma

def add_nan_patches_batch(X, n_patches=2, patch_length=5, n_singles=300):
    """
    Introduce NaN patches e NaN singoli in ciascuna serie del batch.

    Parametri:
    - X: array NumPy di forma (batch_size, series_length)
    - n_patches: numero di blocchi di NaN consecutivi per serie
    - patch_length: lunghezza di ciascun blocco
    - n_singles: numero di NaN singoli sparsi da aggiungere

    Output:
    - corrupted: array con NaN inseriti
    """
    corrupted = X.copy().astype(float)
    batch_size, L = corrupted.shape
    max_start = L - patch_length

    for i in range(batch_size):
        # Aggiunta blocchi consecutivi
        for _ in range(n_patches):
            start = np.random.randint(0, max_start + 1)
            corrupted[i, start:start + patch_length] = np.nan

        # Aggiunta NaN singoli sparsi (evitando sovrapposizioni con i blocchi)
        nan_indices = np.isnan(corrupted[i])
        available_indices = np.where(~nan_indices)[0]  # Posizioni non ancora NaN
        if len(available_indices) > n_singles:
            single_nan_positions = np.random.choice(available_indices, size=n_singles, replace=False)
            corrupted[i, single_nan_positions] = np.nan

    return corrupted

def interpolate_one(data):
    """
    Linearly interpolates NaNs along the time axis for a single series.

    Parameters:
    - data: numpy array of shape (series_length,) or (series_length, 1)

    Returns:
    - interpolated_values: numpy array of shape (series_length,) with NaNs interpolated.
    """
    # Ensure data is a 1-dimensional NumPy array
    data = np.asarray(data).ravel()

    # Find valid (non-NaN) data points
    valid_mask = ~np.isnan(data)
    valid_indices = np.where(valid_mask)[0]
    valid_data = data[valid_mask]

    # Handle cases with too few valid points for interpolation
    if len(valid_data) < 2:
        # If there are fewer than 2 valid points, linear interpolation is not possible.
        # Return the original data (which may still contain NaNs if they couldn't be interpolated)
        # or fill with a default value (e.g., mean, zero) if appropriate for your application.
        # For now, we return the original data as is.
        print(f"Warning: Fewer than 2 valid data points for interpolation. Returning original series.")
        return data

    # Ensure valid_indices and valid_data are explicitly 1D for np.interp
    valid_indices = np.asarray(valid_indices).ravel()
    valid_data = np.asarray(valid_data).ravel()

    # Perform linear interpolation
    # np.arange(len(data)) creates the target x-coordinates (0, 1, ..., len(data)-1)
    interpolated_values = np.interp(np.arange(len(data)), valid_indices, valid_data)
    return interpolated_values

# FIX 9: Single definition of interpolate_batch_fast (removed duplicate)
interpolate_batch_fast = np.vectorize(interpolate_one, signature='(n)->(n)')

def interpolate_batch(X):
    """
    Linearly interpolate NaNs along the time axis (axis=1) for each series.
    X: numpy array of shape (batch_size, series_length)
    """
    for id, x in enumerate(X):
        X[id] = interpolate_one(x)
    return X

def interpolate_gp_one(data, length_scale=0.01, nu=1.5, random_state=42, return_std=False):
    """
    Linearly interpolate NaNs along the time axis for one series.
    X: numpy array of shape (series_length)
    """
    valid_indices = np.where(np.logical_not(np.isnan(data)))[0] #selezione posizioni dei valori non NAN
    valid_data = data[valid_indices] #salva i valori non NAN secondo l'ordine
    matern_kernel = 1.0 * Matern(length_scale=length_scale, nu=nu)
    # Create a Gaussian Process Regressor for specific kernel
    gp_matern = GaussianProcessRegressor(kernel=matern_kernel,
                                alpha=1e-6,  # Small noise for numerical stability
                                normalize_y=True,  # Normalize target values
                                random_state=random_state)
    gp_matern.fit(valid_indices.reshape(-1, 1), valid_data)
    all_indices = np.arange(len(data)).reshape(-1, 1)
    interpolated_values = gp_matern.predict(all_indices)

    #return_std
    if return_std:
        interpolated, std = gp_matern.predict(all_indices, return_std=True)
        return interpolated, std
    return interpolated_values

def interpolate_gp_batch(X):
    """
    Linearly interpolate NaNs along the time axis (axis=1) for each series.
    X: numpy array of shape (batch_size, series_length)
    """
    for id, x in enumerate(X):
        X[id] = interpolate_gp_one(x)
    return X

interpolate_gp_batch_fast = np.vectorize(interpolate_gp_one, signature='(n)->(n)')  # this should be a class



def apply_corruption(X, noise=0.1, missing_days=90):
    noisy_X = add_gaussian_noise_v2(X, noise_std=noise)
    nan_X = add_nan_patches_batch(noisy_X, n_patches=1, patch_length=missing_days)
    interp_X = interpolate_batch_fast(nan_X)
    #interp_X = interpolate_gp_batch_fast(nan_X)
    return noisy_X, nan_X, interp_X


def test_corruption():
    # Demo on a synthetic batch
    lums = np.load("analyticModelEXPSOE_Run1_20230328_07-55-00.npy")

    xscaler = StandardScaler()  # MinMaxScaler()
    batch = xscaler.fit_transform(np.exp(lums[:, :300]))
    # create batches with different corruption level

    gold = apply_corruption(batch, noise=0.1, missing_days=30)
    silver = apply_corruption(batch, noise=0.15, missing_days=60)
    bronze = apply_corruption(batch, noise=0.3, missing_days=90)

    for s in np.random.random_integers(len(batch), size=(30)):
        # Plot first sample through each step
        fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        titles = ["Original", "Corrupted"]
        ok = fig.suptitle("Generated Corrupted samples")

        for corrupted, level, ax, in zip([gold, silver, bronze], ["gold", "silver", "bronze"], axs):
            ok = ax.plot(np.log(xscaler.inverse_transform(batch)[s]), "k")
            color = level
            if level == "bronze":
                color = "darkorange"
            ok = ax.plot(np.log(xscaler.inverse_transform(corrupted)[s]), '+', color=color)
            ok = ax.legend(titles)
            ok = ax.set_xlabel(level)

        ok = plt.ylabel('Magnitude')
        ok = plt.savefig("corrupted" + str(s).zfill(5) + ".png")


if __name__ == "__main__":
    test_corruption()
