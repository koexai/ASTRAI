import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def add_gaussian_noise(X, noise_std):
    """
    Add Gaussian noise to each series in the batch.

    Args:
    X: Input numpy array of shape (batch_size, series_length).
    noise_std: Standard deviation (intensity) of the noise to add.

    Returns:
    The input array X with added noise.
    """
    # Generates a simplified noise array.
    # Note: This logic repeats a smaller set of random numbers across the array
    # to speed up generation ("fast_pseudo_rands"), rather than generating
    # unique noise for every single pixel.
    fast_pseudo_rands = np.tile(np.random.randn((len(X))), len(X[0]))

    # Reshapes the noise to match input dimensions and adds it
    return X + noise_std * fast_pseudo_rands.reshape(*X.shape)  # np.random.randn(*X.shape)*X.shape


def add_nan_patches_batch(X, n_patches, patch_length):
    """
    Simulates missing data by introducing patches of NaNs (Not a Number)
    into each time series in the batch.

    Args:
    X: Input data batch.
    n_patches: How many separate gaps to create per series.
    patch_length: The duration (number of time steps) of each gap.
    """
    corrupted = X.copy().astype(float)
    batch_size, L = corrupted.shape

    # Ensure the patch doesn't start too close to the end of the series
    max_start = L - patch_length
    for i in range(batch_size):
        for _ in range(n_patches):
            # Pick a random start time for the gap
            start = np.random.randint(0, max_start + 1)
            # Set values to NaN to simulate data loss
            corrupted[i, start:start + patch_length] = np.nan
    return corrupted


def interpolate_one(data):
    """
    Helper function to fill NaN values for a SINGLE time series using Linear Interpolation.
    """

    # Find indices where data is present (not NaN)
    valid_indices = np.where(np.logical_not(np.isnan(data)))[0]

    # Extract the valid values
    valid_data = data[valid_indices]

    # Use numpy's interpolation to fill the missing spots based on valid neighbors
    interpolated_values = np.interp(np.arange(len(data)), valid_indices, valid_data)
    return interpolated_values


def interpolate_batch(X):
    """
    Applies interpolation to an entire batch of data using a loop.
    Useful for cleaning up the 'NaN' patches created earlier.
    """
    for id, x in enumerate(X):
        X[id] = interpolate_one(x)
    return X


# Creates a vectorized version of the interpolation function.
# This allows applying 'interpolate_one' efficiently across arrays without writing explicit loops.
# Signature '(n)->(n)' tells numpy it maps a 1D array to a 1D array.
interpolate_batch_fast = np.vectorize(interpolate_one, signature='(n)->(n)')  # this should be a class


def apply_corruption_orig(X, noise=0.1, missing_days=90):
    """
    Original/Legacy version of corruption pipeline.
    Order: Add Noise -> Remove Data -> Interpolate.
    """
    noisy_X = add_gaussian_noise(X, noise_std=noise)
    nan_X = add_nan_patches_batch(noisy_X, n_patches=1, patch_length=missing_days)
    interp_X = interpolate_batch_fast(nan_X)
    return interp_X


def apply_corruption(X, noise=0.1, missing_days=90):
    """
    The main corruption pipeline used for training.

    Logic:
    1. Simulate missing data (NaN patches).
    2. Fill gaps (Interpolation) - mimicking how a pre-processor might guess missing values.
    3. Add Noise.

    Physics Note:
    It converts Log-space input (X) back to Linear-space (exp) before adding noise.
    This is because photon noise is additive in Flux (linear), not Magnitude (log).
    """

    # 1. Create gaps
    nan_X = add_nan_patches_batch(X, n_patches=1, patch_length=missing_days)

    # 2. Fill gaps (the network will see interpolated lines where data was missing)
    interp_X = interpolate_batch_fast(nan_X)

    # 3. Add noise in Linear Space (Physical Flux)
    # Input X is likely Log10(Luminosity).
    # exp(X) -> Linear Luminosity -> Add Gaussian Noise -> log(Result) -> Back to Log Space
    noisy_X = np.log(add_gaussian_noise(np.exp(interp_X), noise_std=noise))

    return noisy_X


def test_corruption():
    """
    Test script to visualize the corruption process.
    Generates plots showing 'Gold', 'Silver', and 'Bronze' quality data
    (varying levels of noise and missing data).
    """

    # Load synthetic supernova data
    lums = np.load("../../four parameter synthetic dataset/analyticModelEXPSOE_Run1_20230328_07-55-00.npy")

    # Scale data (StandardScaler centers mean to 0 and variance to 1)
    xscaler = StandardScaler()  # MinMaxScaler()

    # Take first 300 time steps and normalize
    # Note: The input 'lums' is converted to linear space (exp) before scaling?
    # Depending on previous files, 'lums' might already be log.
    batch = xscaler.fit_transform(np.exp(lums[:, :300]))

    # Generate 3 levels of data quality
    gold = apply_corruption(batch, noise=0.1, missing_days=30)  # Low noise, small gaps
    silver = apply_corruption(batch, noise=0.15, missing_days=60)  # Medium noise
    bronze = apply_corruption(batch, noise=0.3, missing_days=90)  # High noise, large gaps

    # Pick 30 random samples to visualize
    for s in np.random.random_integers(len(batch), size=(30)):
        # Setup plot
        fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        titles = ["Original", "Corrupted"]
        ok = fig.suptitle("Generated Corrupted samples")

        # Iterate over the three quality levels
        for corrupted, level, ax, in zip([gold, silver, bronze], ["gold", "silver", "bronze"], axs):
            # Plot Original Data (Inverse transform to get back to original scale)
            # Uses Log for visualization (Magnitude plot)
            ok = ax.plot(np.log(xscaler.inverse_transform(batch)[s]), "k")
            color = level

            if level == "bronze":
                color = "darkorange"  # Fix color name for visibility

            # Plot Corrupted Data
            ok = ax.plot(np.log(xscaler.inverse_transform(corrupted)[s]), '+', color=color)
            ok = ax.legend(titles)
            ok = ax.set_xlabel(level)

        ok = plt.ylabel('Magnitude')
        # Save each plot as a PNG file
        ok = plt.savefig("corrupted" + str(s).zfill(5) + ".png")


if __name__ == "__main__":
    test_corruption()
