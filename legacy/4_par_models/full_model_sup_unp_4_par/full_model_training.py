import datetime
import numpy as np
import pandas as pd
import torch

# Create a timestamp string for saving unique model filenames later
time_st = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from mlpres import MLPWithResiduals, Conditional_VAE

sym_lum_path = r"C:\Users\39320\Desktop\ASTRAI\four parameter synthetic dataset\analyticModelEXPSOE_Run1_20230328_07-55-00.npy"
attributes_path = r"C:\Users\39320\Desktop\ASTRAI\four parameter synthetic dataset\lista_amEXPSOE.csv"
real_lums_path = r"/real dataset\74_SNE_bolometric.npy"

# --- 1. Data Loading ---
# Load synthetic (simulated) light curves from a .npy file
sym_lums = np.load(sym_lum_path)

# Load the corresponding physical attributes (parameters) for the synthetic curves
attributes = pd.read_csv(attributes_path,sep=";")

# Load real observed bolometric light curves and apply Log10 transformation immediately
# (Log scaling is often used to handle the large dynamic range of luminosity)
real_lums = np.log10(np.load(real_lums_path))

# --- 2. Data Preprocessing & Scaling ---
xscaler = StandardScaler()

# Fit scaler on Real data and transform it.
# Note: We ravel (flatten) the 2D array to 1D to compute global mean/std,
# then reshape back to the original (Batch, TimeSteps) shape.
X_real = xscaler.fit_transform(np.ravel(real_lums).reshape(-1,1)).reshape(real_lums.shape)

# Use the SAME scaler (fitted on real data) to transform the Synthetic data.
# This ensures both datasets are in the same feature space.
X_sym = xscaler.transform(np.ravel(sym_lums).reshape(-1,1)).reshape(sym_lums.shape)

# Store the scale factor (std deviation) for potential inverse transforms later
scale = xscaler.scale_

# Scale the physical parameters (Target variables for the synthetic data)
yscaler = StandardScaler()

# Skip the first column (usually ID/Index) and scale the rest
Y_sym = yscaler.fit_transform(np.array(attributes[attributes.columns[1:]]))

# --- 3. Train/Test Split ---
# Split synthetic data (Input Curves + Target Parameters)
X_train_sym, X_test_sym, y_train_sym, y_test_sym = train_test_split(X_sym, Y_sym, test_size=0.5)

# Split real data (Input Curves only, no labels available)
X_train_real, X_test_real = train_test_split(X_real, test_size=0.5)

# --- 4. Model Configuration ---
N_DAYS = 421      # Input dimension (length of the light curve)
N_PARAMS = 7      # Latent dimension (number of physical parameters)
width = 32        # Hidden layer size
depth = 2         # Number of residual blocks
dropout = 0       # Dropout rate
n_epochs = 10000  # Total training epochs

# --- 5. Model Instantiation ---

# -- Model A: Unsupervised (Autoencoder approach) --
# Encoder: Compresses Light Curve (421) -> Latent Space (7)
regr = MLPWithResiduals(N_DAYS,width,N_PARAMS,depth,dropout)
# Decoder: Reconstructs Latent Space (7) -> Light Curve (421)
lcgen = MLPWithResiduals(N_PARAMS,width,N_DAYS,depth,dropout)
# Combine into a VAE (acting as an Autoencoder here)
full_model_unsupervised = Conditional_VAE(regr,lcgen)

# -- Model B: Supervised (Physics-Informed / Transfer Learning) --
# Same architecture, distinct instance
regr_ = MLPWithResiduals(N_DAYS,width,N_PARAMS,depth,dropout)
lcgen_ = MLPWithResiduals(N_PARAMS,width,N_DAYS,depth,dropout)
full_model_supervised = Conditional_VAE(regr_,lcgen_)

# --- 6. Training ---

# 1. Train the Unsupervised model on REAL data only.
# Since no 'y' is provided, the VAE class likely treats this as an Autoencoder task
# (Input -> Encoder -> Decoder -> Output ≈ Input).
full_model_unsupervised.fit(X_train_real, epochs=n_epochs) #allena modello solo con le curve reali (senza attributi fisici)

# 2. Train the Supervised model on SYNTHETIC data first.
# This acts as "Pre-training". The model learns the mapping from Curve -> Physics Parameters
# because 'y_train_sym' is provided.
full_model_supervised.fit(X_train_sym, epochs=n_epochs//10) #modello usa le curve simulate e i loro attributi

# 3. Fine-tune the Supervised model on REAL data.
# We take the pre-trained physics model and adapt it to the real domain.
# Note: 'y' is NOT provided here. The model switches to Autoencoder mode to adapt
# to the real data distribution while retaining physical knowledge from step 2.
full_model_supervised.fit(X_train_real, epochs=n_epochs) #modello di prima, ma usa quelle reali

# --- 7. Save Models ---
torch.save(full_model_supervised.state_dict(), "full_model_supervised_"+time_st+".pth")
torch.save(full_model_unsupervised.state_dict(), "full_model_unsupervised_"+time_st+".pth")

