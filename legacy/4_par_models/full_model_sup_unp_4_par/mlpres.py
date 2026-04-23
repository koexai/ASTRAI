import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
from data_corruption import apply_corruption, interpolate_batch ,interpolate_batch_fast


class ResidualBlock(nn.Module):
    """
    A standard Residual Block allows the network to learn identity functions easily.
    Structure: Input -> BN -> LeakyReLU -> Dropout -> Linear -> Add Input
    """
    def __init__(self, in_features, dropout):
        super(ResidualBlock, self).__init__()
        # LoRa optimization
        #self.fc1a = nn.Linear(in_features, in_features//8) 
        #self.fc1b = nn.Linear(in_features//8, in_features)
        self.fc1 = nn.Linear(in_features, in_features) 
        self.leaky = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(in_features)
        self.dropout = dropout
        if self.dropout:
            self.drop = nn.Dropout()
    
    def forward(self, x):
        residual = x  # Save the input for the residual connection
        x = self.bn1(x)
        x = self.leaky(x)
        if self.dropout:
            x = self.drop(x)
        # LoRa optimization
        #x = self.fc1a(x)
        #x = self.fc1b(x)
        x = self.fc1(x)
        x += residual  # Add the residual connection
        return x


class MLPWithResiduals(nn.Module):
    """
        Multi-Layer Perceptron built with a stack of Residual Blocks.
        Includes logic for handling missing data (interpolation) and regularization.
        """
    def __init__(self, input_dim, hidden_dim, output_dim, depth, dropout):
        super(MLPWithResiduals, self).__init__()
        # Initial projection layer
        layers = [nn.Linear(input_dim, hidden_dim)]

        # Stack 'depth' number of residual blocks
        for _ in range(depth):
            layers.append(ResidualBlock(hidden_dim, dropout))

        # Final projection to output dimension
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.res_blocks = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass.
        NOTE: This performs interpolation on the CPU using numpy, then moves back to GPU.
        This fixes missing values in the batch before passing it to the network.
        """

        # Detach from graph, move to CPU, convert to numpy to apply custom interpolation
        x = interpolate_batch(x.detach().cpu().numpy())

        # Determine device based on input type (handle case if x was already a tensor)
        device = x.device if isinstance(x, torch.Tensor) else torch.device('cpu')

        # Convert back to tensor and pass through the network
        return self.res_blocks(torch.tensor(x, dtype=torch.float32).to(device))

    def fit(self, X, y, epochs=1000, lr=1e-3, verbose=False, device='cuda', corruption=False, deriv=False):
        """
        Custom training loop handling NaN masking and derivative smoothing.
        """

        self.to(device)
        self.train()

        # If not using corruption, prepare the static input tensor
        if not corruption:
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

        # Ensure target has correct shape (batch_size, 1) if 1D array provided
        if len(y_tensor.shape) == 1:
            y_tensor = y_tensor.unsqueeze(1)

        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            # If corruption is enabled, apply noise/masks dynamically every epoch (Denoising)
            if corruption:
                X_corr = apply_corruption(X)
                X_tensor = torch.tensor(X_corr, dtype=torch.float32).to(device)

            optimizer.zero_grad()
            outputs = self.forward(X_tensor)

            # --- Masking Logic for NaNs ---
            # We create masks to ignore NaN values in both inputs and targets during loss calculation
            y_nan_mask = torch.isnan(y_tensor)
            y_nan_mask = torch.isnan(y_tensor)
            y_valid_mask = torch.logical_not(y_nan_mask)
            X_nan_mask = torch.isnan(X_tensor)
            X_valid_mask = torch.logical_not(X_nan_mask)

            # Calculates MSE only where data is valid.
            # Note: This line compares Input X vs Output y.
            # Depending on the use case, this might be intended to match dimensions,
            # but usually, you compare outputs against y_tensor.
            loss = criterion(X_tensor[X_valid_mask],outputs[y_valid_mask])

            # Standard regression loss (commented out)
            #loss = criterion(outputs, y_tensor)

            # --- Derivative / Smoothness Regularization ---
            # If the output is a time-series or curve (len > 100), enforce smoothness
            if deriv and len(outputs[0]>100):
                # Penalize the difference in derivatives (smoothing the curve)
                loss += criterion(outputs[:,:-2]-outputs[:,2:], y_tensor[:,:-2]-y_tensor[:,2:])

                # Second derivative penalty (Laplacian) - commented out
                #loss += criterion(outputs[:,:-2]+outputs[:,2:]-2*outputs[:,1:-1], y_tensor[:,:-2]+y_tensor[:,2:]-2*y_tensor[:,1:-1])

            if len(outputs[0]>100):
                # Self-consistency loss: Forces the derivative of the output to be continuous
                loss += criterion(outputs[:,:-3]-outputs[:,2:-1],outputs[:,1:-2]-outputs[:,3:])
            loss.backward()
            optimizer.step()

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    def predict(self, X, device='cuda', corruption=False):
        """Performs inference, optionally applying corruption first."""
        if corruption:
            X_corr = apply_corruption(X)
            X_tensor = torch.tensor(X_corr, dtype=torch.float32).to(device)
        if not corruption:
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

        self.to(device)
        self.eval() # Set to evaluation mode (disable dropout/batchnorm update)

        with torch.no_grad():
            outputs = self.forward(X_tensor)
        return outputs.cpu().numpy()

    def score(self, X, y, device='cuda', corruption=False):
        """Calculates R2 score of the model."""
        if corruption:
            X_corr = apply_corruption(X)
            y_pred = self.predict(X_corr, device)
        if not corruption:
            y_pred = self.predict(X, device)

        return r2_score(y, y_pred)


class Conditional_VAE(nn.Module):
    """
    A class that connects an Encoder and a Decoder.
    Despite the name 'VAE', this implementation behaves more like a
    standard Autoencoder or Encoder-Decoder architecture unless the
    sub-modules implement variational sampling logic internally.
    """
    def __init__(self, encoder, decoder):
        super(Conditional_VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        # Connects encoder output directly to decoder input
        self.pipe = nn.Sequential(self.encoder,self.decoder)

    def forward(self, x):
        return self.pipe(x)

    def fit(self, X, y=None, epochs=1000, lr=1e-3, verbose=True, device='cpu', corruption=False, deriv=False):
        self.to(device)
        self.train()

        # Handle case where y is not provided (Unsupervised / Pure Autoencoding)
        if y is not None:
            y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
            if len(y_tensor.shape) == 1:
                y_tensor = y_tensor.unsqueeze(1)
        else:
            y_tensor = None

        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.MSELoss()

        # Check available hardware
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for epoch in range(epochs):
            # --- Separate Training Step ---
            # Attempts to train encoder and decoder separately for 1 epoch.

        # -- TO DO ---
            # WARNING: There appears to be a bug here.
            # 'self' is passed as the first argument to fit(), but 'self' is the VAE,
            # whereas encoder.fit expects X (numpy/tensor).
            # This block might crash or behave unexpectedly.

            if not y is None:
                self.encoder.fit(self, X=X, y=y, epochs=1, lr=lr, device=device, corruption=corruption)
                self.decoder.fit(self, X=y, y=X, epochs=1, lr=lr, device=device, corruption=False)

            # --- Joint Training Step ---
            optimizer.zero_grad()

            # Denoising Autoencoder logic: corrupt input, try to reconstruct original
            if corruption:
                X_corr = apply_corruption(X)
                X_corr = torch.tensor(X_corr, dtype=torch.float32).to(device)
                outputs = self.forward(X_corr)
            else:
                outputs = self.forward(X_tensor)

            # Mask NaNs for loss calculation
            X_nan_mask = torch.isnan(X_tensor)
            X_valid_mask = torch.logical_not(X_nan_mask)
            loss = criterion(X_tensor[X_valid_mask],outputs[X_valid_mask])
            #loss = criterion(outputs, X_tensor)

            # Apply derivative loss (smoothness) if using targets
            if deriv and y_tensor is not None and outputs.shape[1] > 100:
                loss += criterion(outputs[:, :-2] - outputs[:, 2:], y_tensor[:, :-2] - y_tensor[:, 2:])
                # loss += criterion(outputs[:, :-2] + outputs[:, 2:] - 2 * outputs[:, 1:-1], y_tensor[:, :-2] + y_tensor[:, 2:] - 2 * y_tensor[:, 1:-1])

            # Apply internal derivative smoothing (Self-Consistency)
            if outputs.shape[1] > 100:
                deriv_loss = 100 * criterion(outputs[:, :-3] - outputs[:, 2:-1], outputs[:, 1:-2] - outputs[:, 3:])
                loss += deriv_loss
            else:
                deriv_loss = torch.tensor(0.0).to(device)  # fallback per la stampa se < 100

            loss.backward()
            optimizer.step()

            if verbose:
                # Calculates clean loss for display by removing the derivative penalty
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()-deriv_loss.item():.4f}, Deriv Loss: {deriv_loss.item():.4f}")

    def predict(self, X, device='cuda', corruption=False):
        if corruption:
            X_corr = apply_corruption(X)
            X_tensor = torch.tensor(X_corr, dtype=torch.float32).to(device)
        if not corruption:
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

        self.to(device)
        self.eval()

        with torch.no_grad():
            outputs = self.forward(X_tensor)
        return outputs.cpu().numpy()

    def score(self, X, device='cuda', corruption=False):
        """Scores reconstruction accuracy (R2 between Input X and Reconstructed X)."""
        if corruption:
            X_corr = apply_corruption(X)
            X_pred = self.predict(X_corr, device)
        if not corruption:
            X_pred = self.predict(X, device)

        return r2_score(X, X_pred)
