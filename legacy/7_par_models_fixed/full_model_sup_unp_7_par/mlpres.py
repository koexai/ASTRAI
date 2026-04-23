import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score

class ResidualBlock(nn.Module):
    def __init__(self, in_features, dropout):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, in_features)
        self.leaky = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(in_features)
        self.dropout_val = dropout
        if self.dropout_val > 0:
            self.drop = nn.Dropout(p=self.dropout_val)

    def forward(self, x):
        residual = x
        x = self.bn1(x)
        x = self.leaky(x)
        if self.dropout_val > 0:
            x = self.drop(x)
        x = self.fc1(x)
        x += residual
        return x

class MLPWithResiduals(nn.Module):
    """
    Rete standard a singolo blocco (usata per il Generatore).
    """
    def __init__(self, input_dim, width, out_dim, depth, dropout):
        super(MLPWithResiduals, self).__init__()
        layers = [nn.Linear(input_dim, width)]
        for _ in range(depth):
            layers.append(ResidualBlock(width, dropout))
        layers.append(nn.Linear(width, out_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class SplitMLPRegressor(nn.Module):
    """
    NUOVA ARCHITETTURA: Split MLP.
    Crea N reti indipendenti, una per ogni parametro di output.
    Migliora la precisione della caratterizzazione.
    """
    def __init__(self, input_dim, width, num_params, depth, dropout):
        super(SplitMLPRegressor, self).__init__()
        self.nets = nn.ModuleList([
            MLPWithResiduals(input_dim, width, 1, depth, dropout)
            for _ in range(num_params)
        ])

    def forward(self, x):
        # Esegue ogni rete indipendente e concatena i risultati
        outputs = [net(x) for net in self.nets]
        return torch.cat(outputs, dim=1)

class UnifiedModel(nn.Module):
    """
    Wrapper che contiene sia il Regressor (Split MLP) che il Generator.
    """
    def __init__(self, regressor, generator):
        super(UnifiedModel, self).__init__()
        self.regressor = regressor # Curve -> Params (Split MLP)
        self.generator = generator # Params -> Curve (Single MLP)

    def fit(self, train_loader, optimizer, criterion_char, criterion_gen, device,
            epochs=10, alpha_char=1.0, alpha_gen=1.0, scheduler=None):
        """
        Training loop with configurable loss balancing.

        Args:
            alpha_char: weight for characterization loss (default 1.0)
            alpha_gen: weight for generation loss (default 1.0)
            scheduler: optional LR scheduler, stepped each epoch
        """
        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in train_loader:
                # batch_x: Curve (Input Char, Target Gen)
                # batch_y: Parametri (Target Char, Input Gen)
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()

                # --- TASK 1: Caratterizzazione (Curve -> Parametri) ---
                pred_params = self.regressor(batch_x)
                loss_char = criterion_char(pred_params, batch_y)

                # --- TASK 2: Generazione (Parametri -> Curve) ---
                # Usiamo i parametri veri per addestrare il generatore (Teacher Forcing)
                pred_curve = self.generator(batch_y)
                loss_gen = criterion_gen(pred_curve, batch_x)

                # FIX 7: Weighted loss with configurable alpha coefficients
                loss = alpha_char * loss_char + alpha_gen * loss_gen

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Step the LR scheduler if provided
            if scheduler is not None:
                scheduler.step()

            # Stampa ogni 10 epoche
            if (epoch + 1) % 10 == 0:
                lr_info = ""
                if scheduler is not None:
                    lr_info = f" | LR: {optimizer.param_groups[0]['lr']:.2e}"
                print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {total_loss/len(train_loader):.6f}{lr_info}")
