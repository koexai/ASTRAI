"""
models.unified_model - Bi-directional characterization / generation wrapper.

The ``UnifiedModel`` couples two sub-networks trained jointly with a
weighted composite loss:

    L = alpha_char * L_char + alpha_gen * L_gen

where ``L_char`` supervises the *characterization* branch (curves -> params)
and ``L_gen`` supervises the *generation* branch (params -> curves) using
teacher-forced ground-truth parameters.
"""
from torch import nn


class UnifiedModel(nn.Module):
    """Wrapper holding both models
    - Regressor (curves -> params)
    - Generator (params -> curves).

    Parameters
    ----------
    regressor : nn.Module
        Characterization network (e.g. ``SplitMLPRegressor``).
    generator : nn.Module
        Generative decoder (e.g. ``MLPWithResiduals``).
    """

    def __init__(self, regressor, generator):
        super().__init__()
        self.regressor = regressor
        self.generator = generator

    def fit(
        self,
        train_loader,
        optimizer,
        criterion_char,
        criterion_gen,
        device,
        epochs=10,
        alpha_char=1.0,
        alpha_gen=1.0,
        scheduler=None,
    ):
        """Train both branches end-to-end for a given number of epochs.

        Parameters
        ----------
        train_loader : DataLoader
            Yields ``(batch_x, batch_y)`` pairs of (PCA curves, scaled params).
        optimizer : torch.optim.Optimizer
            Shared optimizer for both branches.
        criterion_char : callable
            Loss function for the characterization branch.
        criterion_gen : callable
            Loss function for the generation branch.
        device : torch.device
            Target compute device (CPU / CUDA).
        epochs : int
            Number of full passes over the training set.
        alpha_char : float
            Weight multiplier for the characterization loss.
        alpha_gen : float
            Weight multiplier for the generation loss.
        scheduler : lr_scheduler, optional
            Learning-rate scheduler stepped once per epoch.
        """
        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()

                # Characterization: curves -> params
                pred_params = self.regressor(batch_x)
                loss_char = criterion_char(pred_params, batch_y)

                # Generation: params -> curves (teacher forcing)
                pred_curve = self.generator(batch_y)
                loss_gen = criterion_gen(pred_curve, batch_x)

                loss = alpha_char * loss_char + alpha_gen * loss_gen

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if scheduler is not None:
                scheduler.step()

            if (epoch + 1) % 10 == 0:
                lr_info = loss_info = ""
                if scheduler is not None:
                    lr_info = f" | LR: {optimizer.param_groups[0]['lr']:.2e}"
                    loss_info = f"Avg Loss: {total_loss/len(train_loader):.6f}"
                print(f"Epoch {epoch+1}/{epochs} - {loss_info}{lr_info}")
