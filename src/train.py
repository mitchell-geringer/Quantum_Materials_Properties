"""Train band-gap regressors with early stopping on validation MSE."""

import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, r2_score

from .model import make_network

# Regression Loss Function
loss_fn = nn.MSELoss()

def train_network(name, train_ds, X_val_t, y_val_t, features=141, width=256, depth=4,
                  epochs=500, patience=50, dropout=0.0, lr=1e-3, weight_decay=0.0, batch_size=256, seed=0):
    """Build and train a network. Returns (model, history); the model keeps its best weights."""
    torch.manual_seed(seed)
    model = make_network(features, width, depth, dropout)
    torch.manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    hist = {"train_loss": [], "val_loss": [], "val_r2": [], "val_mae": []}
    best_loss, best_epoch, best_state = np.inf, 0, None
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()                       # dropout on
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)

        model.eval()                        # dropout off

        # 2. Generate predictions for both splits (no gradients needed)
        with torch.no_grad():
            out = model(X_val_t)

        # Calculate R² scores by converting true target tensors back to numpy
        hist["train_loss"].append(total / len(train_ds))
        hist["val_loss"].append(loss_fn(out, y_val_t).item())
        hist["val_r2"].append(r2_score(y_val_t.numpy(), out.numpy()))
        hist["val_mae"].append(mean_absolute_error(y_val_t.numpy(), out.numpy()))

        if hist["val_loss"][-1] < best_loss:
            best_loss, best_epoch = hist["val_loss"][-1], epoch
            best_state = copy.deepcopy(model.state_dict())
        if patience is not None and epoch - best_epoch >= patience:
            break                           # early stopping

    model.load_state_dict(best_state)
    hist.update(best_epoch=best_epoch,
                best_loss=best_loss,
                best_r2=hist["val_r2"][best_epoch - 1],
                best_mae=hist["val_mae"][best_epoch - 1],
                epochs_run=epoch, seconds=time.time() - t0)
    print(f"{name:<28} best val loss {best_loss:.4f} | val MAE {hist['best_mae']:.4f} eV | "
          f"val R2 {hist['best_r2']:.4f} at epoch {best_epoch:3d} | {epoch} epochs, {hist['seconds']:.0f} s")
    return model, hist


def record_metrics(final_results, model_name, hist):
    final_results[model_name] = {
        "Optimal_Epoch": hist["best_epoch"],
        "best_val_loss": hist["best_loss"],
        "best_val_MAE": hist.get("best_mae", np.nan),  # Safe fallback if best_mae isn't present
        "best_val_R2": hist["best_r2"]
    }
    return final_results
