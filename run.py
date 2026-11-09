"""Run the band-gap regression experiments and save metrics and figures."""

import argparse
import json
import platform
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import TensorDataset

from src.data import clean_data, load_data, split_and_scale
from src.train import train_network

ROOT = Path(__file__).resolve().parent
SPLIT_SEED = 0
CORRELATION_THRESHOLD = 0.85
FINAL_SETTINGS = dict(width=256, depth=4, dropout=0.1, lr=1e-3, batch_size=256)
ENSEMBLE_SEEDS = list(range(0, 110, 11))


def select_feature_sets(X_train, y_train):
    """Select feature subsets using only training data."""
    correlations = pd.DataFrame(X_train).corr().abs().to_numpy()
    redundant = np.triu(correlations > CORRELATION_THRESHOLD, k=1).any(axis=0)
    uncorrelated = np.flatnonzero(~redundant)
    forest = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)
    forest.fit(X_train[:, uncorrelated], y_train)
    ranked = uncorrelated[np.argsort(forest.feature_importances_)[::-1]]
    return {"Uncorrelated": uncorrelated, "Top 40": ranked[:40], "Top 20": ranked[:20]}


def metrics(targets, predictions):
    mse = mean_squared_error(targets, predictions)
    return {
        "mse_ev2": mse,
        "rmse_ev": float(np.sqrt(mse)),
        "mae_ev": mean_absolute_error(targets, predictions),
        "r2": r2_score(targets, predictions),
    }


@torch.no_grad()
def predict(model, features):
    model.eval()
    return model(features).numpy().ravel()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/features_1or2_stable.pkl")
    parser.add_argument("--figures", type=Path, default=ROOT / "figures")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--epochs", type=int, default=500, help="Maximum epochs per network (default: 500)")
    parser.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads (default: 1)")
    parser.add_argument("--skip-scans", action="store_true", help="Run the final benchmarks without feature or parameter scans")
    args = parser.parse_args()
    if args.epochs < 1 or args.threads < 1:
        parser.error("--epochs and --threads must be positive")
    if not args.data.is_file():
        parser.error(f"feature cache not found: {args.data}")
    return args


def main():
    args = parse_args()
    matplotlib.use("Agg")
    from src.plots import plot_band_gap_predictions, plot_uncertainty

    torch.set_num_threads(args.threads)
    _, targets, feature_frame = load_data(args.data)
    features, targets = clean_data(feature_frame, targets)
    X_train, X_val, X_test, y_train, y_val, y_test = split_and_scale(
        features.to_numpy(), targets, seed=SPLIT_SEED
    )
    print(f"Training {len(y_train):,} | validation {len(y_val):,} | test {len(y_test):,}")
    train_inputs = torch.from_numpy(X_train)
    val_inputs = torch.from_numpy(X_val)
    test_inputs = torch.from_numpy(X_test)
    train_targets = torch.from_numpy(y_train).reshape(-1, 1)
    val_targets = torch.from_numpy(y_val).reshape(-1, 1)
    all_columns = np.arange(X_train.shape[1])
    validation = []

    def fit(name, columns=all_columns, **settings):
        model, history = train_network(
            name, TensorDataset(train_inputs[:, columns], train_targets),
            val_inputs[:, columns], val_targets, features=len(columns),
            epochs=args.epochs, **settings,
        )
        validation.append({
            "model": name, "features": len(columns),
            **metrics(y_val, predict(model, val_inputs[:, columns])),
            "best_epoch": history["best_epoch"],
        })
        return model

    linear = LinearRegression().fit(X_train, y_train)
    validation.append({
        "model": "Linear regression", "features": len(all_columns),
        **metrics(y_val, linear.predict(X_val)),
    })

    if not args.skip_scans:
        fit("Full network")
        for name, columns in select_feature_sets(X_train, y_train).items():
            fit(f"{name} (256x4)", columns)
            fit(f"{name} (128x2)", columns, width=128, depth=2)
        scans = {
            "dropout": [0.1, 0.2, 0.5],
            "weight_decay": [3.0, 5.0, 10.0],
            "lr": [1e-4, 1e-2, 1e-1],
            "batch_size": [32, 128, 512],
        }
        for parameter, values in scans.items():
            for value in values:
                fit(f"{parameter}={value}", **{parameter: value})

    final_model = fit("Tuned network", **FINAL_SETTINGS)
    # The tuned network also serves as the first ensemble member (seed 0).
    ensemble = [final_model]
    for seed in ENSEMBLE_SEEDS[1:]:
        ensemble.append(fit(f"Ensemble seed {seed}", seed=seed, **FINAL_SETTINGS))

    val_ensemble = np.stack([predict(model, val_inputs) for model in ensemble])
    val_error = np.abs(y_val - val_ensemble.mean(axis=0))
    correlation = spearmanr(val_ensemble.std(axis=0), val_error)
    test_ensemble = np.stack([predict(model, test_inputs) for model in ensemble])
    predictions = {
        "Linear regression": linear.predict(X_test),
        "Tuned network": test_ensemble[0],
        "Ensemble": test_ensemble.mean(axis=0),
    }
    test_metrics = pd.DataFrame([
        {"model": name, **metrics(y_test, values)} for name, values in predictions.items()
    ])
    uncertainty = pd.DataFrame({
        "sigma": test_ensemble.std(axis=0),
        "abs_error": np.abs(y_test - predictions["Ensemble"]),
    })
    bins = pd.qcut(uncertainty["sigma"], q=10, duplicates="drop")
    binned = uncertainty.groupby(bins, observed=True).mean()

    args.figures.mkdir(parents=True, exist_ok=True)
    args.results.mkdir(parents=True, exist_ok=True)
    test_metrics.to_csv(args.results / "test_metrics.csv", index=False)
    pd.DataFrame(validation).to_csv(args.results / "validation_metrics.csv", index=False)
    binned.to_csv(args.results / "uncertainty_bins.csv", index=False)
    run_info = {
        "data": args.data.name,
        "rows": {"train": len(y_train), "validation": len(y_val), "test": len(y_test)},
        "features": len(all_columns),
        "split_seed": SPLIT_SEED,
        "correlation_threshold": CORRELATION_THRESHOLD,
        "final_model": FINAL_SETTINGS,
        "ensemble_seeds": ENSEMBLE_SEEDS,
        "max_epochs": args.epochs,
        "early_stopping_patience": 50,
        "torch_threads": args.threads,
        "scans": not args.skip_scans,
        "validation_uncertainty_spearman": (
            float(correlation.statistic) if np.isfinite(correlation.statistic) else None
        ),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
            "pytorch": torch.__version__,
        },
    }
    (args.results / "run_info.json").write_text(json.dumps(run_info, indent=2) + "\n")
    plot_band_gap_predictions(y_test, predictions, args.figures / "predicted_vs_actual.png")
    plot_uncertainty(binned, args.figures / "uncertainty_vs_error.png")
    print(test_metrics.round(4).to_string(index=False))
    print(f"Figures: {args.figures}\nResults: {args.results}")


if __name__ == "__main__":
    main()
