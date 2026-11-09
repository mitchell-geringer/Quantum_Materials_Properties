"""Load and prepare the course's featurized Materials Project data."""

import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


def load_data(path):
    """Read the trusted course pickle and return features, band gaps, and feature names."""
    with open(path, "rb") as handle:
        data = pickle.load(handle)

    feature_frame = data["X"]
    features = feature_frame.to_numpy(dtype=np.float32)
    band_gaps = data["y"]["band_gap"].to_numpy(dtype=np.float32)
    print(f"Loaded {features.shape[0]:,} samples with {features.shape[1]} features")
    return features, band_gaps, feature_frame


def clean_data(feature_frame, band_gaps):
    """Drop wholly missing columns, then exclude rows with missing features."""
    feature_frame = feature_frame.dropna(axis=1, how="all")
    complete_rows = ~feature_frame.isna().any(axis=1)
    return feature_frame.loc[complete_rows], band_gaps[complete_rows.to_numpy()]


def split_and_scale(features, targets, seed=0):
    """Split 60/20/20, then apply scaling fitted on the training split."""
    X_rest, X_test, y_rest, y_test = train_test_split(
        features, targets, test_size=0.20, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=0.25, random_state=seed
    )
    scaler = RobustScaler().fit(X_train)
    X_train, X_val, X_test = (
        scaler.transform(split).astype(np.float32) for split in (X_train, X_val, X_test)
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
