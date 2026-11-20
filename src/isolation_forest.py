# isolation_forest.py
# Author: Danah Alqahtani (CS 4371 Group 1)
# Trains an Isolation Forest using benign traffic features only.
# The model does not receive attack labels.

import os
import numpy as np

from sklearn.ensemble import IsolationForest
from src.data_loader import load_data


SEED = 42
N_ESTIMATORS = 100
CONTAMINATION = "auto"


def build_model():
    """Create the Isolation Forest with reproducible settings."""
    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=SEED,
        n_jobs=-1
    )

    return model


def train_model(X_train_benign):
    """Train the model using benign network traffic only."""
    if len(X_train_benign) == 0:
        raise ValueError("No benign training records were provided.")

    model = build_model()
    model.fit(X_train_benign)

    return model

def predictions(model, X_test):
    """
    Predict whether each test record is normal or suspicious.

    Returns:
        y_pred: 0 = normal, 1 = suspicious
        score: higher values mean more unusual traffic
    """
    original_predictions = model.predict(X_test)

    # Isolation Forest uses 1 for normal and -1 for anomaly.
    # Convert that to the team's format: 0 normal and 1 suspicious.
    y_pred = np.where(original_predictions == -1, 1, 0)

    # Reverse the score so larger values mean more suspicious.
    score = -model.decision_function(X_test)

    return y_pred, score


if __name__ == "__main__":
    # Load the small sample dataset.
    data = load_data(data_folder="data/sample")

    # Train using benign traffic features only.
    model = train_model(data["X_train_benign"])

    # Predict normal or suspicious for every test record.
    y_pred, score = predictions(model, data["X_test"])

    # Verify that the outputs are valid.
    assert len(y_pred) == len(data["X_test"])
    assert len(score) == len(data["X_test"])
    assert set(np.unique(y_pred)).issubset({0, 1})

    print("\n--- ISOLATION FOREST HANDOFF CHECKS ---")
    print("Test rows:", len(data["X_test"]))
    print("Prediction count:", len(y_pred))
    print("Suspicious records:", int(y_pred.sum()))
    print("Anomaly score range:", score.min(), "to", score.max())
    print("Unique predictions:", sorted(set(y_pred.tolist())))

    # Save outputs for evaluation and integration.
    src_dir = os.path.dirname(__file__)
    root_dir = os.path.dirname(src_dir)
    results_dir = os.path.join(root_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    np.save(
        os.path.join(results_dir, "isolation_forest_predictions.npy"),
        y_pred
    )

    np.save(
        os.path.join(results_dir, "isolation_forest_scores.npy"),
        score
    )

    print("Results saved in:", results_dir)