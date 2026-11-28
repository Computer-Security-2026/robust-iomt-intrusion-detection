import numpy as np
import pytest

from src.isolation_forest import train_model, predictions


def make_test_data():
    """Create clearly normal and clearly unusual fake traffic."""
    random_generator = np.random.default_rng(42)

    benign_train = random_generator.normal(
        loc=0.0,
        scale=0.3,
        size=(200, 45)
    )

    normal_test = random_generator.normal(
        loc=0.0,
        scale=0.3,
        size=(20, 45)
    )

    obvious_anomalies = np.full(
        shape=(10, 45),
        fill_value=8.0
    )

    X_test = np.vstack([normal_test, obvious_anomalies])

    return benign_train, X_test


def test_outputs_have_correct_length_and_values():
    benign_train, X_test = make_test_data()

    model = train_model(benign_train)
    y_pred, score = predictions(model, X_test)

    assert len(y_pred) == len(X_test)
    assert len(score) == len(X_test)
    assert set(np.unique(y_pred)).issubset({0, 1})
    assert np.isfinite(score).all()


def test_obvious_anomalies_have_higher_scores():
    benign_train, X_test = make_test_data()

    model = train_model(benign_train)
    _, score = predictions(model, X_test)

    normal_scores = score[:20]
    anomaly_scores = score[20:]

    assert anomaly_scores.mean() > normal_scores.mean()


def test_empty_training_data_is_rejected():
    empty_data = np.empty((0, 45))

    with pytest.raises(ValueError):
        train_model(empty_data)