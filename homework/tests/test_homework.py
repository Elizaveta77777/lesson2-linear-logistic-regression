import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from homework_model_modification import (
    LinearRegression,
    Classifier,
    EarlyStopping,
    regularized_mse
)

from homework_datasets import (
    CSVDataset,
    prepare_csv_split
)

from homework_experiments import engineer_features


def test_linear_regression():
    model = LinearRegression(3)
    result = model(torch.randn(4, 3))
    assert result.shape == (4, 1)


def test_classifier():
    model = Classifier(3, 4)
    result = model(torch.randn(5, 3))
    assert result.shape == (5, 4)


def test_regularization():
    model = LinearRegression(2)
    x = torch.randn(6, 2)
    y = torch.randn(6, 1)

    predictions = model(x)

    plain = regularized_mse(model, predictions, y)

    regularized = regularized_mse(
        model, predictions, y,
        l1=0.1, l2=0.1
    )

    assert regularized.item() >= plain.item()


def test_early_stopping():
    model = LinearRegression(1)
    stopping = EarlyStopping(patience=2)

    assert stopping.step(1.0, model) is False
    assert stopping.step(2.0, model) is False
    assert stopping.step(3.0, model) is True


def test_csv_dataset(tmp_path):
    path = tmp_path / "test.csv"

    df = pd.DataFrame({
        "feature": [1, 2, 3, 4, 5, 6, 7, 8],
        "category": ["A", "B"] * 4,
        "target": [0, 1] * 4
    })

    df.to_csv(path, index=False)

    dataset = CSVDataset(
        path, "target", "classification"
    )

    assert len(dataset) == 8

    X_train, X_test, y_train, y_test = prepare_csv_split(
        path, "target", "classification"
    )

    assert len(X_train) + len(X_test) == 8


def test_feature_engineering():
    X_train = np.ones((4, 3))
    X_test = np.ones((2, 3))

    train, test = engineer_features(X_train, X_test)

    assert train.shape[1] > 3
    assert train.shape[1] == test.shape[1]