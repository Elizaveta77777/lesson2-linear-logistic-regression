from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score
)


ROOT = Path(__file__).resolve().parent

for folder in ["data", "models", "plots"]:
    (ROOT / folder).mkdir(exist_ok=True)


# ==========================================
# ЗАДАНИЕ 2.1 — СОБСТВЕННЫЙ DATASET
# ==========================================

class CSVDataset(Dataset):

    def __init__(self, csv_path, target_column,
                 task="regression"):

        if task not in ["regression", "classification"]:
            raise ValueError("Неизвестный тип задачи")

        df = pd.read_csv(csv_path)

        if target_column not in df.columns:
            raise ValueError("Целевой столбец не найден")

        df = df.dropna(subset=[target_column])

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Кодирование категориальных признаков
        X = pd.get_dummies(
            X,
            dummy_na=True,
            dtype=float
        )

        # Обработка пропусков
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median()).fillna(0)

        # Нормализация
        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype("float32")

        if task == "classification":

            encoder = LabelEncoder()
            y = encoder.fit_transform(
                y.astype(str)
            ).astype("int64")

        else:

            y = pd.to_numeric(y).to_numpy(
                dtype="float32"
            ).reshape(-1, 1)

        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


# ==========================================
# РАЗДЕЛЕНИЕ ДАННЫХ
# ==========================================

def prepare_csv_split(csv_path, target_column,
                      task, seed=42):

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[target_column])

    X = df.drop(columns=[target_column])
    y = df[target_column]

    stratify = None

    if task == "classification":
        if y.value_counts().min() >= 2:
            stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=seed,
        stratify=stratify
    )

    X_train = pd.get_dummies(
        X_train, dummy_na=True, dtype=float
    )

    X_test = pd.get_dummies(
        X_test, dummy_na=True, dtype=float
    )

    X_test = X_test.reindex(
        columns=X_train.columns,
        fill_value=0
    )

    X_train = X_train.replace(
        [np.inf, -np.inf], np.nan
    )

    X_test = X_test.replace(
        [np.inf, -np.inf], np.nan
    )

    medians = X_train.median()

    X_train = X_train.fillna(medians).fillna(0)
    X_test = X_test.fillna(medians).fillna(0)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    ).astype("float32")

    X_test = scaler.transform(
        X_test
    ).astype("float32")

    if task == "classification":

        encoder = LabelEncoder()

        y_train = encoder.fit_transform(
            y_train.astype(str)
        ).astype("int64")

        y_test = encoder.transform(
            y_test.astype(str)
        ).astype("int64")

    else:

        y_train = pd.to_numeric(
            y_train
        ).to_numpy(dtype="float32").reshape(-1, 1)

        y_test = pd.to_numeric(
            y_test
        ).to_numpy(dtype="float32").reshape(-1, 1)

    return X_train, X_test, y_train, y_test


# ==========================================
# ЗАДАНИЕ 2.2 — РАБОТА С ДАТАСЕТАМИ
# ==========================================

def create_csv_files():

    diabetes = load_diabetes(as_frame=True)

    regression_df = diabetes.frame.copy()

    regression_path = ROOT / "data" / "diabetes.csv"

    regression_df.to_csv(
        regression_path,
        index=False
    )

    cancer = load_breast_cancer(as_frame=True)

    classification_df = cancer.frame.copy()

    classification_path = (
        ROOT / "data" / "breast_cancer.csv"
    )

    classification_df.to_csv(
        classification_path,
        index=False
    )

    return regression_path, classification_path


# ==========================================
# ОБУЧЕНИЕ МОДЕЛИ
# ==========================================

def train_csv_model(csv_path, target, task,
                    epochs=120):

    torch.manual_seed(42)

    X_train, X_test, y_train, y_test = prepare_csv_split(
        csv_path,
        target,
        task
    )

    classification = task == "classification"

    if classification:
        output_dim = len(np.unique(y_train))
    else:
        output_dim = 1

    model = nn.Linear(
        X_train.shape[1],
        output_dim
    )

    if classification:
        criterion = nn.CrossEntropyLoss()
        learning_rate = 0.01
        y_train_fit = y_train

    else:
        criterion = nn.MSELoss()
        learning_rate = 0.02

        mean = float(y_train.mean())
        std = max(float(y_train.std()), 1e-8)

        y_train_fit = (y_train - mean) / std

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    dataset = TensorDataset(
        torch.tensor(X_train),
        torch.tensor(y_train_fit)
    )

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )

    history = []

    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for batch_X, batch_y in dataloader:

            optimizer.zero_grad()

            predictions = model(batch_X)

            loss = criterion(predictions, batch_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_X)

        history.append(total_loss / len(dataset))

    model.eval()

    with torch.no_grad():

        predictions = model(
            torch.tensor(X_test)
        ).numpy()

    if classification:

        predicted_classes = predictions.argmax(axis=1)

        results = {
            "accuracy": accuracy_score(
                y_test, predicted_classes
            ),
            "f1_macro": f1_score(
                y_test,
                predicted_classes,
                average="macro"
            )
        }

    else:

        predictions = predictions * std + mean

        results = {
            "mse": mean_squared_error(
                y_test, predictions
            ),
            "r2": r2_score(
                y_test, predictions
            )
        }

    name = Path(csv_path).stem

    torch.save(
        model.state_dict(),
        ROOT / "models" / f"csv_{name}.pth"
    )

    plt.figure(figsize=(8, 5))
    plt.plot(history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training: {name}")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        ROOT / "plots" / f"csv_{name}_loss.png"
    )

    plt.close()

    return results


# ==========================================
# ЗАПУСК
# ==========================================

if __name__ == "__main__":

    regression_path, classification_path = create_csv_files()

    print("\nРегрессия:")
    print(
        train_csv_model(
            regression_path,
            "target",
            "regression"
        )
    )

    print("\nКлассификация:")
    print(
        train_csv_model(
            classification_path,
            "target",
            "classification"
        )
    )

    dataset = CSVDataset(
        classification_path,
        "target",
        "classification"
    )

    print("\nРазмер Dataset:", len(dataset))
    print("Размерность признаков:", dataset[0][0].shape)

    print("\nЗадание 2 выполнено!")