import copy
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_diabetes, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


ROOT = Path(__file__).resolve().parent

(ROOT / "models").mkdir(exist_ok=True)
(ROOT / "plots").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "training.log", encoding="utf-8")
    ],
    force=True
)

logger = logging.getLogger(__name__)


# ==========================================
# ЗАДАНИЕ 1.1 — ЛИНЕЙНАЯ РЕГРЕССИЯ
# ==========================================

class LinearRegression(nn.Module):

    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)


# L1 и L2 регуляризация
def regularized_mse(model, pred, y, l1=0.0, l2=0.0):

    loss = nn.functional.mse_loss(pred, y)

    for name, param in model.named_parameters():

        if name.endswith("weight"):
            loss = (
                loss
                + l1 * param.abs().sum()
                + l2 * param.square().sum()
            )

    return loss


# Ранняя остановка обучения
class EarlyStopping:

    def __init__(self, patience=15, min_delta=0.00001):

        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.wait = 0
        self.best_state = None

    def step(self, val_loss, model):

        if val_loss < self.best_loss - self.min_delta:

            self.best_loss = val_loss
            self.wait = 0
            self.best_state = copy.deepcopy(model.state_dict())

        else:
            self.wait += 1

        return self.wait >= self.patience

    def restore(self, model):

        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def regression_experiment(seed=42):

    torch.manual_seed(seed)

    X, y = load_diabetes(return_X_y=True)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.25,
        random_state=seed
    )

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_x.fit_transform(X_train).astype("float32")
    X_val = scaler_x.transform(X_val).astype("float32")

    y_train = scaler_y.fit_transform(
        y_train.reshape(-1, 1)
    ).astype("float32")

    y_val = scaler_y.transform(
        y_val.reshape(-1, 1)
    ).astype("float32")

    dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train)
    )

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )

    model = LinearRegression(X_train.shape[1])

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01
    )

    early_stopping = EarlyStopping(patience=20)

    train_losses = []
    val_losses = []

    for epoch in range(1, 301):

        model.train()
        total_loss = 0.0

        for batch_X, batch_y in dataloader:

            optimizer.zero_grad()

            predictions = model(batch_X)

            loss = regularized_mse(
                model,
                predictions,
                batch_y,
                l1=0.0001,
                l2=0.001
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_X)

        model.eval()

        with torch.no_grad():

            val_predictions = model(torch.from_numpy(X_val))

            val_loss = nn.functional.mse_loss(
                val_predictions,
                torch.from_numpy(y_val)
            ).item()

        train_losses.append(total_loss / len(dataset))
        val_losses.append(val_loss)

        if epoch % 20 == 0:

            logger.info(
                "Epoch %d | Train Loss: %.4f | Val Loss: %.4f",
                epoch,
                train_losses[-1],
                val_loss
            )

        if early_stopping.step(val_loss, model):

            print(f"Early Stopping на эпохе {epoch}")
            break

    early_stopping.restore(model)

    torch.save(
        model.state_dict(),
        ROOT / "models" / "linear_regression.pth"
    )

    plt.figure(figsize=(8, 5))

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Linear Regression Training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(ROOT / "plots" / "regression_loss.png")
    plt.close()

    return {
        "epochs": len(train_losses),
        "best_val_mse": early_stopping.best_loss
    }


# ==========================================
# ЗАДАНИЕ 1.2 — ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ
# ==========================================

class Classifier(nn.Module):

    def __init__(self, in_features, num_classes):

        super().__init__()

        self.linear = nn.Linear(
            in_features,
            num_classes
        )

    def forward(self, x):
        return self.linear(x)


def classification_experiment(seed=42):

    torch.manual_seed(seed)

    X, y = load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=seed,
        stratify=y
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train).astype("float32")
    X_test = scaler.transform(X_test).astype("float32")

    dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train.astype("int64"))
    )

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True
    )

    model = Classifier(
        X_train.shape[1],
        len(np.unique(y))
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.05
    )

    for epoch in range(150):

        model.train()

        for batch_X, batch_y in dataloader:

            optimizer.zero_grad()

            predictions = model(batch_X)

            loss = criterion(predictions, batch_y)

            loss.backward()
            optimizer.step()

    model.eval()

    with torch.no_grad():

        logits = model(torch.from_numpy(X_test))

        probabilities = torch.softmax(
            logits,
            dim=1
        ).numpy()

    predictions = probabilities.argmax(axis=1)

    metrics = {
        "precision": precision_score(
            y_test, predictions,
            average="macro",
            zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions,
            average="macro",
            zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions,
            average="macro",
            zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
            multi_class="ovr",
            average="macro"
        )
    }

    print("\nМетрики классификации:")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    cm = confusion_matrix(y_test, predictions)

    ConfusionMatrixDisplay(
        cm,
        display_labels=load_iris().target_names
    ).plot(cmap="Blues")

    plt.title("Confusion Matrix")
    plt.tight_layout()

    plt.savefig(ROOT / "plots" / "confusion_matrix.png")
    plt.close()

    torch.save(
        model.state_dict(),
        ROOT / "models" / "multiclass_logistic.pth"
    )

    return metrics


# ==========================================
# ЗАПУСК ПРОГРАММЫ
# ==========================================

if __name__ == "__main__":

    print("Задание 1.1 — Линейная регрессия")
    print(regression_experiment())

    print("\nЗадание 1.2 — Логистическая регрессия")
    print(classification_experiment())

    print("\nЗадание 1 выполнено!")