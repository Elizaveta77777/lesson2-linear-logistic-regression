import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parent

(ROOT / "plots").mkdir(exist_ok=True)
(ROOT / "models").mkdir(exist_ok=True)


# ==========================================
# ПОДГОТОВКА ДАННЫХ
# ==========================================

def get_data(seed=42):

    X, y = load_diabetes(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=seed
    )

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_x.fit_transform(X_train)
    X_test = scaler_x.transform(X_test)

    y_train = scaler_y.fit_transform(
        y_train.reshape(-1, 1)
    )

    y_test = scaler_y.transform(
        y_test.reshape(-1, 1)
    )

    return X_train, X_test, y_train, y_test, scaler_y


# ==========================================
# ОБУЧЕНИЕ ЛИНЕЙНОЙ МОДЕЛИ
# ==========================================

def fit_linear(
    X_train,
    y_train,
    X_test,
    y_test,
    lr=0.01,
    batch_size=32,
    optimizer_name="Adam",
    epochs=100
):

    torch.manual_seed(42)

    X_train = np.asarray(X_train, dtype="float32")
    X_test = np.asarray(X_test, dtype="float32")

    y_train = np.asarray(y_train, dtype="float32")
    y_test = np.asarray(y_test, dtype="float32")

    dataset = TensorDataset(
        torch.tensor(X_train),
        torch.tensor(y_train)
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    model = nn.Linear(X_train.shape[1], 1)

    optimizers = {
        "SGD": torch.optim.SGD,
        "Adam": torch.optim.Adam,
        "RMSprop": torch.optim.RMSprop
    }

    optimizer = optimizers[optimizer_name](
        model.parameters(),
        lr=lr
    )

    history = []

    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for batch_X, batch_y in dataloader:

            optimizer.zero_grad()

            predictions = model(batch_X)

            loss = nn.functional.mse_loss(
                predictions,
                batch_y
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_X)

        history.append(total_loss / len(dataset))

    model.eval()

    with torch.no_grad():

        predictions = model(
            torch.tensor(X_test)
        ).numpy()

    mse = mean_squared_error(y_test, predictions)

    return {
        "mse": mse,
        "history": history,
        "model": model,
        "predictions": predictions
    }


# ==========================================
# ЗАДАНИЕ 3.1 — ГИПЕРПАРАМЕТРЫ
# ==========================================

def hyperparameter_experiments():

    X_train, X_test, y_train, y_test, _ = get_data()

    settings = []

    # Различные learning rate
    for lr in [0.001, 0.01, 0.05]:
        settings.append((lr, 32, "Adam"))

    # Различные batch size
    for batch in [16, 32, 64]:
        settings.append((0.01, batch, "Adam"))

    # Различные оптимизаторы
    for optimizer in ["SGD", "Adam", "RMSprop"]:
        settings.append((0.01, 32, optimizer))

    settings = list(dict.fromkeys(settings))

    results = []
    curves = {}

    for lr, batch, optimizer_name in settings:

        result = fit_linear(
            X_train,
            y_train,
            X_test,
            y_test,
            lr=lr,
            batch_size=batch,
            optimizer_name=optimizer_name
        )

        label = (
            f"{optimizer_name}, "
            f"lr={lr}, batch={batch}"
        )

        results.append({
            "optimizer": optimizer_name,
            "learning_rate": lr,
            "batch_size": batch,
            "mse": float(result["mse"])
        })

        curves[label] = result["history"]

        print(label, "MSE:", result["mse"])

    # Сохранение таблицы результатов
    with open(
        ROOT / "plots" / "hyperparameters.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "optimizer",
                "learning_rate",
                "batch_size",
                "mse"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    # Графики обучения
    plt.figure(figsize=(10, 6))

    for label, history in curves.items():
        plt.plot(history, label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Hyperparameter Comparison")
    plt.legend(fontsize=7)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        ROOT / "plots" / "hyperparameter_curves.png"
    )

    plt.close()

    return results


# ==========================================
# ЗАДАНИЕ 3.2 — FEATURE ENGINEERING
# ==========================================

def engineer_features(X_train, X_test):

    polynomial = PolynomialFeatures(
        degree=2,
        include_bias=False
    )

    train_poly = polynomial.fit_transform(X_train)
    test_poly = polynomial.transform(X_test)

    # Среднее значение признаков
    train_mean = X_train.mean(axis=1)
    test_mean = X_test.mean(axis=1)

    # Дисперсия признаков
    train_var = X_train.var(axis=1)
    test_var = X_test.var(axis=1)

    train_features = np.column_stack([
        train_poly,
        train_mean,
        train_var
    ])

    test_features = np.column_stack([
        test_poly,
        test_mean,
        test_var
    ])

    return train_features, test_features


def feature_engineering_experiment():

    X_train, X_test, y_train, y_test, scaler_y = get_data()

    # Исходная модель
    baseline = fit_linear(
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=150
    )

    # Создание дополнительных признаков
    new_X_train, new_X_test = engineer_features(
        X_train,
        X_test
    )

    scaler = StandardScaler()

    new_X_train = scaler.fit_transform(new_X_train)
    new_X_test = scaler.transform(new_X_test)

    # Модель с новыми признаками
    engineered = fit_linear(
        new_X_train,
        y_train,
        new_X_test,
        y_test,
        epochs=150
    )

    results = {}

    for name, result in [
        ("baseline", baseline),
        ("engineered", engineered)
    ]:

        predictions = scaler_y.inverse_transform(
            result["predictions"]
        )

        true_values = scaler_y.inverse_transform(y_test)

        results[name] = {
            "mse": mean_squared_error(
                true_values,
                predictions
            ),
            "r2": r2_score(
                true_values,
                predictions
            )
        }

        torch.save(
            result["model"].state_dict(),
            ROOT / "models" / f"features_{name}.pth"
        )

    plt.figure(figsize=(7, 5))

    plt.bar(
        list(results.keys()),
        [value["mse"] for value in results.values()]
    )

    plt.ylabel("MSE")
    plt.title("Feature Engineering Comparison")
    plt.tight_layout()

    plt.savefig(
        ROOT / "plots" / "feature_comparison.png"
    )

    plt.close()

    return results


# ==========================================
# ЗАПУСК
# ==========================================

if __name__ == "__main__":

    print("\nЗадание 3.1 — Гиперпараметры")
    hyperparameter_experiments()

    print("\nЗадание 3.2 — Feature Engineering")

    results = feature_engineering_experiment()

    for name, metrics in results.items():
        print(name, metrics)

    print("\nЗадание 3 выполнено!")