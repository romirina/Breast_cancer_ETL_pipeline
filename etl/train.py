# Train script: Logistic Regression
import os
import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

def train_model(data_dir, model_path):
    # Загружаем подготовленные данные
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv"))

    # Обучаем модель
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train.values.ravel())

    # Сохраняем модель
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    print(f"[train] Модель сохранена в {model_path}")

# Запуск как скрипт
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="/home/irina/project/data/processed",  # абсолютный путь по умолчанию
        help="Папка с X_train и y_train"
    )
    parser.add_argument(
        "--model_path",
        default="/home/irina/project/results/model.pkl",  # путь для сохранения модели
        help="Путь к файлу для сохранения модели"
    )
    args = parser.parse_args()

    train_model(args.data_dir, args.model_path)

