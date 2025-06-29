import os
import argparse
import json
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(data_dir, model_path, output_path):
    # Загружаем тестовые данные
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv"))

    # Загружаем модель и делаем предсказание
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    # Считаем метрики
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }

    # Создаём папку при необходимости и сохраняем метрики
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"[evaluate] Метрики сохранены в {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="/home/irina/project/data/processed",
        help="Папка с X_test и y_test"
    )
    parser.add_argument(
        "--model_path",
        default="/home/irina/project/results/model.pkl",
        help="Путь к сохранённой модели"
    )
    parser.add_argument(
        "--output_path",
        default="/home/irina/project/results/metrics.json",
        help="Путь для сохранения метрик"
    )
    args = parser.parse_args()

    evaluate_model(args.data_dir, args.model_path, args.output_path)

