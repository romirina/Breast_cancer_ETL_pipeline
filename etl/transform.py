import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse
import os
import logging

def transform_data(input_path, output_dir):
    logging.info(f"Загружаю данные из: {input_path}")

    column_names = [
        'id', 'diagnosis',
        *['feature_' + str(i) for i in range(1, 31)]
    ]
    
    df = pd.read_csv(input_path, header=None, names=column_names)
    logging.info(f"Форма данных: {df.shape}")

    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

    # 👉 проверка внутри функции:
    missing = df['diagnosis'].isna().sum()
    if missing > 0:
        logging.warning(f"Удалено {missing} строк с некорректным diagnosis")
        df = df.dropna(subset=['diagnosis'])

    X = df.drop(columns=['diagnosis'])
    y = df['diagnosis']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logging.info("Масштабирование признаков выполнено")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    logging.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(X_train).to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    pd.DataFrame(X_test).to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    logging.info(f"Файлы сохранены в: {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="/home/irina/project/data/raw_data.csv")
    parser.add_argument("--output_dir", default="/home/irina/project/data/processed")
    args = parser.parse_args()

    transform_data(args.input_path, args.output_dir)
