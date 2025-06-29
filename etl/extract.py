import pandas as pd
import argparse
import os

def extract_data(input_path: str, output_path: str):
    df = pd.read_csv(input_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[extract] Данные сохранены в {output_path}")

import pandas as pd
import argparse
import os

# Полные названия признаков из описания UCI
COLUMN_NAMES = [
    "id", "diagnosis",
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
]

def extract_data(input_url, output_path):
    df = pd.read_csv(input_url, header=None)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[extract] Данные загружены из {input_url} и сохранены в {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_url", default="https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data")
    parser.add_argument("--output_path", default="/home/irina/project/data/raw_data.csv")
    args = parser.parse_args()

    extract_data(args.input_url, args.output_path)
