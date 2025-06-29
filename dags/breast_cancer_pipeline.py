from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
import sys

# Добавляем путь к проекту, чтобы Airflow находил модули
sys.path.append('/home/irina/project')

# Импорт функций из etl
from etl.extract import extract_data
from etl.transform import transform_data
from etl.train import train_model
from etl.evaluate import evaluate_model

default_args = {
    "owner": "airflow",
    "start_date": datetime(2023, 1, 1),
    "retries": 1,
}

with DAG(
    dag_id="breast_cancer_pipeline",
    default_args=default_args,
    schedule_interval=None,  # запуск вручную
    catchup=False,
    tags=["ML", "ETL", "breast_cancer"]
) as dag:

    t1 = PythonOperator(
        task_id="extract_data",
        python_callable=extract_data,
        op_kwargs={
            "input_url": "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
            "output_path": "/home/irina/project/data/raw_data.csv"
        }
    )

    t2 = PythonOperator(
        task_id="transform_data",
        python_callable=transform_data,
        op_kwargs={
            "input_path": "/home/irina/project/data/raw_data.csv",
            "output_dir": "/home/irina/project/data/processed"
        }
    )



    t3 = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
        op_kwargs={
            "data_dir": "/home/irina/project/data/processed",
            "model_path": "/home/irina/project/results/model.pkl"
        }
    )

    t4 = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
        op_kwargs={
            "data_dir": "/home/irina/project/data/processed",
            "model_path": "/home/irina/project/results/model.pkl",
            "output_path": "/home/irina/project/results/metrics.json"
        }
    )

    t1 >> t2 >> t3 >> t4
