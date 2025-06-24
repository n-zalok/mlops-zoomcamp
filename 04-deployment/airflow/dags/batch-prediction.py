#!/usr/bin/env python
# coding: utf-8

from airflow.sdk import task, dag, Param
from airflow.models.baseoperator import chain
from datetime import datetime, timedelta

import pickle
import pandas as pd
import uuid
from pathlib import Path


import mlflow

@dag(
    dag_id='linear_regression_batch_prediction',
    schedule='0 22 * * *',
    default_args={'start_date':  datetime(2025, 6, 21), "retries": 2, "retry_delay": timedelta(minutes=5)},
    params={'year': Param(2025, type='integer'),
            'month': Param(1, type='integer'),
            'run_id': Param("", type='string')},
    catchup=False
)

def batch_prediction_dag():

    @task
    def read_dataframe(**context):
        year = context['params']['year']
        month = context['params']['month']

        url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
        df = pd.read_parquet(url)

        df['duration'] = df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
        df['duration'] = df['duration'].apply(lambda td: td.total_seconds() / 60)

        df = df[(df['duration'] >= 1) & (df['duration'] <= 60)]

        categorical = ['PULocationID', 'DOLocationID']
        df[categorical] = df[categorical].astype(str)

        ride_ids = [str(uuid.uuid4()) for _ in range(len(df))]
        ride_ids_path = f'/tmp/ride_ids_{year}-{month:02d}.b'
        with open(ride_ids_path, "wb") as f_out:
            pickle.dump(ride_ids, f_out)
        context['ti'].xcom_push(key='ride_ids_path', value=ride_ids_path)

        df_path = f'/tmp/yellow_tripdata_{year}-{month:02d}.csv'
        df.to_csv(df_path, index=False)
        context['ti'].xcom_push(key=f'{year}-{month:02d}', value=df_path)

        print(f'Dataframe saved to {df_path}')

    @task
    def create_X(**context):
        year = context['params']['year']
        month = context['params']['month']

        categorical = ['PULocationID', 'DOLocationID']
        numerical = ['trip_distance']
        
        df = pd.read_csv(context['ti'].xcom_pull(task_ids='read_dataframe', key=f"{year}-{month:02d}"))
        dicts = df[categorical + numerical].to_dict(orient='records')

        dicts_path = f'/tmp/dicts_{year}-{month:02d}.b'
        with open(dicts_path, "wb") as f_out:
            pickle.dump(dicts, f_out)
        context['ti'].xcom_push(key='dicts_path', value=dicts_path)

        print('Created feature matrix X')
    
    @task
    def create_y(**context):
        year = context['params']['year']
        month = context['params']['month']
        target = 'duration'

        df = pd.read_csv(context['ti'].xcom_pull(task_ids='read_dataframe', key=f"{year}-{month:02d}"))
        y = df[target].values

        y_path = f'/tmp/y_{year}-{month:02d}.b'
        with open(y_path, "wb") as f_out:
            pickle.dump(y, f_out)
        context['ti'].xcom_push(key='y_path', value=y_path)

        print('Created target vector y')

    @task
    def evaluate_batch(**context):
        year = context['params']['year']
        month = context['params']['month']

        mlflow.set_tracking_uri("http://localhost:5000")
        print("Connected to MLflow tracking server.")

        mlflow_run_id = context['params']['run_id']
        print(f"Using run_id: {mlflow_run_id}")

        model = mlflow.sklearn.load_model(f"runs:/{mlflow_run_id}/models")
        print("Model loaded successfully.")

        X = pickle.load(open(context['ti'].xcom_pull(task_ids='create_X', key='dicts_path'), 'rb'))
        y = pickle.load(open(context['ti'].xcom_pull(task_ids='create_y', key='y_path'), 'rb'))
        ride_ids = pickle.load(open(context['ti'].xcom_pull(task_ids='read_dataframe', key='ride_ids_path'), 'rb'))

        y_pred = model.predict(X)
        output = pd.DataFrame({'ride_id': ride_ids, 'actual': y, 'predicted': y_pred})

        Path(f'evaluation/{mlflow_run_id}').mkdir(parents=True, exist_ok=True)
        output.to_csv(f'evaluation/{mlflow_run_id}/batch_evaluation_{year}-{month:02d}.csv', index=False)

        print(f'Batch evaluation results saved to evaluation/{mlflow_run_id}/batch_evaluation_{year}-{month:02d}.csv')

    chain(
        read_dataframe(),
        create_X(),
        create_y(),
        evaluate_batch()
    )

batch_prediction_dag()