#!/usr/bin/env python
# coding: utf-8

from airflow.sdk import task, dag, Param
from airflow.models.baseoperator import chain
from datetime import datetime, timedelta

import pickle
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

import mlflow
from mlflow.tracking import MlflowClient

@dag(
    dag_id='linear_regression_duration_prediction',
    schedule='0 22 * * *',
    default_args={'start_date':  datetime(2025, 6, 21), "retries": 2, "retry_delay": timedelta(minutes=5)},
    params={'year': Param(2025, type='integer'),
            'month': Param(1, type='integer'),
            'objective': Param('train', type='string', allowed_values=['train', 'predict']),
            'dv_year': Param(2025, type='integer'),
            'dv_month': Param(1, type='integer')},
    catchup=False
)

def duration_prediction_dag():

    @task
    def connect_to_mlflow():
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        print("Connected to MLflow tracking server.")

    @task
    def create_directories():
        Path('dvs').mkdir(parents=True, exist_ok=True)
        print("Created necessary directories.")

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

        df_path = f'/tmp/yellow_tripdata_{year}-{month:02d}.csv'
        df.to_csv(df_path, index=False)

        context['ti'].xcom_push(key=f'{year}-{month:02d}', value=df_path)
        print(f'Dataframe saved to {df_path}')

    @task
    def create_X(**context):
        year = context['params']['year']
        month = context['params']['month']
        objective = context['params']['objective']

        categorical = ['PULocationID', 'DOLocationID']
        numerical = ['trip_distance']
        
        df = pd.read_csv(context['ti'].xcom_pull(task_ids='read_dataframe', key=f"{year}-{month:02d}"))
        dicts = df[categorical + numerical].to_dict(orient='records')

        if objective == 'train':
            dv = DictVectorizer(sparse=True)
            X = dv.fit_transform(dicts)

            dv_path = f'dvs/dv_{year}-{month:02d}.b'
            with open(dv_path, "wb") as f_out:
                pickle.dump(dv, f_out)
            context['ti'].xcom_push(key='dv_path', value=dv_path)
            
            X_path = f'/tmp/X_{year}-{month:02d}.b'
            with open(X_path, "wb") as f_out:
                pickle.dump(X, f_out)
            context['ti'].xcom_push(key='X_path', value=X_path)
        else:
            dv_year = context['params']['dv_year']
            dv_month = context['params']['dv_month']

            client = MlflowClient()
            runs = client.search_runs(experiment_ids='0', filter_string=f"tags.mlflow.runName='{dv_year}-{dv_month:02d}'", max_results=1)
            run_id = runs[0].info.run_id
            dv_path = client.download_artifacts(run_id=run_id, path=f'preprocessor/dv_{dv_year}-{dv_month:02d}.b')
            print(dv_path)

            dv = pickle.load(open(dv_path, 'rb'))
            X = dv.transform(dicts)

            X_path = f'/tmp/X_{year}-{month:02d}.b'
            with open(X_path, "wb") as f_out:
                pickle.dump(X, f_out)
            context['ti'].xcom_push(key='X_path', value=X_path)

        print('Created feature matrix X with shape:', X.shape)
    
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

        print('Created target vector y with shape:', y.shape)

    @task
    def train_model(**context):
        objective = context['params']['objective']
        year = context['params']['year']
        month = context['params']['month']

        if objective == 'train':
            with mlflow.start_run() as run:
                mlflow.set_tag("mlflow.runName", f"{year}-{month:02d}")

                lr = LinearRegression()
                X = pickle.load(open(context['ti'].xcom_pull(task_ids='create_X', key='X_path'), 'rb'))
                y = pickle.load(open(context['ti'].xcom_pull(task_ids='create_y', key='y_path'), 'rb'))
                lr.fit(X, y)

                y_pred = lr.predict(X)
                rmse = root_mean_squared_error(y, y_pred)
                mlflow.log_metric("rmse", rmse)
                
                dv_path = context['ti'].xcom_pull(task_ids='create_X', key='dv_path')
                mlflow.log_artifact(dv_path, artifact_path="preprocessor")
                mlflow.sklearn.log_model(lr, artifact_path="models")
                print(f"Model trained and logged with run_id: {run.info.run_id}")
        
        else:
            dv_year = context['params']['dv_year']
            dv_month = context['params']['dv_month']

            client = MlflowClient()
            runs = client.search_runs(experiment_ids='0', filter_string=f"tags.mlflow.runName='{dv_year}-{dv_month:02d}'", max_results=1)
            run_id = runs[0].info.run_id

            with mlflow.start_run(run_id=run_id) as run:
                X = pickle.load(open(context['ti'].xcom_pull(task_ids='create_X', key='X_path'), 'rb'))
                y = pickle.load(open(context['ti'].xcom_pull(task_ids='create_y', key='y_path'), 'rb'))

                model = mlflow.sklearn.load_model(f'runs:/{run.info.run_id}/models')
                y_pred = model.predict(X)

                rmse = root_mean_squared_error(y, y_pred)
                mlflow.log_metric(f"rmse-{year}-{month:02d}", rmse)
                print(f"Model evaluated with run_id: {run.info.run_id}")
        return run.info.run_id

    chain(
        connect_to_mlflow(),
        create_directories(),
        read_dataframe(),
        create_X(),
        create_y(),
        train_model()
        )

duration_prediction_dag()