#!/usr/bin/env python
# coding: utf-8

from airflow.sdk import task, dag, Param
from airflow.models.baseoperator import chain
from datetime import datetime, timedelta

import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import Pipeline

import mlflow

@dag(
    dag_id='linear_regression_duration_prediction',
    schedule='0 22 * * *',
    default_args={'start_date':  datetime(2025, 6, 21), "retries": 2, "retry_delay": timedelta(minutes=5)},
    params={'year': Param(2025, type='integer'),
            'month': Param(1, type='integer')},
    catchup=False
)

def duration_prediction_dag():

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
    def train_model(**context):
        year = context['params']['year']
        month = context['params']['month']

        mlflow.set_tracking_uri("http://localhost:5000")
        print("Connected to MLflow tracking server.")

        with mlflow.start_run() as run:
            mlflow.set_tag("mlflow.runName", f"{year}-{month:02d}")

            X = pickle.load(open(context['ti'].xcom_pull(task_ids='create_X', key='dicts_path'), 'rb'))
            y = pickle.load(open(context['ti'].xcom_pull(task_ids='create_y', key='y_path'), 'rb'))
            
            pipeline = Pipeline(steps=[
                ('dv', DictVectorizer(sparse=True)),
                ('lr', LinearRegression())
            ])
            
            lr = pipeline.fit(X, y)
            mlflow.sklearn.log_model(lr, artifact_path="models")

            y_pred = lr.predict(X)
            rmse = root_mean_squared_error(y, y_pred)
            mlflow.log_metric("rmse", rmse)
                
            print(f"Model trained and logged with run_id: {run.info.run_id}")
        
        return run.info.run_id

    chain(
        read_dataframe(),
        create_X(),
        create_y(),
        train_model()
        )

duration_prediction_dag()