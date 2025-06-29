from airflow.sdk import task, dag, Param
from airflow.models.baseoperator import chain
from datetime import datetime, timedelta

import pickle
import pandas as pd


from evidently import Dataset, DataDefinition, Report, Regression
from evidently.presets import DataDriftPreset, RegressionPreset

import sqlite3

target = "duration_min"
num_features = ["passenger_count", "trip_distance", "fare_amount", "total_amount"]
cat_features = ["PULocationID", "DOLocationID"]


@dag(
    dag_id='performance',
    schedule='0 22 * * *',
    default_args={'start_date':  datetime(2025, 6, 21), "retries": 2, "retry_delay": timedelta(minutes=5)},
    params={'year': Param(2021, type='integer'),
            'month': Param(1, type='integer')},
    catchup=False
)

def performance_dag():

    @task
    def read_dataframe(**context):
        year = context['params']['year']
        month = context['params']['month']

        url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
        df = pd.read_parquet(url)

        # create target
        df["duration_min"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
        df.duration_min = df.duration_min.apply(lambda td : float(td.total_seconds())/60)

        # filter out outliers
        df = df[(df.duration_min >= 0) & (df.duration_min <= 60)]
        df = df[(df.passenger_count > 0) & (df.passenger_count <= 8)]

        df = df[[target] + num_features + cat_features]


        df_path = f'/tmp/green_tripdata_{year}-{month:02d}.csv'
        df.to_csv(df_path, index=False)
        context['ti'].xcom_push(key=f'prepared_df', value=df_path)

        print(f'Dataframe saved to {df_path}')

    @task
    def predict(**context):
        year = context['params']['year']
        month = context['params']['month']

        df = pd.read_csv(context['ti'].xcom_pull(task_ids='read_dataframe', key=f"prepared_df"))
        df[cat_features] = df[cat_features].astype(str)

        model = pickle.load(open('lin_reg.bin', 'rb'))

        df['prediction'] = model.predict(df[num_features + cat_features])

        output_path = f'/tmp/predictions_{year}-{month:02d}.csv'
        df.to_csv(output_path, index=False)
        context['ti'].xcom_push(key=f'predictions', value=output_path)

        print(f'Predictions saved to {output_path}')
        
    
    @task
    def report(**context):
        year = context['params']['year']
        month = context['params']['month']
        
        reference = pd.read_parquet(f'reference.parquet')
        current = pd.read_csv(context['ti'].xcom_pull(task_ids='predict', key=f'predictions'))

        schema = DataDefinition(
            numerical_columns=num_features,
            categorical_columns=cat_features,
            regression=[Regression(target="duration_min", prediction="prediction")]
            )

        current_dataset = Dataset.from_pandas(
            current,
            data_definition=schema
        )

        reference_dataset = Dataset.from_pandas(
            reference,
            data_definition=schema
        )

        report = Report([
            DataDriftPreset(),
            RegressionPreset()
        ])

        my_eval = report.run(reference_data=reference_dataset, current_data=current_dataset)
        result = my_eval.dict()
        
        report_path = f'reports/report_{year}-{month:02d}.pkl'
        with open(report_path, 'wb') as f:
            pickle.dump(result, f)
        context['ti'].xcom_push(key=f'report_{year}-{month:02d}', value=report_path)

        print(f'Report saved to {report_path}')

    @task
    def insert_into_db(**context):
        year = context['params']['year']
        month = context['params']['month']
        report_path = context['ti'].xcom_pull(task_ids='report', key=f'report_{year}-{month:02d}')
        with open(report_path, 'rb') as f:
            report = pickle.load(f)
        print(report)

        month_new = 1 if month == 12 else month + 1
        year_new = year if month_new > 1 else year + 1
        
        share_of_drifted_columns = report['metrics'][0]['value']['share']
        RMSE = report['metrics'][9]['value']

        connection = sqlite3.connect('dbs/performance.db')
        cursor = connection.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY,
            time_col TIMESTAMP,
            share_of_drifted_columns REAL,
            RMSE REAL
        )
        """)
        connection.commit()

        cursor.execute("""
        INSERT INTO metrics (time_col, share_of_drifted_columns, RMSE)
        VALUES (?, ?, ?)
        """, (datetime(year_new, month_new, 1).timestamp(), share_of_drifted_columns, RMSE))
        connection.commit()

        connection.close()

        print(f'Metrics inserted into database for {year}-{month:02d}')

    chain(
        read_dataframe(),
        predict(),
        report(),
        insert_into_db()
    )

performance_dag()