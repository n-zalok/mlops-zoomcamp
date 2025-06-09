from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

with DAG(
    dag_id='train_nyc_duration_model_on_demand',
    description='Train NYC taxi duration model with custom year and month',
    start_date=days_ago(1),
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    tags=['ml', 'xgboost', 'on_demand'],
) as dag:

    run_training = BashOperator(
        task_id='train_duration_model',
        bash_command=(
            "python /workspaces/mlops-zoomcamp/03-orchestration/duration-prediction.py "
            "--year {{ dag_run.conf.get('year', 2023) }} "
            "--month {{ dag_run.conf.get('month', 3) }}"
        )
    )

    run_training