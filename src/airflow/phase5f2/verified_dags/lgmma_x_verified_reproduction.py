"""Isolated Phase 5F.2 version of the VERIFIED inference-only DAG.

A copy, never a replacement of the historical or Phase 5F.1 DAG.
Failure of the verified runner blocks downstream tasks; no retraining or promotion.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator

PROJECT = '/opt/airflow/project'
PYTHON = '/opt/lgmma-verified-python/bin/python'
SAFE_RUN_ID = '{{ ts_nodash }}'
CMD = f'cd {PROJECT} && {PYTHON} {PROJECT}/lgmma_verified_runner.py'

with DAG(
    dag_id='lgmma_x_verified_reproduction',
    description='Isolated Phase 5F.2: verified frozen inference, no retraining',
    start_date=datetime(2026, 11, 18),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=8),
    default_args={'owner': 'datasci2', 'retries': 0},
    tags=['lgmma-x', 'verified', 'manual', 'no-retrain'],
) as dag:
    preflight = BashOperator(
        task_id='verify_inputs_and_environment',
        bash_command=CMD + f' preflight --run-id "{SAFE_RUN_ID}"',
        execution_timeout=timedelta(minutes=10),
    )
    replay = BashOperator(
        task_id='replay_full_frozen_experiment',
        bash_command=CMD + f' replay --run-id "{SAFE_RUN_ID}"',
        execution_timeout=timedelta(hours=7),
    )
    verify = BashOperator(
        task_id='enforce_final_verification_gate',
        bash_command=CMD + f' verify --run-id "{SAFE_RUN_ID}"',
        execution_timeout=timedelta(minutes=10),
    )
    preflight >> replay >> verify
