from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from airflow.decorators import dag, task
from airflow.models.param import Param

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.evaluation import (
    RunConfig,
    build_run_config,
    prepare_run_dir,
    run_agent,
    run_evaluation,
    summarize_run,
)


@dag(
    dag_id="evaluate_agent",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["swe-bench", "mlflow", "manual"],
    params={
        "split": Param("test", type="string"),
        "subset": Param("verified", type="string"),
        "workers": Param(1, type="integer", minimum=1),
        "model": Param("nebius/moonshotai/Kimi-K2.6", type="string"),
        "task_slice": Param("0:1", type="string"),
        "run_id": Param("", type="string"),
        "cost_limit": Param(0, type=["number", "integer"]),
    },
)
def evaluate_agent_dag():
    @task
    def prepare_run(**context) -> dict:
        config = build_run_config(context["params"], PROJECT_ROOT)
        prepare_run_dir(config)
        return config.__dict__

    @task
    def run_agent_task(config_dict: dict) -> str:
        return run_agent(RunConfig(**config_dict))

    @task
    def run_eval_task(config_dict: dict, predictions_path: str) -> str:
        return run_evaluation(RunConfig(**config_dict), predictions_path)

    @task
    def summarize_and_log_task(config_dict: dict, _eval_dir: str) -> dict:
        return summarize_run(RunConfig(**config_dict))

    config = prepare_run()
    predictions = run_agent_task(config)
    eval_dir = run_eval_task(config, predictions)
    summarize_and_log_task(config, eval_dir)


evaluate_agent_dag()
