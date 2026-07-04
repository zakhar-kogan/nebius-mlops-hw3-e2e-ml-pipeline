from __future__ import annotations

import os
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from airflow.decorators import dag, task
from airflow.models.param import Param
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.evaluation import (
    RunConfig,
    build_run_config,
    prepare_run_dir,
    summarize_run,
)

PIPELINE_IMAGE = os.getenv("PIPELINE_IMAGE", "mlops-assignment-pipeline:latest")
DOCKER_NETWORK = os.getenv("PIPELINE_DOCKER_NETWORK", "bridge")
DOCKER_URL = os.getenv("PIPELINE_DOCKER_URL", "unix://var/run/docker.sock")


def task_environment() -> dict[str, str]:
    keys = [
        "NEBIUS_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "HF_TOKEN",
        "MSWEA_MODEL_NAME",
    ]
    env = {key: value for key in keys if (value := os.getenv(key))}
    env["MSWEA_COST_TRACKING"] = os.getenv("MSWEA_COST_TRACKING", "ignore_errors")
    return env


def docker_task(task_id: str, command: str) -> DockerOperator:
    return DockerOperator(
        task_id=task_id,
        image=PIPELINE_IMAGE,
        command=["bash", "-lc", command],
        docker_url=DOCKER_URL,
        network_mode=DOCKER_NETWORK,
        working_dir=str(PROJECT_ROOT),
        environment=task_environment(),
        mounts=[
            Mount(source=str(PROJECT_ROOT), target=str(PROJECT_ROOT), type="bind"),
            Mount(source="/var/run/docker.sock", target="/var/run/docker.sock", type="bind"),
        ],
        mount_tmp_dir=False,
        auto_remove="success",
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
        config = replace(build_run_config(context["params"], PROJECT_ROOT), use_uv=False)
        prepare_run_dir(config)
        return config.__dict__

    @task
    def summarize_and_log_task(config_dict: dict) -> dict:
        return summarize_run(RunConfig(**config_dict))

    config = prepare_run()
    config_path = "{{ ti.xcom_pull(task_ids='prepare_run')['project_root'] }}/runs/{{ ti.xcom_pull(task_ids='prepare_run')['run_id'] }}/config.json"
    predictions_path = "{{ ti.xcom_pull(task_ids='prepare_run')['project_root'] }}/runs/{{ ti.xcom_pull(task_ids='prepare_run')['run_id'] }}/run-agent/preds.json"

    run_agent_task = docker_task(
        "run_agent_task",
        f"python -m pipeline.evaluation run-agent --config-path {config_path}",
    )
    run_eval_task = docker_task(
        "run_eval_task",
        f"python -m pipeline.evaluation run-eval --config-path {config_path} --predictions-path {predictions_path}",
    )
    summarize = summarize_and_log_task(config)

    config >> run_agent_task >> run_eval_task >> summarize


evaluate_agent_dag()
