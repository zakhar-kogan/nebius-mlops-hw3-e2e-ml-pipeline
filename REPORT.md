# Evaluation Pipeline Preparation

## Current State

The repo runs a manual Airflow-triggered `prepare_run -> run_agent_task -> run_eval_task -> summarize_and_log_task` workflow. `prepare_run` and `summarize_and_log_task` run in Airflow Python tasks; the heavy agent and SWE-bench evaluation steps run in isolated DockerOperator containers. Successful standalone and production-style smoke runs have been completed and uploaded to S3-compatible object storage.

## Services

Start MLflow:

```bash
docker compose up -d mlflow
```

Start Airflow from the repo root:

```bash
bash run-airflow-standalone.sh
```

The startup script re-enters the `docker` group with `sg docker` when needed so Airflow workers can access `/var/run/docker.sock`.

Open:

- Airflow: http://localhost:8080
- MLflow: http://localhost:5000

## Production-Style Compose Stack

Build the shared pipeline image and start MLflow plus Airflow:

```bash
docker compose build
docker compose up -d mlflow airflow
```

The Compose Airflow service uses the same image as DockerOperator task containers. It mounts the repo at `HOST_PROJECT_ROOT`, mounts `/var/run/docker.sock`, and runs the heavy `run_agent` and `run_eval` steps in isolated DockerOperator containers.

Open:

- Airflow: http://localhost:8080
- MLflow: http://localhost:5000

Default standalone/compose Airflow credentials:

```text
admin / admin
```

Trigger the production-style smoke run from Airflow with:

```json
{
  "run_id": "compose-dockeroperator-smoke",
  "task_slice": "0:1",
  "workers": 1,
  "cost_limit": 0
}
```

Verify remote artifacts:

```bash
aws s3 ls s3://nebius-mlops-hw3/mlops-assignment-runs/compose-dockeroperator-smoke/ --recursive
```

Completed production-style verification:

- Run name: `compose-dockeroperator-smoke-final`
- Airflow status: all tasks succeeded
- Local run folder: `runs/compose-dockeroperator-smoke-final/`
- Remote artifact URI: `s3://nebius-mlops-hw3/mlops-assignment-runs/compose-dockeroperator-smoke-final/`
- MLflow experiment: `coding-agent-evals`
- MLflow status: `FINISHED`
- Metrics: `total_instances=1`, `resolved_rate=1.0`, `patch_applied_rate=1.0`, `failed_eval_count=0`
- Manifest commands use direct in-container entrypoints (`mini-extra ...` and `python -m swebench...`), not host `uv run` wrappers.

Verify that production-style run:

```bash
aws s3 ls s3://nebius-mlops-hw3/mlops-assignment-runs/compose-dockeroperator-smoke-final/ --recursive
```

## Configuration

Copy `.env.example` to `.env` and fill secrets locally. `S3_BUCKET` controls artifact upload: leave it empty to disable uploads. The current run uses Cloudflare R2 through the S3-compatible endpoint. The DAG logs to MLflow only when `MLFLOW_TRACKING_URI` is present in the Airflow environment and the final task actually runs.


## Run Configuration

Required local inputs:

- Put `NEBIUS_API_KEY` in `.env`.
- Configure S3-compatible credentials with `aws configure`; this run used Cloudflare R2.
- The model, cost limit, workers, split, subset, and task slice are Airflow DAG params, so they do not need to be hard-coded in `.env`.
- The upstream `mini-swe-agent/` and `SWE-bench/` clones are useful reference/debugging material, but the first run does not require them now that the DAG uses the installed mini-swe-agent default config.

## DAG

`evaluate_agent` is manual-only (`schedule=None`) with safe defaults:

- `split=test`
- `subset=verified`
- `workers=1`
- `task_slice=0:1`
- `cost_limit=0`

Each real run writes `runs/<run-id>/` with config, predictions, trajectories, evaluation outputs, metrics, and a manifest.

## Completed Smoke Run

Triggered Airflow DAG `evaluate_agent` with:

```json
{
  "run_id": "first-r2-smoke-real",
  "task_slice": "0:1",
  "workers": 1,
  "cost_limit": 0
}
```

All Airflow tasks succeeded:

- `prepare_run`
- `run_agent_task`
- `run_eval_task`
- `summarize_and_log_task`

Metrics from `runs/first-r2-smoke-real/metrics.json`:

- `total_instances`: 1
- `resolved_count`: 1
- `resolved_rate`: 1.0
- `patch_applied_count`: 1
- `patch_applied_rate`: 1.0
- `failed_eval_count`: 0

Artifacts:

- Local run folder: `runs/first-r2-smoke-real/`
- Manifest: `runs/first-r2-smoke-real/manifest.json`
- Remote artifact URI: `s3://nebius-mlops-hw3/mlops-assignment-runs/first-r2-smoke-real/`

MLflow:

- Experiment: `coding-agent-evals`
- Run name: `first-r2-smoke-real`
- Status: `FINISHED`
- `artifact_uri` tag: `s3://nebius-mlops-hw3/mlops-assignment-runs/first-r2-smoke-real/`

Verify remote artifacts:

```bash
aws s3 ls s3://nebius-mlops-hw3/mlops-assignment-runs/first-r2-smoke-real/ --recursive
```
