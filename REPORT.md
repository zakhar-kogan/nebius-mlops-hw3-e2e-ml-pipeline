# Evaluation Pipeline Preparation

## Current State

The repo runs a manual Airflow-triggered `prepare_run -> run_agent -> run_eval -> summarize_and_log` workflow. A successful smoke run has been completed and uploaded to S3-compatible object storage.

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
