# Evaluation Pipeline Preparation

## Current State

The repo runs a manual Airflow-triggered `prepare_run -> run_agent_task -> run_eval_task -> summarize_and_log_task` workflow. `prepare_run` and `summarize_and_log_task` run in Airflow Python tasks; the heavy agent and SWE-bench evaluation steps run in isolated DockerOperator containers. A three-instance production-style run has been completed, logged to MLflow, and uploaded to S3-compatible object storage.

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
docker compose -f docker-compose.yaml build
docker compose -f docker-compose.yaml up -d mlflow airflow
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
  "run_id": "compose-dockeroperator-slice-3",
  "task_slice": "0:3",
  "workers": 1,
  "cost_limit": 0,
  "input_cost_per_1m_tokens": 0.95,
  "output_cost_per_1m_tokens": 4.0
}
```

Verify remote artifacts:

```bash
aws s3 ls s3://nebius-mlops-hw3/mlops-assignment-runs/compose-dockeroperator-slice-3/ --recursive
```

Completed production-style verification:

- Run name: `compose-dockeroperator-slice-3`
- Airflow status: all tasks succeeded
- Local run folder: `runs/compose-dockeroperator-slice-3/`
- Remote artifact URI: `s3://nebius-mlops-hw3/mlops-assignment-runs/compose-dockeroperator-slice-3/`
- MLflow experiment: `coding-agent-evals`
- MLflow run id: `666423b57f5f4dfda0daf59f5f0a50fb`
- MLflow status: `FINISHED`
- Metrics: `total_instances=3`, `resolved_rate=0.3333333333333333`, `patch_applied_rate=1.0`, `failed_eval_count=0`
- Token/cost metrics: `prompt_tokens=1548019`, `completion_tokens=71868`, `estimated_total_cost=1.75809005`
- MLflow artifacts include config, manifest, metrics, predictions, trajectories, eval logs, and reports.
- MLflow traces: 3 lightweight traces derived from mini-swe-agent trajectories.
- Manifest commands use direct in-container entrypoints (`mini-extra ...` and `python -m swebench...`), not host `uv run` wrappers.

Submission bundle in this repository:

- Pipeline code: `dags/evaluate_agent.py`, `pipeline/`, `scripts/`, `Dockerfile`, `docker-compose.yaml`.
- Environment template: `.env.example`.
- Final committed run sample: `runs/compose-dockeroperator-slice-3/`.
- Object storage proof: `evidence/s3/compose-dockeroperator-slice-3-listing.txt`.
- MLflow proof: `evidence/mlflow/compose-dockeroperator-slice-3-summary.json`.
- Screenshots: `screenshots/airflow_dag.png`, `screenshots/mlflow_runs.png`, `screenshots/object_storage_artifacts.png`.

Verify that production-style run:

```bash
aws s3 ls s3://nebius-mlops-hw3/mlops-assignment-runs/compose-dockeroperator-slice-3/ --recursive
```

## Configuration

Copy `.env.example` to `.env` and fill secrets locally. `S3_BUCKET` controls artifact upload: leave it empty to disable uploads. The current run uses Cloudflare R2 through the S3-compatible endpoint. The DAG logs to MLflow only when `MLFLOW_TRACKING_URI` is present in the Airflow environment and the final task actually runs.

Object Storage note: Nebius Object Storage access was attempted first, but the available service-account access was not sufficient to write to the intended Nebius bucket during the submission window. To keep the required long-term artifact storage behavior working and reproducible, the final run uses Cloudflare R2 as an S3-compatible Object Storage backend. The DAG and upload code use standard AWS/S3-compatible configuration, so the same pipeline can target Nebius Object Storage by changing the configured endpoint and credentials.


## Run Configuration

Required local inputs:

- Put `NEBIUS_API_KEY` in `.env`.
- Configure S3-compatible credentials with `aws configure`; this run used Cloudflare R2.
- The model, cost limit, workers, split, subset, task slice, rerun source, and optional input/output token prices are Airflow DAG params, so they do not need to be hard-coded in `.env`.
- `cost_limit` is passed to mini-swe-agent/LiteLLM as a best-effort runtime guard. Since provider pricing may be unavailable for some Nebius models, the DAG also accepts `input_cost_per_1m_tokens` and `output_cost_per_1m_tokens` for reporting fallback estimates.
- The upstream `mini-swe-agent/` and `SWE-bench/` clones are useful reference/debugging material, but the first run does not require them now that the DAG uses the installed mini-swe-agent default config.

## DAG

`evaluate_agent` is manual-only (`schedule=None`) with safe defaults:

- `split=test`
- `subset=verified`
- `workers=1`
- `task_slice=0:1`
- `cost_limit=0`
- `rerun_from_run_id=""`
- `input_cost_per_1m_tokens=0`
- `output_cost_per_1m_tokens=0`

Each real run writes `runs/<run-id>/` with config, predictions, trajectories, evaluation outputs, metrics, and a manifest.

The committed sample follows this layout:

```text
runs/compose-dockeroperator-slice-3/
  config.json
  run-agent/
    preds.json
    trajectories/
  run-eval/
    logs/
    reports/
  metrics.json
  manifest.json
```

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
