from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "nebius/moonshotai/Kimi-K2.6"
DEFAULT_AGENT_CONFIG = ""


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    split: str
    subset: str
    workers: int
    model: str
    task_slice: str
    cost_limit: float
    dataset_name: str
    agent_config: str
    project_root: str
    created_at: str


def safe_run_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.=-]+", "-", value.strip())
    cleaned = cleaned.strip("-_.")
    if not cleaned:
        raise ValueError("run_id must contain at least one safe character")
    return cleaned[:120]


def generated_run_id() -> str:
    return datetime.now(timezone.utc).strftime("eval-%Y%m%dT%H%M%SZ")


def build_run_config(params: dict[str, Any], project_root: Path) -> RunConfig:
    run_id = safe_run_id(str(params.get("run_id") or generated_run_id()))
    subset = str(params.get("subset") or "verified")
    split = str(params.get("split") or "test")
    workers = int(params.get("workers") or 1)
    if workers < 1:
        raise ValueError("workers must be >= 1")

    cost_limit = float(params.get("cost_limit") if params.get("cost_limit") is not None else 0)
    task_slice = str(params.get("task_slice") or "0:1")
    model = str(params.get("model") or DEFAULT_MODEL)
    dataset_name = str(params.get("dataset_name") or f"princeton-nlp/SWE-bench_{subset.capitalize()}")
    agent_config = str(params.get("agent_config") or DEFAULT_AGENT_CONFIG)

    return RunConfig(
        run_id=run_id,
        split=split,
        subset=subset,
        workers=workers,
        model=model,
        task_slice=task_slice,
        cost_limit=cost_limit,
        dataset_name=dataset_name,
        agent_config=agent_config,
        project_root=str(project_root),
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def run_dir(config: RunConfig) -> Path:
    return Path(config.project_root) / "runs" / config.run_id


def prepare_run_dir(config: RunConfig) -> str:
    root = run_dir(config)
    if root.exists():
        raise FileExistsError(f"run directory already exists: {root}")

    (root / "run-agent" / "trajectories").mkdir(parents=True)
    (root / "run-eval").mkdir(parents=True)
    write_json(root / "config.json", asdict(config))
    return str(root)


def build_agent_command(config: RunConfig, output_dir: Path | None = None) -> list[str]:
    output = output_dir or run_dir(config) / "run-agent" / "trajectories"
    command = [
        "uv",
        "run",
        "mini-extra",
        "swebench",
        "--subset",
        config.subset,
        "--split",
        config.split,
        "--model",
        config.model,
        "--slice",
        config.task_slice,
        "--workers",
        str(config.workers),
        "-o",
        str(output),
    ]
    if config.agent_config:
        command.extend(["--config", config.agent_config])
    if config.cost_limit >= 0:
        if not config.agent_config:
            command.extend(["--config", "swebench.yaml"])
        command.extend(["--config", f"agent.cost_limit={config.cost_limit}"])
    return command


def build_eval_command(config: RunConfig, predictions_path: Path) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        config.dataset_name,
        "--predictions_path",
        str(predictions_path),
        "--max_workers",
        str(config.workers),
        "--run_id",
        config.run_id,
    ]


def run_agent(config: RunConfig) -> str:
    root = run_dir(config)
    agent_dir = root / "run-agent"
    trajectories_dir = agent_dir / "trajectories"
    env = {
        **os.environ,
        "MSWEA_COST_TRACKING": "ignore_errors",
    }
    subprocess.run(
        build_agent_command(config, trajectories_dir),
        cwd=config.project_root,
        env=env,
        check=True,
    )

    preds = trajectories_dir / "preds.json"
    if not preds.exists():
        raise FileNotFoundError(f"mini-swe-agent did not produce {preds}")
    shutil.copy2(preds, agent_dir / "preds.json")
    return str(agent_dir / "preds.json")


def run_evaluation(config: RunConfig, predictions_path: str) -> str:
    root = run_dir(config)
    eval_dir = root / "run-eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    with (eval_dir / "run_evaluation.stdout.log").open("w", encoding="utf-8") as stdout:
        subprocess.run(
            build_eval_command(config, Path(predictions_path)),
            cwd=config.project_root,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            check=True,
        )
    _copy_swebench_logs(config, eval_dir)
    return str(eval_dir)


def collect_metrics(eval_dir: str | Path, agent_dir: str | Path | None = None) -> dict[str, float | int]:
    reports = list(Path(eval_dir).rglob("report.json"))
    total = 0
    resolved = 0
    patch_applied = 0
    missing_patch = 0
    failed_eval = 0

    for report_path in reports:
        report = read_json(report_path)
        for instance in report.values():
            total += 1
            resolved += int(bool(instance.get("resolved")))
            patch_applied += int(bool(instance.get("patch_successfully_applied")))
            missing_patch += int(bool(instance.get("patch_is_None")) or not bool(instance.get("patch_exists", True)))
            failed_eval += int(not bool(instance.get("patch_successfully_applied")) and not bool(instance.get("resolved")))
    metrics: dict[str, float | int] = {
        "total_instances": total,
        "resolved_count": resolved,
        "resolved_rate": resolved / total if total else 0.0,
        "patch_applied_count": patch_applied,
        "patch_applied_rate": patch_applied / total if total else 0.0,
        "missing_patch_count": missing_patch,
        "failed_eval_count": failed_eval,
    }
    if agent_dir:
        metrics.update(_collect_cost_metrics(Path(agent_dir)))
    return metrics


def write_manifest(config: RunConfig, metrics: dict[str, Any], artifact_uri: str | None = None) -> str:
    root = run_dir(config)
    manifest = {
        "run_id": config.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": "config.json",
        "metrics": "metrics.json",
        "artifact_uri": artifact_uri,
        "paths": {
            "run_dir": str(root),
            "agent_dir": str(root / "run-agent"),
            "predictions": str(root / "run-agent" / "preds.json"),
            "trajectories": str(root / "run-agent" / "trajectories"),
            "eval_dir": str(root / "run-eval"),
        },
        "commands": {
            "run_agent": build_agent_command(config),
            "run_eval": build_eval_command(config, root / "run-agent" / "preds.json"),
        },
        "metrics_summary": metrics,
    }
    path = root / "manifest.json"
    write_json(path, manifest)
    return str(path)


def summarize_run(config: RunConfig) -> dict[str, Any]:
    root = run_dir(config)
    metrics = collect_metrics(root / "run-eval", root / "run-agent")
    write_json(root / "metrics.json", metrics)
    artifact_uri = remote_artifact_uri(root)
    manifest_path = write_manifest(config, metrics, artifact_uri)
    uploaded_uri = upload_run_artifacts_if_configured(root)
    log_mlflow_run(config, metrics, str(root), uploaded_uri)
    return {"metrics": metrics, "manifest": manifest_path, "artifact_uri": artifact_uri}


def remote_artifact_uri(root: Path) -> str | None:
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        return None
    prefix = os.getenv("S3_PREFIX", "mlops-assignment-runs").strip("/")
    key_prefix = "/".join(part for part in (prefix, root.name) if part).strip("/")
    return f"s3://{bucket}/{key_prefix}/"


def upload_run_artifacts_if_configured(root: Path) -> str | None:
    artifact_uri = remote_artifact_uri(root)
    if not artifact_uri:
        return None

    import boto3

    bucket = os.getenv("S3_BUCKET")
    endpoint_url = os.getenv("AWS_ENDPOINT_URL")
    prefix = os.getenv("S3_PREFIX", "mlops-assignment-runs").strip("/")
    client = boto3.client("s3", endpoint_url=endpoint_url)
    key_prefix = "/".join(part for part in (prefix, root.name) if part).strip("/")

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative_key = path.relative_to(root).as_posix()
        key = f"{key_prefix}/{relative_key}" if key_prefix else relative_key
        client.upload_file(str(path), bucket, key)

    return artifact_uri


def log_mlflow_run(
    config: RunConfig,
    metrics: dict[str, Any],
    local_artifact_path: str,
    artifact_uri: str | None,
) -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        return

    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "coding-agent-evals"))
    with mlflow.start_run(run_name=config.run_id):
        mlflow.log_params(
            {
                "run_id": config.run_id,
                "split": config.split,
                "subset": config.subset,
                "workers": config.workers,
                "model": config.model,
                "task_slice": config.task_slice,
                "cost_limit": config.cost_limit,
                "dataset_name": config.dataset_name,
            }
        )
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
        mlflow.set_tag("local_artifact_path", local_artifact_path)
        if artifact_uri:
            mlflow.set_tag("artifact_uri", artifact_uri)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_swebench_logs(config: RunConfig, eval_dir: Path) -> None:
    logs_root = Path(config.project_root) / "logs" / "run_evaluation" / config.run_id
    if logs_root.exists():
        destination = eval_dir / "logs"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(logs_root, destination)


def _collect_cost_metrics(agent_dir: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    total_cost = 0.0
    cost_count = 0

    for traj_path in agent_dir.rglob("*.traj.json"):
        try:
            data = read_json(traj_path)
        except Exception:
            continue
        text = json.dumps(data)
        for key in ("cost", "total_cost", "tokens", "input_tokens", "output_tokens"):
            values = [float(match) for match in re.findall(rf'"{key}"\s*:\s*([0-9.]+)', text)]
            if values:
                metrics[f"{key}_sum"] = metrics.get(f"{key}_sum", 0.0) + sum(values)
                if key in {"cost", "total_cost"}:
                    total_cost += sum(values)
                    cost_count += len(values)

    if cost_count:
        metrics["cost_observations"] = float(cost_count)
        metrics["cost_sum"] = total_cost
    return metrics
