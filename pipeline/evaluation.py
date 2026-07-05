from __future__ import annotations

import argparse
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
    use_uv: bool = True
    rerun_from_run_id: str = ""
    input_cost_per_1m_tokens: float = 0.0
    output_cost_per_1m_tokens: float = 0.0


def safe_run_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.=-]+", "-", value.strip())
    cleaned = cleaned.strip("-_.")
    if not cleaned:
        raise ValueError("run_id must contain at least one safe character")
    return cleaned[:120]


def generated_run_id() -> str:
    return datetime.now(timezone.utc).strftime("eval-%Y%m%dT%H%M%SZ")


def _float_param(params: dict[str, Any], key: str, default: float, env_key: str | None = None) -> float:
    value = params.get(key)
    if value not in (None, ""):
        return float(value)
    if env_key and os.getenv(env_key) not in (None, ""):
        return float(os.getenv(env_key, "0"))
    return float(default)


def build_run_config(params: dict[str, Any], project_root: Path) -> RunConfig:
    rerun_from_run_id = str(params.get("rerun_from_run_id") or "").strip()
    if rerun_from_run_id:
        run_id_value = str(params.get("run_id") or "").strip()
        if not run_id_value:
            raise ValueError("run_id is required when rerun_from_run_id is set")
        config = load_run_config(project_root / "runs" / safe_run_id(rerun_from_run_id) / "config.json")
        return RunConfig(
            run_id=safe_run_id(run_id_value),
            split=config.split,
            subset=config.subset,
            workers=config.workers,
            model=config.model,
            task_slice=config.task_slice,
            cost_limit=config.cost_limit,
            dataset_name=config.dataset_name,
            agent_config=config.agent_config,
            project_root=str(project_root),
            created_at=datetime.now(timezone.utc).isoformat(),
            use_uv=bool(params.get("use_uv", config.use_uv)),
            rerun_from_run_id=safe_run_id(rerun_from_run_id),
            input_cost_per_1m_tokens=_float_param(
                params,
                "input_cost_per_1m_tokens",
                config.input_cost_per_1m_tokens,
                "MODEL_INPUT_COST_PER_1M_TOKENS",
            ),
            output_cost_per_1m_tokens=_float_param(
                params,
                "output_cost_per_1m_tokens",
                config.output_cost_per_1m_tokens,
                "MODEL_OUTPUT_COST_PER_1M_TOKENS",
            ),
        )

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
    input_cost_per_1m_tokens = _float_param(params, "input_cost_per_1m_tokens", 0, "MODEL_INPUT_COST_PER_1M_TOKENS")
    output_cost_per_1m_tokens = _float_param(params, "output_cost_per_1m_tokens", 0, "MODEL_OUTPUT_COST_PER_1M_TOKENS")

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
        use_uv=bool(params.get("use_uv", True)),
        rerun_from_run_id=rerun_from_run_id,
        input_cost_per_1m_tokens=input_cost_per_1m_tokens,
        output_cost_per_1m_tokens=output_cost_per_1m_tokens,
    )


def run_dir(config: RunConfig) -> Path:
    return Path(config.project_root) / "runs" / config.run_id


def prepare_run_dir(config: RunConfig) -> str:
    root = run_dir(config)
    if root.exists():
        raise FileExistsError(f"run directory already exists: {root}")

    (root / "run-agent" / "trajectories").mkdir(parents=True)
    (root / "run-eval" / "logs").mkdir(parents=True)
    (root / "run-eval" / "reports").mkdir(parents=True)
    write_json(root / "config.json", asdict(config))
    return str(root)


def build_agent_command(config: RunConfig, output_dir: Path | None = None, *, use_uv: bool = True) -> list[str]:
    output = output_dir or run_dir(config) / "run-agent" / "trajectories"
    command = []
    if use_uv:
        command.extend(["uv", "run"])
    command.extend(
        [
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
    )
    if config.agent_config:
        command.extend(["--config", config.agent_config])
    if config.cost_limit >= 0:
        if not config.agent_config:
            command.extend(["--config", "swebench.yaml"])
        command.extend(["--config", f"agent.cost_limit={config.cost_limit}"])
    return command


def build_eval_command(config: RunConfig, predictions_path: Path, *, use_uv: bool = True) -> list[str]:
    command = []
    if use_uv:
        command.extend(["uv", "run"])
    command.extend(
        [
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
    )
    return command


def run_agent(config: RunConfig, *, use_uv: bool = True) -> str:
    root = run_dir(config)
    agent_dir = root / "run-agent"
    trajectories_dir = agent_dir / "trajectories"
    env = {
        **os.environ,
        "MSWEA_COST_TRACKING": "ignore_errors",
    }
    subprocess.run(
        build_agent_command(config, trajectories_dir, use_uv=use_uv),
        cwd=config.project_root,
        env=env,
        check=True,
    )

    preds = trajectories_dir / "preds.json"
    if not preds.exists():
        raise FileNotFoundError(f"mini-swe-agent did not produce {preds}")
    shutil.copy2(preds, agent_dir / "preds.json")
    return str(agent_dir / "preds.json")


def run_evaluation(config: RunConfig, predictions_path: str, *, use_uv: bool = True) -> str:
    root = run_dir(config)
    eval_dir = root / "run-eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    with (eval_dir / "run_evaluation.stdout.log").open("w", encoding="utf-8") as stdout:
        subprocess.run(
            build_eval_command(config, Path(predictions_path), use_uv=use_uv),
            cwd=config.project_root,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            check=True,
        )
    _copy_swebench_logs(config, eval_dir)
    _write_eval_reports(eval_dir)
    return str(eval_dir)


def collect_metrics(eval_dir: str | Path, agent_dir: str | Path | None = None) -> dict[str, float | int]:
    reports = _report_paths(Path(eval_dir))
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


def write_manifest(
    config: RunConfig,
    metrics: dict[str, Any],
    artifact_uri: str | None = None,
    mlflow_run_id: str | None = None,
) -> str:
    root = run_dir(config)
    manifest = {
        "run_id": config.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": "config.json",
        "metrics": "metrics.json",
        "artifact_uri": artifact_uri,
        "mlflow_run_id": mlflow_run_id,
        "paths": {
            "run_dir": str(root),
            "config": str(root / "config.json"),
            "agent_dir": str(root / "run-agent"),
            "predictions": str(root / "run-agent" / "preds.json"),
            "trajectories": str(root / "run-agent" / "trajectories"),
            "eval_dir": str(root / "run-eval"),
            "eval_logs": str(root / "run-eval" / "logs"),
            "eval_reports": str(root / "run-eval" / "reports"),
            "metrics": str(root / "metrics.json"),
        },
        "commands": {
            "run_agent": build_agent_command(config, use_uv=config.use_uv),
            "run_eval": build_eval_command(config, root / "run-agent" / "preds.json", use_uv=config.use_uv),
        },
        "metrics_summary": metrics,
    }
    path = root / "manifest.json"
    write_json(path, manifest)
    return str(path)


def summarize_run(config: RunConfig) -> dict[str, Any]:
    root = run_dir(config)
    _write_eval_reports(root / "run-eval")
    metrics = collect_metrics(root / "run-eval", root / "run-agent")
    write_json(root / "metrics.json", metrics)
    artifact_uri = remote_artifact_uri(root)
    manifest_path = write_manifest(config, metrics, artifact_uri)
    uploaded_uri = upload_run_artifacts_if_configured(root)
    mlflow_run_id = log_mlflow_run(config, metrics, str(root), uploaded_uri)
    if mlflow_run_id:
        manifest_path = write_manifest(config, metrics, artifact_uri, mlflow_run_id=mlflow_run_id)
        uploaded_uri = upload_run_artifacts_if_configured(root)
    return {"metrics": metrics, "manifest": manifest_path, "artifact_uri": artifact_uri, "mlflow_run_id": mlflow_run_id}


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
) -> str | None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        return None

    import mlflow

    root = run_dir(config)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "coding-agent-evals"))
    with mlflow.start_run(run_name=config.run_id) as active_run:
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
                "rerun_from_run_id": config.rerun_from_run_id,
                "input_cost_per_1m_tokens": config.input_cost_per_1m_tokens,
                "output_cost_per_1m_tokens": config.output_cost_per_1m_tokens,
            }
        )
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
        mlflow.set_tag("local_artifact_path", local_artifact_path)
        mlflow.set_tag("run_id", config.run_id)
        mlflow.set_tag("model_name", config.model)
        mlflow.set_tag("dataset_name", config.dataset_name)
        mlflow.set_tag("split", config.split)
        mlflow.set_tag("subset", config.subset)
        mlflow.set_tag("task_slice", config.task_slice)
        mlflow.set_tag("airflow_dag_id", "evaluate_agent")
        if artifact_uri:
            mlflow.set_tag("artifact_uri", artifact_uri)
            mlflow.set_tag("s3_artifact_uri", artifact_uri)
        _log_mlflow_artifacts(mlflow, root)
        _log_mlflow_dataset(mlflow, config, metrics)
        _log_mlflow_traces(mlflow, config, root)
        return active_run.info.run_id


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_run_config(config_path: str | Path) -> RunConfig:
    data = read_json(Path(config_path))
    defaults = {
        "use_uv": True,
        "rerun_from_run_id": "",
        "input_cost_per_1m_tokens": 0.0,
        "output_cost_per_1m_tokens": 0.0,
    }
    return RunConfig(**{**defaults, **data})


def cli_run_agent(args: argparse.Namespace) -> None:
    print(run_agent(load_run_config(args.config_path), use_uv=False))


def cli_run_eval(args: argparse.Namespace) -> None:
    print(run_evaluation(load_run_config(args.config_path), args.predictions_path, use_uv=False))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluation pipeline task entrypoints")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_agent_parser = subparsers.add_parser("run-agent")
    run_agent_parser.add_argument("--config-path", required=True)
    run_agent_parser.set_defaults(func=cli_run_agent)

    run_eval_parser = subparsers.add_parser("run-eval")
    run_eval_parser.add_argument("--config-path", required=True)
    run_eval_parser.add_argument("--predictions-path", required=True)
    run_eval_parser.set_defaults(func=cli_run_eval)

    args = parser.parse_args(argv)
    args.func(args)


def _copy_swebench_logs(config: RunConfig, eval_dir: Path) -> None:
    logs_root = Path(config.project_root) / "logs" / "run_evaluation" / config.run_id
    if logs_root.exists():
        destination = eval_dir / "logs"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(logs_root, destination)


def _report_paths(eval_dir: Path) -> list[Path]:
    reports_dir = eval_dir / "reports"
    reports = sorted(reports_dir.glob("*.json")) if reports_dir.exists() else []
    if reports:
        return reports
    logs_dir = eval_dir / "logs"
    return sorted(logs_dir.rglob("report.json")) if logs_dir.exists() else sorted(eval_dir.rglob("report.json"))


def _write_eval_reports(eval_dir: Path) -> None:
    logs_dir = eval_dir / "logs"
    reports_dir = eval_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    if not logs_dir.exists():
        return

    for report_path in sorted(logs_dir.rglob("report.json")):
        try:
            report = read_json(report_path)
        except Exception:
            continue
        for instance_id, instance_report in report.items():
            safe_instance_id = safe_run_id(str(instance_id))
            write_json(reports_dir / f"{safe_instance_id}.report.json", {instance_id: instance_report})


def _collect_cost_metrics(agent_dir: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    provider_cost = 0.0
    cost_count = 0
    prompt_tokens = 0.0
    completion_tokens = 0.0

    for traj_path in agent_dir.rglob("*.traj.json"):
        try:
            data = read_json(traj_path)
        except Exception:
            continue
        costs, prompt, completion = _trajectory_usage(data)
        provider_cost += sum(costs)
        cost_count += len(costs)
        prompt_tokens += prompt
        completion_tokens += completion

    estimated_input_cost = prompt_tokens * metrics_input_price(agent_dir) / 1_000_000
    estimated_output_cost = completion_tokens * metrics_output_price(agent_dir) / 1_000_000
    estimated_total_cost = estimated_input_cost + estimated_output_cost

    metrics["provider_cost_sum"] = provider_cost
    metrics["prompt_tokens"] = prompt_tokens
    metrics["completion_tokens"] = completion_tokens
    metrics["total_tokens"] = prompt_tokens + completion_tokens
    metrics["estimated_input_cost"] = estimated_input_cost
    metrics["estimated_output_cost"] = estimated_output_cost
    metrics["estimated_total_cost"] = estimated_total_cost
    metrics["cost_sum"] = provider_cost if provider_cost > 0 else estimated_total_cost
    if cost_count:
        metrics["cost_observations"] = float(cost_count)
    return metrics


def metrics_input_price(agent_dir: Path) -> float:
    config = _config_for_agent_dir(agent_dir)
    return config.input_cost_per_1m_tokens if config else 0.0


def metrics_output_price(agent_dir: Path) -> float:
    config = _config_for_agent_dir(agent_dir)
    return config.output_cost_per_1m_tokens if config else 0.0


def _config_for_agent_dir(agent_dir: Path) -> RunConfig | None:
    config_path = agent_dir.parent / "config.json"
    if not config_path.exists():
        return None
    try:
        return load_run_config(config_path)
    except Exception:
        return None


def _trajectory_usage(value: Any, path: tuple[str, ...] = ()) -> tuple[list[float], float, float]:
    costs: list[float] = []
    prompt_tokens = 0.0
    completion_tokens = 0.0

    if isinstance(value, dict):
        for key, child in value.items():
            child_path = (*path, str(key))
            if key in {"cost", "total_cost"} and "cost_limit" not in path and isinstance(child, int | float):
                costs.append(float(child))
            if key == "prompt_tokens" and isinstance(child, int | float):
                prompt_tokens += float(child)
            if key == "completion_tokens" and isinstance(child, int | float):
                completion_tokens += float(child)
            child_costs, child_prompt, child_completion = _trajectory_usage(child, child_path)
            costs.extend(child_costs)
            prompt_tokens += child_prompt
            completion_tokens += child_completion
    elif isinstance(value, list):
        for child in value:
            child_costs, child_prompt, child_completion = _trajectory_usage(child, path)
            costs.extend(child_costs)
            prompt_tokens += child_prompt
            completion_tokens += child_completion
    return costs, prompt_tokens, completion_tokens


def _log_mlflow_artifacts(mlflow: Any, root: Path) -> None:
    for path in [
        root / "config.json",
        root / "manifest.json",
        root / "metrics.json",
        root / "run-agent" / "preds.json",
    ]:
        if path.exists():
            mlflow.log_artifact(str(path))
    for path, artifact_path in [
        (root / "run-agent" / "trajectories", "run-agent/trajectories"),
        (root / "run-eval" / "logs", "run-eval/logs"),
        (root / "run-eval" / "reports", "run-eval/reports"),
    ]:
        if path.exists():
            mlflow.log_artifacts(str(path), artifact_path=artifact_path)


def _log_mlflow_dataset(mlflow: Any, config: RunConfig, metrics: dict[str, Any]) -> None:
    try:
        import pandas as pd

        dataset = mlflow.data.from_pandas(
            pd.DataFrame(
                [
                    {
                        "dataset_name": config.dataset_name,
                        "split": config.split,
                        "subset": config.subset,
                        "task_slice": config.task_slice,
                        "total_instances": metrics.get("total_instances", 0),
                    }
                ]
            ),
            source=config.dataset_name,
            name=f"{config.dataset_name}:{config.split}:{config.task_slice}",
        )
        mlflow.log_input(dataset, context="evaluation")
    except Exception as exc:
        mlflow.set_tag("mlflow_dataset_logging_error", str(exc)[:250])


def _log_mlflow_traces(mlflow: Any, config: RunConfig, root: Path) -> None:
    for traj_path in sorted((root / "run-agent" / "trajectories").glob("*/*.traj.json")):
        try:
            trajectory = read_json(traj_path)
            instance_id = str(trajectory.get("instance_id") or traj_path.parent.name)
            messages = trajectory.get("messages") or []
            response = _last_assistant_message(messages)
            mlflow.log_trace(
                name=f"swe-bench:{instance_id}",
                request={
                    "instance_id": instance_id,
                    "dataset_name": config.dataset_name,
                    "model": config.model,
                    "task_slice": config.task_slice,
                },
                response=response,
                intermediate_outputs={
                    "trajectory_path": traj_path.relative_to(root).as_posix(),
                    "message_count": len(messages),
                    "run_id": config.run_id,
                    "instance_id": instance_id,
                    "model": config.model,
                    "dataset_name": config.dataset_name,
                },
                tags={
                    "run_id": config.run_id,
                    "instance_id": instance_id,
                    "model": config.model,
                },
            )
        except Exception as exc:
            mlflow.set_tag(f"mlflow_trace_error_{safe_run_id(traj_path.parent.name)}", str(exc)[:250])


def _last_assistant_message(messages: list[Any]) -> str:
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        role = message.get("role") or message.get("type")
        if role and "assistant" not in str(role).lower():
            continue
        content = message.get("content")
        if isinstance(content, str) and content:
            return content[:4000]
    return ""


if __name__ == "__main__":
    main()
