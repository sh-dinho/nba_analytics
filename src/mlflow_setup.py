# ============================================================
# File: mlflow_setup.py
# Purpose: Configure MLflow tracking for nba_analysis experiments
# Project: nba_analysis
# Version: 1.5 (adds system resource logging)
# ============================================================

import datetime
import json
import os
import platform
import subprocess
import sys
from tempfile import NamedTemporaryFile
from contextlib import contextmanager

import mlflow
import pkg_resources
import psutil  # ✅ new dependency for system metrics


def configure_mlflow(
    tracking_uri: str = None, experiment_name: str = None, strict: bool = True
):
    """
    Configure MLflow tracking URI and experiment.
    Auto-creates experiment if it doesn't exist.
    """
    tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    experiment_name = experiment_name or os.getenv(
        "MLFLOW_EXPERIMENT_NAME", "NBA_Analysis_Experiments"
    )

    try:
        mlflow.set_tracking_uri(tracking_uri)
        experiment = mlflow.get_experiment_by_name(experiment_name)

        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
            print(
                f"✅ Created new MLflow experiment '{experiment_name}' (ID={experiment_id})"
            )
        else:
            mlflow.set_experiment(experiment_name)
            print(
                f"✅ Using existing MLflow experiment '{experiment_name}' (ID={experiment.experiment_id})"
            )

        print(f"✅ MLflow configured with URI={tracking_uri}")
    except Exception as e:
        msg = f"❌ Failed to configure MLflow: {e}"
        if strict:
            print(msg, file=sys.stderr)
            sys.exit(1)
        else:
            print(msg + " — continuing without MLflow logging", file=sys.stderr)


def start_run_with_metadata(run_name: str = None, strict: bool = True):
    """
    Start an MLflow run with reproducibility metadata.
    """
    run_name = (
        run_name
        or f"nba_analysis_run_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    )
    configure_mlflow(strict=strict)

    timestamp = datetime.datetime.utcnow().isoformat()

    # Attempt to get git commit
    try:
        git_commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        git_commit = "unknown"

    run = mlflow.start_run(run_name=run_name)

    # Set tags for reproducibility
    mlflow.set_tags(
        {
            "project": "nba_analysis",
            "timestamp": timestamp,
            "git_commit": git_commit,
            "python_version": platform.python_version(),
            "os": platform.system(),
            "os_version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "run_origin": os.getenv("RUN_ORIGIN", "manual"),
            "schema_version": "2.1",
        }
    )

    # Log installed packages as JSON artifact
    try:
        installed_packages = {
            dist.project_name: dist.version for dist in pkg_resources.working_set
        }
        with NamedTemporaryFile("w", delete=False, suffix=".json") as f:
            json.dump(installed_packages, f, indent=2)
            mlflow.log_artifact(f.name, artifact_path="environment")
    except Exception as e:
        print(f"⚠️ Failed to log package versions: {e}", file=sys.stderr)

    print(f"✅ MLflow run started: {run.info.run_id}")
    return run


def log_system_metrics():
    """
    Log current system resource usage (CPU %, memory) to MLflow.
    """
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        mem = psutil.virtual_memory()
        mlflow.log_metric("cpu_percent", cpu_percent)
        mlflow.log_metric("memory_percent", mem.percent)
        mlflow.log_metric("memory_used_mb", mem.used / (1024 * 1024))
        print(f"📊 Logged system metrics: CPU={cpu_percent}%, MEM={mem.percent}%")
    except Exception as e:
        print(f"⚠️ Failed to log system metrics: {e}", file=sys.stderr)


def end_run_with_cleanup(status: str = "FINISHED"):
    """
    End the current MLflow run safely.
    """
    try:
        mlflow.end_run(status=status)
        print(f"✅ MLflow run ended with status={status}")
    except Exception as e:
        print(f"⚠️ Failed to end MLflow run: {e}", file=sys.stderr)


@contextmanager
def mlflow_run_context(run_name: str = None, strict: bool = True):
    """
    Context manager to ensure MLflow runs are always closed.
    Also logs system metrics at start and end.
    """
    run = None
    try:
        run = start_run_with_metadata(run_name=run_name, strict=strict)
        log_system_metrics()  # log at start
        yield run
        log_system_metrics()  # log at end
        end_run_with_cleanup("FINISHED")
    except Exception as e:
        print(f"❌ Exception during MLflow run: {e}", file=sys.stderr)
        log_system_metrics()  # log crash state
        end_run_with_cleanup("FAILED")
        raise


if __name__ == "__main__":
    configure_mlflow()
