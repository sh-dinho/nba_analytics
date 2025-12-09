# ============================================================
# File: mlflow_setup.py
# Purpose: Configure MLflow tracking for nba_analysis experiments.
# Project: nba_analysis
# ============================================================

import os
import sys
import mlflow
import datetime
import subprocess
import platform
import pkg_resources


def configure_mlflow():
    """Configure MLflow tracking URI and experiment, auto-creating if needed."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "NBA_Analysis_Experiments")

    try:
        mlflow.set_tracking_uri(tracking_uri)

        # Ensure experiment exists
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
            print(f"✅ Created new MLflow experiment '{experiment_name}' (ID={experiment_id})")
        else:
            mlflow.set_experiment(experiment_name)
            print(f"✅ Using existing MLflow experiment '{experiment_name}' (ID={experiment.experiment_id})")

        print(f"✅ MLflow configured with URI={tracking_uri}")
    except Exception as e:
        print(f"❌ Failed to configure MLflow: {e}", file=sys.stderr)
        sys.exit(1)


def start_run_with_metadata(run_name: str = "nba_analysis_run"):
    """
    Start an MLflow run with standard metadata:
    - Project name
    - Timestamp
    - Git commit hash (if available)
    - System environment info (Python version, OS, machine)
    - Installed package versions
    """
    configure_mlflow()

    # Collect metadata
    timestamp = datetime.datetime.utcnow().isoformat()
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        git_commit = "unknown"

    run = mlflow.start_run(run_name=run_name)

    # Tags for reproducibility
    mlflow.set_tags({
        "project": "nba_analysis",
        "timestamp": timestamp,
        "git_commit": git_commit,
        "python_version": platform.python_version(),
        "os": platform.system(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    })

    # Log installed package versions
    try:
        installed_packages = {dist.project_name: dist.version for dist in pkg_resources.working_set}
        for pkg, version in installed_packages.items():
            mlflow.log_param(f"pkg_{pkg}", version)
    except Exception as e:
        print(f"⚠️ Failed to log package versions: {e}", file=sys.stderr)

    print(f"✅ MLflow run started: {run.info.run_id}")
    return run


if __name__ == "__main__":
    configure_mlflow()
