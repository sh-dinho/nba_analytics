# ============================================================
# File: mlflow_setup.py
# Purpose: Configure MLflow tracking for nba_analysis experiments
# Project: nba_analysis
# Version: 1.1 (Improved metadata logging and safety)
# ============================================================

import datetime
import os
import platform
import subprocess
import sys

import mlflow
import pkg_resources


def configure_mlflow(tracking_uri: str = None, experiment_name: str = None):
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


def start_run_with_metadata(run_name: str = None):
    """
    Start an MLflow run with standard reproducibility metadata.

    Adds:
    - Project name
    - Timestamp
    - Git commit hash
    - System info (Python version, OS, machine, processor)
    - Installed package versions
    """
    run_name = run_name or f"nba_analysis_run_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    configure_mlflow()

    # Collect timestamp
    timestamp = datetime.datetime.utcnow().isoformat()

    # Attempt to get git commit
    try:
        git_commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        git_commit = "unknown"

    # Start MLflow run
    run = mlflow.start_run(run_name=run_name)

    # Set tags for reproducibility
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

    # Log installed packages
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
