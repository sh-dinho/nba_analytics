# ============================================================
# File: tasks.py
# Purpose: Invoke task definitions for NBA AI project
# Project: nba_analysis
# Version: 1.1 (adds default pipeline task chaining)
#
# Dependencies:
# - invoke
# ============================================================

from invoke import task


@task
def check(c):
    """Run local validation pipeline (tests + lint)."""
    c.run("pytest -q")
    c.run("flake8 src")


@task
def ci(c):
    """Run CI pipeline (tests + lint + type checks)."""
    c.run("pytest --maxfail=1 --disable-warnings -q")
    c.run("flake8 src")
    c.run("mypy src")


@task
def clean(c):
    """Clean caches and temp files."""
    c.run("rm -rf .pytest_cache build dist *.egg-info")
    c.run("find . -name '__pycache__' -exec rm -rf {} +")


@task
def docs(c):
    """Build documentation."""
    c.run("sphinx-build -b html docs build/docs")


@task
def serve_docs(c):
    """Serve documentation locally."""
    c.run("python -m http.server --directory build/docs 8000")


@task
def train(c):
    """Train model."""
    c.run("python src/model_training/train_combined.py")


@task
def features(c):
    """Generate features."""
    c.run("python src/features/feature_engineering.py")


@task
def mlflow(c):
    """Launch MLflow UI."""
    c.run("mlflow ui --backend-store-uri mlruns")


@task
def run(c, date="today"):
    """Run pipeline script for predictions."""
    c.run(f"python src/run_pipeline.py --date {date}")


# --- Default pipeline task chaining ---
@task(pre=[clean, check, features, train, run])
def pipeline(c):
    """
    Run the full end-to-end pipeline:
    clean -> check -> features -> train -> run
    """
    print("✅ Full pipeline completed successfully.")
