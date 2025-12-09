# ============================================================
# File: tasks.py
# Purpose: Python-native task runner for nba_analysis workflows.
# ============================================================

from invoke import task, Collection

@task
def test(c, cov=False):
    cmd = "pytest"
    if cov:
        cmd += " --cov=src --cov-report=term-missing"
    c.run(cmd)

@task
def lint(c):
    c.run("flake8 src tests")

@task
def format(c):
    c.run("black src tests")
    c.run("isort src tests")

@task
def typecheck(c):
    c.run("mypy src")

@task
def security(c):
    c.run("bandit -r src")

@task
def check(c):
    format(c)
    lint(c)
    typecheck(c)
    security(c)
    test(c)

@task
def integration(c):
    c.run("pytest tests/test_integration_pipeline_mlflow.py -v")

@task
def ci(c):
    format(c)
    lint(c)
    typecheck(c)
    security(c)
    test(c, cov=True)
    integration(c)

@task
def train(c):
    c.run("python -m src.model_training.training --model_type all")

@task
def features(c, game_ids="", season=""):
    if game_ids:
        c.run(
            f"python -m src.prediction_engine.game_features --save --game_ids={game_ids} --log_level=DEBUG"
        )
    elif season:
        c.run(
            f"python -m src.prediction_engine.game_features --save --season={season} --log_level=DEBUG"
        )
    else:
        c.run(
            "python -m src.prediction_engine.game_features --save --game_ids=0042300401,0022300649 --log_level=DEBUG"
        )

@task
def mlflow(c):
    c.run("mlflow ui --backend-store-uri file:./mlruns")

@task
def clean(c):
    """Remove temporary files and caches."""
    c.run("rm -rf .pytest_cache .mypy_cache .coverage htmlcov")
    c.run("find . -type d -name '__pycache__' -exec rm -rf {} +")

@task
def docs(c):
    """Build project documentation with Sphinx."""
    c.run("sphinx-build -b html docs build/docs")

# Default namespace
ns = Collection()
ns.add_task(test)
ns.add_task(lint)
ns.add_task(format)
ns.add_task(typecheck)
ns.add_task(security)
ns.add_task(check)
ns.add_task(integration)
ns.add_task(ci, default=True)
ns.add_task(train)
ns.add_task(features)
ns.add_task(mlflow)
ns.add_task(clean)
ns.add_task(docs)
