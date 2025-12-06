from setuptools import setup, find_packages

setup(
    name="nba_analytics",
    version="0.1.0",
    description="NBA analytics pipeline with models and notifications",
    author="Mohamadou",
    packages=find_packages(),  # automatically finds nba_core, pipelines, etc.
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "joblib",
        "requests",
    ],
    python_requires=">=3.9",
)
