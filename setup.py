from setuptools import setup, find_packages

setup(
    name="nba_analytics",
    version="0.1.0",
    description="NBA prediction and analytics pipeline",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "mlflow",
        "pydantic",
        "pydantic-settings",
        "pyyaml",
    ],
    python_requires=">=3.9",
)
