import os

# Define project structure
PROJECT_ROOT = r"C:\Users\Mohamadou\projects\nba_analytics"

structure = {
    "app": {
        "__init__.py": "",
        "app.py": """import streamlit as st
st.set_page_config(page_title="NBA Analytics Dashboard", layout="wide")
st.title("🏀 NBA Analytics Dashboard")
st.sidebar.success("Use the sidebar to navigate pages")
""",
        "predict_pipeline.py": """# Prediction pipeline placeholder
def generate_predictions():
    pass
""",
        "pages": {
            "ai_predictions.py": "# Streamlit page: AI Predictions",
            "bankroll_sim.py": "# Streamlit page: Bankroll Simulation",
            "stats_overview.py": "# Streamlit page: Stats Overview",
        },
    },
    "nba_analytics_core": {
        "__init__.py": "",
        "data.py": "# Functions to fetch NBA stats",
        "odds.py": "# Functions to fetch bookmaker odds",
        "utils.py": "# Utility functions",
    },
    "scripts": {
        "__init__.py": "",
        "train_model.py": "# Script to retrain model",
        "run_cli.py": """import argparse
def main():
    parser = argparse.ArgumentParser(description="NBA Analytics CLI")
    args = parser.parse_args()
    print("CLI running...")
if __name__ == "__main__":
    main()
""",
        "cleanup.py": "# Cleanup utility script",
    },
    "models": {
        # Empty folder for trained models
    },
    "": {  # Root-level files
        "config.py": """import logging
ODDS_API_KEY = "your-odds-api-key-here"
DB_PATH = "data/nba.db"
def configure_logging():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
""",
        "requirements.txt": """streamlit
nba_api
scikit-learn
pandas
numpy
requests
joblib
""",
        ".gitignore": """__pycache__/
*.log
*.tmp
artifacts/
models/*.pkl
.env
.streamlit/
.ipynb_checkpoints/
.pytest_cache/
""",
        "README.md": """# NBA Analytics Project

This project provides:
- Streamlit dashboard for NBA analytics
- CLI for predictions
- Automated model retraining
- Cleanup utilities

## Usage
- Run Streamlit: `streamlit run app/app.py`
- Run CLI: `python scripts/run_cli.py`
- Retrain model: `python scripts/train_model.py`
- Cleanup: `python scripts/cleanup.py`
""",
    },
}

def create_structure(base_path, struct):
    for name, content in struct.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)
        else:
            os.makedirs(base_path, exist_ok=True)
            file_path = os.path.join(base_path, name)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

if __name__ == "__main__":
    create_structure(PROJECT_ROOT, structure)
    print(f"✅ Project structure created at {PROJECT_ROOT}")