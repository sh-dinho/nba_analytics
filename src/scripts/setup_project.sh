#!/usr/bin/env bash
# ============================================================
# File: setup_project.sh
# Purpose: Bootstrap NBA AI project structure and hygiene tools.
# ============================================================

set -euo pipefail

echo "🚀 Setting up NBA AI project..."

# --- Check prerequisites ---
command -v python3 >/dev/null 2>&1 || { echo "❌ Python3 not found, please install it"; exit 1; }
command -v pip >/dev/null 2>&1 || { echo "❌ pip not found, please install it"; exit 1; }

# --- Create directories ---
mkdir -p src/prediction_engine src/model_training src/utils src/features tests data results models logs docs

# --- Create placeholder files ---
touch src/prediction_engine/__init__.py \
      src/model_training/__init__.py \
      src/utils/__init__.py \
      src/features/__init__.py \
      tests/__init__.py

# --- Initialize Git repo if not already ---
if [ ! -d ".git" ]; then
  git init
fi

# --- Add hygiene configs ---
cat > .gitignore << 'EOF'
__pycache__/
*.pyc
*.pyo
*.pyd
env/
.venv/
.ENV/
venv/
*.log
logs/
*.sqlite3
*.db
results/
models/
data/
archives/
.ipynb_checkpoints/
*.ipynb
.pytest_cache/
.mypy_cache/
.coverage
coverage.xml
htmlcov/
.DS_Store
Thumbs.db
.vscode/
.idea/
build/
dist/
*.egg-info/
*.tmp
*.bak
*.swp
EOF

cat > .editorconfig << 'EOF'
root = true

[*]
charset = utf-8
end_of_line = lf
insert_final_newline = true
trim_trailing_whitespace = true

[*.py]
indent_style = space
indent_size = 4
max_line_length = 88

[*.md]
indent_style = space
indent_size = 2
trim_trailing_whitespace = false
insert_final_newline = true

[*.{yml,yaml,json}]
indent_style = space
indent_size = 2

[*.toml]
indent_style = space
indent_size = 2

[*.sh]
indent_style = space
indent_size = 2
end_of_line = lf

[Makefile]
indent_style = tab
EOF

# --- Setup virtual environment ---
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate

# --- Install dependencies ---
pip install --upgrade pip setuptools wheel
if [ -f requirements.txt ]; then
  pip install -r requirements.txt
else
  echo "❌ No requirements.txt found. Please ensure dependencies are listed."
  exit 1
fi

# --- Fix common dependency conflicts ---
pip install --upgrade "protobuf>=5.0,<7.0" "packaging>=24.2"

# --- Ensure pinned mlflow version ---
pip uninstall -y mlflow mlflow-skinny || true
pip install mlflow==2.9.2

# --- Initialize pre-commit ---
pip install pre-commit
if ! command -v pre-commit >/dev/null 2>&1; then
  echo "❌ pre-commit installation failed."
  exit 1
fi
pre-commit install --hook-type pre-push --install-hooks

# --- Initialize GitHub Actions workflow ---
mkdir -p .github/workflows
cat > .github/workflows/tests.yml << 'EOF'
name: CI
on: [push, pull_request]
jobs:
  ci:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install --upgrade pip setuptools wheel
      - run: pip install -r requirements.txt || true
      - run: pip install "protobuf>=5.0,<7.0" "packaging>=24.2" mlflow==2.9.2
      - run: make ci
EOF

# --- Compatibility check ---
echo "🔍 Verifying installed versions..."
python3 -c "import mlflow, packaging, google.protobuf as pb; \
print(f'MLflow version: {mlflow.__version__}'); \
print(f'Packaging version: {packaging.__version__}'); \
print(f'Protobuf version: {pb.__version__}')"

echo "✅ NBA AI project setup complete!"
echo "Next steps:"
echo "1. Run 'make precommit' to ensure hooks are installed."
echo "2. Run 'make check' to validate your environment."
echo "3. Start coding 🚀"
