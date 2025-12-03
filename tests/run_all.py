# ============================================================
# File: tests/run_all.py
# Purpose: Run all pipeline unit tests in one go
# ============================================================

import pytest
import sys
import os

def main():
    # Ensure we're in the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    # Run pytest on the tests directory
    print("=== Running all pipeline tests ===")
    sys.exit(pytest.main(["-v", "tests"]))

if __name__ == "__main__":
    main()