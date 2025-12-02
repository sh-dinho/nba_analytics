# setup_project.py
import os
import shutil
import argparse

KEEP_FILES = {".env", ".gitignore", "README.md", "requirements.txt", ".git"}

FOLDERS = [
    "core",
    "data_ingest",
    "features",
    "modeling",
    "betting",
    "pipelines",
    "cli",
    "app",
    "app/dashboard",
    "results",
]

PLACEHOLDERS = {
    "core/config.py": "# Configurations\n",
    "core/logging.py": "# Logging setup\n",
    "core/utils.py": "# Utility functions\n",
    "core/paths.py": "# Paths and directories\n",
    "core/odds_cache.py": "# Odds caching logic\n",
    "core/data_models.py": "# Data models\n",
    "core/exceptions.py": "# Custom exceptions\n",
    "cli/main.py": "# CLI entry point\n",
    "app/__main__.py": "# App entry point\n",
    "pipelines/base_pipeline.py": "# Base pipeline template\n",
}

def cleanup():
    """Delete everything except KEEP_FILES."""
    for item in os.listdir("."):
        if item in KEEP_FILES:
            continue
        if os.path.isdir(item):
            shutil.rmtree(item)
            print(f"Deleted folder: {item}")
        else:
            os.remove(item)
            print(f"Deleted file: {item}")
    print("✅ Cleanup complete.")

def create_structure():
    """Create folder structure with __init__.py files."""
    for folder in FOLDERS:
        os.makedirs(folder, exist_ok=True)
        init_file = os.path.join(folder, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w") as f:
                f.write("# Package init\n")
        print(f"Created folder and __init__.py: {folder}")
    print("✅ Folder structure created.")

def add_placeholders():
    """Create placeholder files with starter content."""
    for file, content in PLACEHOLDERS.items():
        if not os.path.exists(file):
            with open(file, "w") as f:
                f.write(content)
            print(f"Created placeholder: {file}")
        else:
            print(f"Skipped existing file: {file}")
    print("✅ Placeholders added.")

def main():
    parser = argparse.ArgumentParser(description="Project setup utility")
    parser.add_argument("--cleanup", action="store_true", help="Delete all except KEEP_FILES")
    parser.add_argument("--structure", action="store_true", help="Create folder structure")
    parser.add_argument("--placeholders", action="store_true", help="Add placeholder files")
    parser.add_argument("--init", action="store_true", help="Run full initialization (cleanup + structure + placeholders)")
    args = parser.parse_args()

    if args.cleanup:
        cleanup()
    if args.structure:
        create_structure()
    if args.placeholders:
        add_placeholders()
    if args.init:
        cleanup()
        create_structure()
        add_placeholders()

if __name__ == "__main__":
    main()