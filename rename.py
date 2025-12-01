import os
import re
import shutil

def rewrite_imports(project_root: str, backup_dir: str = "backup_imports"):
    """
    Rewrite all imports from 'nba_core' to 'nba_analytics_core' in .py files.
    Creates backups before rewriting.
    Skips venv and handles encoding issues gracefully.
    """
    # Ensure backup directory exists
    backup_path = os.path.join(project_root, backup_dir)
    os.makedirs(backup_path, exist_ok=True)

    for root, _, files in os.walk(project_root):
        # Skip virtual environment and hidden folders
        if "venv" in root or "__pycache__" in root:
            continue

        for file in files:
            if not file.endswith(".py"):
                continue

            file_path = os.path.join(root, file)

            # Create backup copy
            rel_path = os.path.relpath(file_path, project_root)
            backup_file_path = os.path.join(backup_path, rel_path)
            os.makedirs(os.path.dirname(backup_file_path), exist_ok=True)
            shutil.copy2(file_path, backup_file_path)

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Fallback for non-UTF8 files
                with open(file_path, "r", encoding="cp1252", errors="ignore") as f:
                    content = f.read()

            # Replace imports
            new_content = re.sub(r'\bfrom nba_core\b', 'from nba_analytics_core', content)
            new_content = re.sub(r'\bimport nba_core\b', 'import nba_analytics_core', new_content)
            new_content = re.sub(r'\bfrom nba_core\.', 'from nba_analytics_core.', new_content)
            new_content = re.sub(r'\bimport nba_core\.', 'import nba_analytics_core.', new_content)

            if new_content != content:
                print(f"Updated imports in {file_path}")
                with open(file_path, "w", encoding="utf-8", errors="ignore") as f:
                    f.write(new_content)

if __name__ == "__main__":
    project_root = r"C:\Users\Mohamadou\projects\nba_analytics"
    rewrite_imports(project_root)