import os
import re

def rename_core_package(project_root: str):
    """
    Rename 'core' folder to 'nba_core' and update all imports in .py files.
    Args:
        project_root (str): Path to the root of your project (e.g., 'c:/Users/Mohamadou/projects/nba_analytics')
    """
    core_path = os.path.join(project_root, "core")
    nba_core_path = os.path.join(project_root, "nba_core")

    # Step 1: Rename the folder if it exists
    if os.path.exists(core_path):
        print(f"Renaming {core_path} -> {nba_core_path}")
        os.rename(core_path, nba_core_path)
    else:
        print("No 'core' folder found, skipping rename.")

    # Step 2: Walk through all .py files and update imports
    for root, _, files in os.walk(project_root):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Replace imports like "from nba_core..." or "import nba_core..."
                new_content = re.sub(r'\bfrom core\b', 'from nba_core', content)
                new_content = re.sub(r'\bimport core\b', 'import nba_core', new_content)
                new_content = re.sub(r'\bfrom core\.', 'from nba_core.', new_content)
                new_content = re.sub(r'\bimport core\.', 'import nba_core.', new_content)

                if new_content != content:
                    print(f"Updated imports in {file_path}")
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(new_content)

if __name__ == "__main__":
    project_root = r"c:\Users\Mohamadou\projects\nba_analytics"
    rename_core_package(project_root)