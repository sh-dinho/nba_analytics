import os
import shutil
import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def find_project_root():
    """Finds the root directory (one level up from 'scripts')."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def cleanup_project(root_dir):
    """
    Removes temporary files, caches, and generated artifacts from the project.
    """
    logging.info(f"Starting cleanup in project root: {root_dir}")
    
    items_to_remove = [
        # Caches and temporary directories
        "__pycache__", 
        ".pytest_cache",
        ".ipynb_checkpoints",
        
        # Streamlit-related caches
        ".streamlit", 
        
        # Generated model artifacts
        "artifacts", 
    ]
    
    # 1. Traverse and remove directories
    removed_count = 0
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        for dirname in list(dirnames):
            if dirname in items_to_remove:
                full_path = os.path.join(dirpath, dirname)
                try:
                    shutil.rmtree(full_path)
                    logging.info(f"Removed directory: {full_path}")
                    dirnames.remove(dirname) # Optimize by not traversing removed dir
                    removed_count += 1
                except OSError as e:
                    logging.error(f"Error removing directory {full_path}: {e}")

    # 2. Handle files (like old artifacts if not inside 'artifacts' dir)
    # The 'artifacts' directory removal in step 1 covers the .pkl file, 
    # but we can add a check for specific files if necessary.
    
    logging.info(f"Cleanup complete. Removed {removed_count} directories.")
    
    # Optional: Log if a critical file exists (like .env) but will not be removed
    if os.path.exists(os.path.join(root_dir, ".env")):
        logging.info("Note: .env file found and preserved (as requested).")

if __name__ == "__main__":
    root = find_project_root()
    cleanup_project(root)