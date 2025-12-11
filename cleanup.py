import os
import shutil
import logging

# Initialize logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# List of directories to clean up (You can add/remove based on your needs)
directories_to_clean = [
    'docs/',
    'mlruns/',
    'models/',
    'results/',
    'cache/',
    'notebooks/'
]

def check_empty(directory: str) -> bool:
    """Check if a directory is empty."""
    return len(os.listdir(directory)) == 0

def delete_directory(directory: str):
    """Delete a directory after checking its contents."""
    try:
        # Ensure that the directory exists
        if os.path.exists(directory):
            if check_empty(directory):
                logging.info(f"{directory} is empty. Deleting...")
                shutil.rmtree(directory)
                logging.info(f"Deleted {directory}")
            else:
                logging.info(f"{directory} is not empty. Skipping deletion.")
        else:
            logging.warning(f"{directory} does not exist.")
    except Exception as e:
        logging.error(f"Error deleting {directory}: {e}")

def prompt_user_for_cleanup():
    """Prompt the user for confirmation before performing cleanup."""
    print("\n--- Cleanup Warning ---")
    print("This script will delete certain directories if they are not needed anymore.")
    print("Please ensure you have backed up any important data before proceeding.")
    confirmation = input("Do you want to proceed with cleanup? (yes/no): ").lower()
    if confirmation != "yes":
        print("Cleanup aborted.")
        exit()

def clean_up():
    """Main function to clean up the directories."""
    prompt_user_for_cleanup()

    for directory in directories_to_clean:
        delete_directory(directory)

    print("\n--- Cleanup Complete ---")

if __name__ == "__main__":
    clean_up()
