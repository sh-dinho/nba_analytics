# Cleanup utility scriptimport argparse
import os
import shutil

PRESERVE = {".gitignore", ".env"}

def wipe(path):
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.isfile(path) and (os.path.basename(path) not in PRESERVE):
        os.remove(path)

def main():
    parser = argparse.ArgumentParser(description="Cleanup artifacts")
    parser.add_argument("--yes", action="store_true", help="Confirm cleanup")
    args = parser.parse_args()

    if not args.yes:
        print("Use --yes to confirm cleanup.")
        return

    for d in ["models", "logs", "artifacts"]:
        wipe(d)
        os.makedirs(d, exist_ok=True)

    for d in ["results"]:
        os.makedirs(d, exist_ok=True)  # keep results folder
    print("✅ Cleanup complete and folders rebuilt.")

if __name__ == "__main__":
    main()