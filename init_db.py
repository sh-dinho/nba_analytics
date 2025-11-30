# clean_setup.py
import os, shutil

def clean_folder(base="nba-analytics"):
    if os.path.exists(base):
        shutil.rmtree(base)
    os.makedirs(base, exist_ok=True)
    print("✔ Clean folder created:", base)

if __name__ == "__main__":
    clean_folder()