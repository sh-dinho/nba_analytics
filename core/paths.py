# core/paths.py
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

DATA_DIR = os.path.join(BASE_DIR, "../data")
MODELS_DIR = os.path.join(BASE_DIR, "../models")
RESULTS_DIR = os.path.join(BASE_DIR, "../results")

def ensure_dirs():
    for d in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
        os.makedirs(d, exist_ok=True)
