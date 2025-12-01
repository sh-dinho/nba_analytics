# Utility functionsimport logging

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        logging.warning(f"Could not convert to float: {x}")
        return default