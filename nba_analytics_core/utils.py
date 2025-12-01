# nba_analytics_core/utils.py
import os
import yaml
import logging

def load_yaml_config(path: str = "config.yaml") -> dict:
    try:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f) or {}
            return cfg
    except FileNotFoundError:
        logging.warning(f"config.yaml not found at {path}. Using defaults/env vars.")
        return {}

def get_cfg_value(cfg: dict, keys: list, default=None):
    cur = cfg
    try:
        for k in keys:
            cur = cur[k]
        return cur
    except Exception:
        return default