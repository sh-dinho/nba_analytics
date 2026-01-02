from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Config Exports
# File: src/config/__init__.py
# Author: Sadiq
#
# Description:
#     Public export surface for all path constants.
#     Automatically exports all UPPERCASE names from paths.py.
# ============================================================

from .paths import *  # noqa: F401,F403

# Export all uppercase constants (ROOT_DIR, DATA_DIR, etc.)
__all__ = [name for name in globals().keys() if name.isupper()]
