# ============================================================
# File: core/exceptions.py
# Purpose: Custom exceptions for pipeline safety
# ============================================================

class PipelineError(Exception):
    pass

class FileError(Exception):
    def __init__(self, message: str, file_path: str | None = None):
        super().__init__(message)
        self.file_path = file_path

class DataError(Exception):
    pass
