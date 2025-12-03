# ============================================================
# File: core/exceptions.py
# Purpose: Custom exceptions for pipeline error handling
# ============================================================

class PipelineError(Exception):
    """Base class for all pipeline-related errors."""

    def __init__(self, message: str = "An error occurred in the pipeline"):
        super().__init__(message)


class DataError(PipelineError):
    """Raised when there is an issue with input or processed data."""

    def __init__(self, message: str = "Invalid or missing data encountered"):
        super().__init__(message)