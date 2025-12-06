# ============================================================
# File: core/exceptions.py
# Purpose: Custom exceptions for NBA analytics pipeline error handling
# ============================================================

import logging

class PipelineError(Exception):
    """Base class for all pipeline-related errors."""
    def __init__(self, message: str = "An error occurred in the pipeline"):
        super().__init__(message)
        logging.error(f"{self.__class__.__name__}: {message}")

class ConfigError(PipelineError):
    """Configuration issue."""
    def __init__(self, message: str = "Invalid or missing configuration"):
        super().__init__(message)

class DataError(PipelineError):
    """Input or processed data issue."""
    def __init__(self, message: str = "Invalid or missing data encountered", dataset: str = None):
        super().__init__(message)
        self.dataset = dataset

class ModelError(PipelineError):
    """Model training or inference issue."""
    def __init__(self, message: str = "Model training or inference failed", model_name: str = None):
        super().__init__(message)
        self.model_name = model_name

class APIError(PipelineError):
    """External API call failure."""
    def __init__(self, message: str = "External API request failed", endpoint: str = None):
        super().__init__(message)
        self.endpoint = endpoint

class FileError(PipelineError):
    """File operation failure (read/write/permissions)."""
    def __init__(self, message: str = "File operation failed", file_path: str = None):
        super().__init__(message)
        self.file_path = file_path

class DataError(Exception):
    """Custom exception for data-related issues."""
    pass