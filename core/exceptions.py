# core/exceptions.py

class PipelineError(Exception):
    """General pipeline error."""
    pass

class DataError(PipelineError):
    """Raised when there is an issue with input data."""
    pass
