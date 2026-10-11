"""VoiceLab exception hierarchy."""


class VoiceLabError(Exception):
    """Base exception for all VoiceLab errors."""


class DatasetUnavailableError(VoiceLabError):
    """Raised when the dataset archive is missing or incomplete."""


class ModelLoadError(VoiceLabError):
    """Raised when a model cannot be downloaded or loaded."""


class InferenceError(VoiceLabError):
    """Raised when too many samples fail during inference."""


class RunNotFoundError(VoiceLabError):
    """Raised when a requested MLflow run does not exist."""
