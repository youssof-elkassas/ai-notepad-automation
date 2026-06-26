"""Custom exception hierarchy for the automation pipeline."""


class AutomationError(Exception):
    """Base exception for all automation failures."""


class GroundingError(AutomationError):
    """Raised when visual grounding fails."""


class GeminiQuotaError(GroundingError):
    """Raised when Gemini API quota or rate limit is exceeded."""


class LowConfidenceError(GroundingError):
    """Raised when grounding confidence is below threshold."""

    def __init__(self, message: str, reason: str, confidence: float = 0.0) -> None:
        super().__init__(message)
        self.reason = reason
        self.confidence = confidence


class BboxParseError(GroundingError):
    """Raised when model output cannot be parsed into a bounding box."""


class PlannerError(GroundingError):
    """Raised when the planner VLM fails or returns invalid output."""


class VerificationError(AutomationError):
    """Raised when post-action state verification fails."""


class WindowNotFoundError(VerificationError):
    """Raised when expected window does not appear."""


class SaveError(AutomationError):
    """Raised when file save operation fails."""


class ScreenshotError(AutomationError):
    """Raised when screenshot capture fails."""


class ConfigurationError(AutomationError):
    """Raised when configuration is invalid."""


class PlatformError(AutomationError):
    """Raised when running on unsupported platform."""
