from __future__ import annotations


class UvscemError(Exception):
    """Base class for all uVSCEM domain errors."""


class InstallationWorkflowError(RuntimeError, UvscemError):
    """Raised when extension installation workflow fails."""


class OfflineBundleExportError(ValueError, UvscemError):
    """Raised when offline bundle export fails."""


class OfflineBundleImportMissingFileError(FileNotFoundError, UvscemError):
    """Raised when required offline bundle files are missing."""


class OfflineBundleImportValidationError(ValueError, UvscemError):
    """Raised when offline bundle data validation fails."""


class OfflineModeError(RuntimeError, UvscemError):
    """Raised when strict offline mode blocks network access."""


class OfflineBundleImportExecutionError(RuntimeError, UvscemError):
    """Raised when offline bundle import execution fails."""
