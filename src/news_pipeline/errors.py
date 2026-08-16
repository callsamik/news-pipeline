"""Typed errors for news-pipeline."""


class NewsPipelineError(Exception):
    """Base error for news-pipeline."""


class UnsupportedParserKind(NewsPipelineError):
    """Raised when a source kind has no registered parser."""


class MissingCredential(NewsPipelineError):
    """Raised when ``credential_env`` is set but the env var is absent."""
