"""Typed errors for news-pipeline."""


class NewsPipelineError(Exception):
    """Base error for news-pipeline."""


class UnsupportedParserKind(NewsPipelineError):
    """Raised when a source kind has no registered parser."""
