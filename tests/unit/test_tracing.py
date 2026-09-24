"""
Unit tests for LangSmith tracing configuration and @traceable decorator fallback.
"""
import os
from unittest.mock import patch

from app.config.settings import Settings
from app.config.tracing import traceable


def test_traceable_decorator_execution():
    """Verify that @traceable wraps functions transparently and returns values correctly."""

    @traceable(name="test_function", run_type="chain")
    def sample_func(x: int, y: int) -> int:
        """Sample docstring."""
        return x + y

    result = sample_func(10, 25)
    assert result == 35
    assert sample_func.__name__ == "sample_func"
    assert sample_func.__doc__ == "Sample docstring."


def test_configure_tracing_with_valid_key():
    """Verify configure_tracing exports environment variables when a non-placeholder key is provided."""
    settings = Settings(
        langchain_tracing_v2=True,
        langchain_api_key="lsv2_pt_test_key_12345",
        langchain_project="test-project",
        langchain_endpoint="https://api.smith.langchain.com",
    )

    with patch.dict(os.environ, {}, clear=False):
        settings.configure_tracing()
        assert os.environ.get("LANGCHAIN_TRACING_V2") == "true"
        assert os.environ.get("LANGCHAIN_API_KEY") == "lsv2_pt_test_key_12345"
        assert os.environ.get("LANGCHAIN_PROJECT") == "test-project"
        assert os.environ.get("LANGCHAIN_ENDPOINT") == "https://api.smith.langchain.com"


def test_configure_tracing_skips_placeholder_key():
    """Verify configure_tracing does not export environment variables for placeholder keys."""
    settings = Settings(
        langchain_tracing_v2=True,
        langchain_api_key="your_langsmith_api_key_here",
    )

    with patch.dict(os.environ, {}, clear=False):
        if "LANGCHAIN_API_KEY" in os.environ:
            del os.environ["LANGCHAIN_API_KEY"]
        settings.configure_tracing()
        assert "LANGCHAIN_API_KEY" not in os.environ
