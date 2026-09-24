"""
LangSmith tracing and observability integration.
Provides a resilient, transparent traceable decorator with automatic fallback
when LangSmith is unconfigured or unavailable.
"""
import functools
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("novacore.tracing")

try:
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False

    def traceable(
        run_type: str = "chain",
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Callable:
        """No-op fallback decorator when langsmith is not installed."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args: Any, **func_kwargs: Any) -> Any:
                return func(*args, **func_kwargs)
            return wrapper
        return decorator
