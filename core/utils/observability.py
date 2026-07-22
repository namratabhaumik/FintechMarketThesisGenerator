"""Langfuse tracing: one shared client + callback handler for the whole app.

"""

import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

_client: Optional[Any] = None
_handler: Optional[Any] = None
_initialized = False


def _init() -> None:
    """Initialize the global Langfuse client + handler once.

    v4's CallbackHandler binds to the global Langfuse client singleton, so the
    client must exist first. Any failure leaves both as None (tracing disabled).
    """
    global _client, _handler, _initialized
    if _initialized:
        return
    _initialized = True

    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        logger.info("Langfuse credentials not set - tracing disabled")
        return

    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler

        # Reads keys + LANGFUSE_BASE_URL from env.
        _client = Langfuse()
        _handler = CallbackHandler()
        logger.info("Langfuse tracing enabled")
    except Exception:
        logger.exception("Langfuse init failed - tracing disabled")
        _client = None
        _handler = None


def get_langfuse() -> Optional[Any]:
    """Return the shared Langfuse client, or None when tracing is disabled."""
    _init()
    return _client


def get_callback_handler() -> Optional[Any]:
    """Return the shared LangChain callback handler, or None when disabled.

    Safe to reuse across concurrent requests: the handler is stateless and
    attaches each generation to whatever trace is active in the ambient context.
    """
    _init()
    return _handler


class _TraceHandle:
    """Lets a route annotate the active trace without importing Langfuse.

    Wraps the root span (or nothing, when tracing is off). ``annotate`` is
    best-effort: a tracing hiccup must never change request behavior.
    """

    def __init__(self, span: Optional[Any] = None):
        self._span = span

    def annotate(self, *, output: Optional[Any] = None, **metadata: Any) -> None:
        """Attach metadata (and optional trace output) to the current trace."""
        if self._span is None:
            return
        try:
            if metadata:
                self._span.update(metadata=metadata)
            if output is not None:
                self._span.set_trace_io(output=output)
        except Exception:
            logger.debug("Langfuse annotate failed", exc_info=True)


@contextmanager
def trace_request(name: str, *, input: Optional[Any] = None) -> Iterator[_TraceHandle]:
    """Open a per-request root span so nested LLM generations group under it.

    Yields a handle for annotating the trace. When tracing is disabled (or the
    span fails to start) it yields a no-op handle; request exceptions always
    propagate unchanged - only the span lifecycle is guarded.
    """
    client = get_langfuse()
    cm = None
    span = None
    if client is not None:
        try:
            cm = client.start_as_current_observation(
                name=name, as_type="span", input=input
            )
            span = cm.__enter__()
        except Exception:
            logger.debug("Langfuse span start failed - continuing untraced", exc_info=True)
            cm = None
            span = None

    try:
        yield _TraceHandle(span)
    finally:
        if cm is not None:
            try:
                cm.__exit__(None, None, None)
            except Exception:
                logger.debug("Langfuse span end failed", exc_info=True)