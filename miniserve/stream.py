"""miniserve.stream — SSE streaming helpers with OpenTelemetry span propagation.

Provides SSE formatting and an async generator wrapper that injects
OTEL trace context into each event's data envelope.
"""
from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Optional

try:
    from opentelemetry import trace
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False


def sse_event(data: Any, event: Optional[str] = None) -> str:
    """Format a single SSE frame."""
    payload = data if isinstance(data, str) else json.dumps(data)
    lines = []
    if event:
        lines.append(f"event: {event}")
    lines.append(f"data: {payload}")
    lines.append("")
    return "\n".join(lines) + "\n"


def sse_done() -> str:
    return "data: [DONE]\n\n"


class SSEStream:
    """Wrap an async generator, injecting OTEL traceparent into the first chunk."""

    def __init__(
        self,
        source: AsyncGenerator[str, None],
        span_context: Any = None,
    ) -> None:
        self._source = source
        self._span_context = span_context
        self._first = True

    def _traceparent(self) -> Optional[str]:
        if not _OTEL_AVAILABLE or self._span_context is None:
            return None
        carrier: dict = {}
        ctx = trace.set_span_in_context(
            trace.NonRecordingSpan(self._span_context)
        )
        TraceContextTextMapPropagator().inject(carrier, context=ctx)
        return carrier.get("traceparent")

    async def __aiter__(self) -> AsyncGenerator[str, None]:
        tp = self._traceparent()
        async for chunk in self._source:
            if self._first and tp:
                # Prepend traceparent comment line (non-standard but useful)
                yield f": traceparent={tp}\n"
                self._first = False
            yield chunk
