"""LlamaIndex integration for promptfirewall.

Provides two handler types:

- :class:`PromptFirewallHandler` -- legacy callback handler
  (``BaseCallbackHandler``) for existing LlamaIndex applications.
- :class:`PromptFirewallEventHandler` -- modern instrumentation handler
  (``BaseEventHandler``) for the new observability API.

Both handlers scan LLM and query inputs for PII and prompt injection
using the ``promptfirewall`` Rust engine, with sub-millisecond latency
and zero network calls.
"""

from ._callback_handler import (
    PiiDetectedError,
    PromptFirewallHandler,
    PromptInjectionError,
    ScanEvent,
)
from ._instrumentation_handler import PromptFirewallEventHandler

__all__ = [
    "PromptFirewallHandler",
    "PromptFirewallEventHandler",
    "PromptInjectionError",
    "PiiDetectedError",
    "ScanEvent",
]

__version__ = "0.1.0"
