"""Instrumentation handler for LlamaIndex (BaseEventHandler).

This is the modern replacement for the legacy callback system. It uses
LlamaIndex's instrumentation module to listen for LLM and query events
and scan them with promptfirewall.

Usage::

    from llama_index_callbacks_promptfirewall import PromptFirewallEventHandler
    import llama_index.core.instrumentation as instrument

    handler = PromptFirewallEventHandler()
    dispatcher = instrument.get_dispatcher()
    dispatcher.add_event_handler(handler)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events import BaseEvent

import promptfirewall

from ._callback_handler import (
    PiiDetectedError,
    PromptInjectionError,
    ScanEvent,
)

logger = logging.getLogger("promptfirewall.llamaindex.instrumentation")

# Event class names we respond to. Using strings instead of isinstance()
# keeps the handler resilient to llama-index-core version changes and
# makes unit testing straightforward without mock-patching isinstance.
_CHAT_START = "LLMChatStartEvent"
_COMPLETION_START = "LLMCompletionStartEvent"
_PREDICT_START = "LLMPredictStartEvent"


class PromptFirewallEventHandler(BaseEventHandler):
    """LlamaIndex instrumentation handler that scans inputs with promptfirewall.

    Listens for ``LLMChatStartEvent``, ``LLMCompletionStartEvent``,
    and ``LLMPredictStartEvent`` events, extracts text from them, and
    scans for PII and prompt injection.

    Args:
        detect_pii: Enable PII detection. Default ``True``.
        detect_injection: Enable injection detection. Default ``True``.
        injection_threshold: Score threshold (0.0-1.0) above which a prompt
            is considered an injection attempt. Default ``0.5``.
        on_injection: Action when injection is detected: ``"block"`` raises
            :class:`PromptInjectionError`, ``"warn"`` only logs. Default
            ``"block"``.
        on_pii: Action when PII is detected: ``"block"`` raises
            :class:`PiiDetectedError`, ``"warn"`` only logs. Default
            ``"warn"``.
        pii_types: List of PII types to detect. ``None`` detects all
            supported types.
        on_scan: Optional callback invoked with each :class:`ScanEvent`.
        logger: Logger instance to use.
    """

    def __init__(
        self,
        *,
        detect_pii: bool = True,
        detect_injection: bool = True,
        injection_threshold: float = 0.5,
        on_injection: str = "block",
        on_pii: str = "warn",
        pii_types: Optional[List[str]] = None,
        on_scan: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__()
        if on_injection not in ("block", "warn"):
            raise ValueError(
                f"on_injection must be 'block' or 'warn', got {on_injection!r}"
            )
        if on_pii not in ("block", "warn"):
            raise ValueError(
                f"on_pii must be 'block' or 'warn', got {on_pii!r}"
            )

        self.detect_pii = detect_pii
        self.detect_injection = detect_injection
        self.injection_threshold = injection_threshold
        self.on_injection = on_injection
        self.on_pii = on_pii
        self.pii_types = pii_types
        self.on_scan = on_scan
        self._logger = logger or globals()["logger"]
        self._scan_history: List[ScanEvent] = []

    @classmethod
    def class_name(cls) -> str:
        """Return the class name for LlamaIndex registration."""
        return "PromptFirewallEventHandler"

    @property
    def scan_history(self) -> List[ScanEvent]:
        """Return a copy of the scan history."""
        return list(self._scan_history)

    def handle(self, event: BaseEvent, **kwargs: Any) -> None:
        """Handle an instrumentation event.

        Extracts text from LLM-related start events and scans it.
        Uses string-based class name matching for resilience across
        llama-index-core versions.
        """
        texts: List[str] = []
        event_name = type(event).__name__

        if event_name == _CHAT_START:
            for msg in event.messages:
                content = getattr(msg, "content", None)
                if isinstance(content, str) and content:
                    texts.append(content)

        elif event_name == _COMPLETION_START:
            if event.prompt:
                texts.append(event.prompt)

        elif event_name == _PREDICT_START:
            template = event.template
            if template is not None:
                template_str = getattr(template, "template", None)
                if isinstance(template_str, str) and template_str:
                    texts.append(template_str)
                if event.template_args:
                    for val in event.template_args.values():
                        if isinstance(val, str) and val:
                            texts.append(val)

        for text in texts:
            self._scan_text(text, event_name)

    def _scan_text(self, text: str, event_name: str) -> None:
        """Run promptfirewall.scan on *text* and act on findings."""
        result = promptfirewall.scan(
            text,
            detect_pii=self.detect_pii,
            detect_injection=self.detect_injection,
            pii_types=self.pii_types,
            injection_threshold=self.injection_threshold,
        )

        pii_dicts = [f.to_dict() for f in result.pii_findings]
        action = "passed"

        # -- Injection check -------------------------------------------------
        if (
            self.detect_injection
            and result.injection_score >= self.injection_threshold
        ):
            if self.on_injection == "block":
                action = "blocked_injection"
                self._record(text, event_name, result, pii_dicts, action)
                raise PromptInjectionError(
                    f"Prompt injection detected in {event_name} "
                    f"(score={result.injection_score:.2f})",
                    injection_score=result.injection_score,
                    injection_labels=list(result.injection_labels),
                )
            else:
                action = "warned_injection"
                self._logger.warning(
                    "Prompt injection detected in %s (score=%.2f) - "
                    "blocking disabled",
                    event_name,
                    result.injection_score,
                )

        # -- PII check -------------------------------------------------------
        if self.detect_pii and result.pii_findings:
            types_found = [f["entity_type"] for f in pii_dicts]
            if self.on_pii == "block":
                action = "blocked_pii"
                self._record(text, event_name, result, pii_dicts, action)
                raise PiiDetectedError(
                    f"PII detected in {event_name}: {types_found}",
                    pii_findings=pii_dicts,
                )
            else:
                if action == "passed":
                    action = "warned_pii"
                self._logger.warning(
                    "PII detected in %s: %d finding(s) %s",
                    event_name,
                    len(result.pii_findings),
                    types_found,
                )

        self._record(text, event_name, result, pii_dicts, action)

    def _record(
        self,
        text: str,
        event_name: str,
        result: Any,
        pii_dicts: List[Dict[str, Any]],
        action: str,
    ) -> None:
        event = ScanEvent(
            text=text,
            event_type=event_name,
            is_safe=result.is_safe,
            injection_score=result.injection_score,
            pii_findings=pii_dicts,
            action_taken=action,
            latency_us=result.latency_us,
        )
        self._scan_history.append(event)
        if self.on_scan is not None:
            self.on_scan(event)
