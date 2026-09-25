"""Legacy callback handler for LlamaIndex (BaseCallbackHandler).

Scans queries and LLM inputs for PII and prompt injection using promptfirewall.
LlamaIndex callbacks cannot mutate payloads in-place, so this handler operates
in scan-and-act mode: it logs warnings and optionally blocks requests by raising
exceptions.

Usage::

    from llama_index_callbacks_promptfirewall import PromptFirewallHandler
    from llama_index.core.callbacks import CallbackManager

    handler = PromptFirewallHandler()
    callback_manager = CallbackManager([handler])

    from llama_index.core import Settings
    Settings.callback_manager = callback_manager
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.callbacks.base_handler import BaseCallbackHandler

import promptfirewall

logger = logging.getLogger("promptfirewall.llamaindex")


class PromptInjectionError(ValueError):
    """Raised when a prompt injection attempt is detected and blocking is enabled."""

    def __init__(
        self,
        message: str,
        injection_score: float,
        injection_labels: List[str],
    ) -> None:
        super().__init__(message)
        self.injection_score = injection_score
        self.injection_labels = injection_labels


class PiiDetectedError(ValueError):
    """Raised when PII is detected and blocking mode is enabled."""

    def __init__(
        self,
        message: str,
        pii_findings: List[Dict[str, Any]],
    ) -> None:
        super().__init__(message)
        self.pii_findings = pii_findings


@dataclass
class ScanEvent:
    """Record of a single scan performed by the handler."""

    text: str
    event_type: str
    is_safe: bool
    injection_score: float
    pii_findings: List[Dict[str, Any]] = field(default_factory=list)
    action_taken: str = "passed"
    latency_us: int = 0


class PromptFirewallHandler(BaseCallbackHandler):
    """LlamaIndex callback handler that scans inputs with promptfirewall.

    Intercepts LLM and QUERY events, scanning text for PII and prompt
    injection. Because LlamaIndex callbacks cannot mutate payloads, this
    handler logs findings and optionally raises exceptions to block unsafe
    requests.

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
        logger: Logger instance to use. Defaults to
            ``"promptfirewall.llamaindex"``.
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
        super().__init__(
            event_starts_to_ignore=[],
            event_ends_to_ignore=[],
        )
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

    @property
    def scan_history(self) -> List[ScanEvent]:
        """Return a copy of the scan history."""
        return list(self._scan_history)

    # -- BaseCallbackHandler required methods --------------------------------

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Called when a trace starts. No-op."""

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Called when a trace ends. No-op."""

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Scan text when LLM or QUERY events start."""
        if payload is None:
            return event_id

        texts: List[str] = []

        if event_type == CBEventType.LLM:
            if EventPayload.PROMPT in payload:
                prompt = payload[EventPayload.PROMPT]
                if isinstance(prompt, str) and prompt:
                    texts.append(prompt)
            if EventPayload.MESSAGES in payload:
                messages = payload[EventPayload.MESSAGES]
                if isinstance(messages, list):
                    for msg in messages:
                        content = getattr(msg, "content", None)
                        if isinstance(content, str) and content:
                            texts.append(content)
                        elif isinstance(msg, str) and msg:
                            texts.append(msg)

        elif event_type == CBEventType.QUERY:
            if EventPayload.QUERY_STR in payload:
                query = payload[EventPayload.QUERY_STR]
                if isinstance(query, str) and query:
                    texts.append(query)

        for text in texts:
            self._scan_text(text, event_type.name)

        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Called when an event ends. No-op for scanning."""

    # -- Internal scanning ---------------------------------------------------

    def _scan_text(self, text: str, event_type_name: str) -> None:
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
                self._record(text, event_type_name, result, pii_dicts, action)
                raise PromptInjectionError(
                    f"Prompt injection detected in {event_type_name} "
                    f"(score={result.injection_score:.2f})",
                    injection_score=result.injection_score,
                    injection_labels=list(result.injection_labels),
                )
            else:
                action = "warned_injection"
                self._logger.warning(
                    "Prompt injection detected in %s (score=%.2f) - "
                    "blocking disabled",
                    event_type_name,
                    result.injection_score,
                )

        # -- PII check -------------------------------------------------------
        if self.detect_pii and result.pii_findings:
            types_found = [f["entity_type"] for f in pii_dicts]
            if self.on_pii == "block":
                action = "blocked_pii"
                self._record(text, event_type_name, result, pii_dicts, action)
                raise PiiDetectedError(
                    f"PII detected in {event_type_name}: {types_found}",
                    pii_findings=pii_dicts,
                )
            else:
                if action == "passed":
                    action = "warned_pii"
                self._logger.warning(
                    "PII detected in %s: %d finding(s) %s",
                    event_type_name,
                    len(result.pii_findings),
                    types_found,
                )

        self._record(text, event_type_name, result, pii_dicts, action)

    def _record(
        self,
        text: str,
        event_type_name: str,
        result: Any,
        pii_dicts: List[Dict[str, Any]],
        action: str,
    ) -> None:
        event = ScanEvent(
            text=text,
            event_type=event_type_name,
            is_safe=result.is_safe,
            injection_score=result.injection_score,
            pii_findings=pii_dicts,
            action_taken=action,
            latency_us=result.latency_us,
        )
        self._scan_history.append(event)
        if self.on_scan is not None:
            self.on_scan(event)
