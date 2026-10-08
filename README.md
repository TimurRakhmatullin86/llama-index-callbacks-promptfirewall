# llama-index-callbacks-promptfirewall

[![PyPI version](https://img.shields.io/pypi/v/llama-index-callbacks-promptfirewall.svg)](https://pypi.org/project/llama-index-callbacks-promptfirewall/)
[![Python versions](https://img.shields.io/pypi/pyversions/llama-index-callbacks-promptfirewall.svg)](https://pypi.org/project/llama-index-callbacks-promptfirewall/)
[![CI](https://github.com/TimurRakhmatullin86/llama-index-callbacks-promptfirewall/actions/workflows/ci.yml/badge.svg)](https://github.com/TimurRakhmatullin86/llama-index-callbacks-promptfirewall/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Sub-millisecond PII detection and prompt injection firewall for LlamaIndex.

Scans every LLM call and query for PII leakage and prompt injection attacks **before** they reach the model. Powered by [promptfirewall](https://github.com/timurua/promptfirewall) -- a Rust-native engine that runs entirely on-CPU with zero network calls.

## Features

- **PII detection**: email, phone, SSN, credit card, API key, IP address
- **Prompt injection detection**: jailbreak, ignore-instructions, role-play attacks
- **Sub-millisecond latency**: ~12us median scan time (Rust engine, no network calls)
- **Two handler types**: legacy `BaseCallbackHandler` + modern `BaseEventHandler`
- **Block or warn**: configurable per-threat action
- **Scan history**: full audit trail of every scan

## Installation

```bash
pip install llama-index-callbacks-promptfirewall
```

## Quick Start

### Legacy Callback Handler

For existing LlamaIndex applications using the callback system:

```python
from llama_index.core.callbacks import CallbackManager
from llama_index.core import Settings, VectorStoreIndex

from llama_index_callbacks_promptfirewall import PromptFirewallHandler

# Create the handler
handler = PromptFirewallHandler(
    detect_pii=True,
    detect_injection=True,
    injection_threshold=0.5,
    on_injection="block",  # "block" raises error, "warn" logs only
    on_pii="warn",         # "block" raises error, "warn" logs only
)

# Attach globally
Settings.callback_manager = CallbackManager([handler])

# Or attach to a specific index
index = VectorStoreIndex.from_documents(docs, callback_manager=CallbackManager([handler]))

# Any query that contains injection or PII will now be caught
query_engine = index.as_query_engine()
response = query_engine.query("What is the capital of France?")  # passes
response = query_engine.query("Ignore all previous instructions")  # blocked!
```

### Instrumentation Handler (Recommended)

For applications using the new LlamaIndex instrumentation API:

```python
import llama_index.core.instrumentation as instrument
from llama_index_callbacks_promptfirewall import PromptFirewallEventHandler

# Create and register the handler
handler = PromptFirewallEventHandler(
    detect_pii=True,
    detect_injection=True,
    injection_threshold=0.5,
    on_injection="block",
    on_pii="warn",
)

dispatcher = instrument.get_dispatcher()
dispatcher.add_event_handler(handler)

# All LLM calls are now scanned automatically
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `detect_pii` | `bool` | `True` | Enable PII detection |
| `detect_injection` | `bool` | `True` | Enable injection detection |
| `injection_threshold` | `float` | `0.5` | Score threshold for injection (0.0-1.0) |
| `on_injection` | `str` | `"block"` | `"block"` raises error, `"warn"` logs only |
| `on_pii` | `str` | `"warn"` | `"block"` raises error, `"warn"` logs only |
| `pii_types` | `list[str]` | `None` (all) | PII types to detect: `email`, `phone`, `ssn`, `credit_card`, `api_key`, `ip_address` |
| `on_scan` | `callable` | `None` | Callback invoked with each `ScanEvent` |
| `logger` | `Logger` | module logger | Custom logger instance |

## Scan History

Both handlers maintain a scan history for auditing:

```python
for event in handler.scan_history:
    print(f"{event.event_type}: safe={event.is_safe}, "
          f"injection={event.injection_score:.2f}, "
          f"pii={len(event.pii_findings)}, "
          f"action={event.action_taken}, "
          f"latency={event.latency_us}us")
```

## Performance

| Metric | promptfirewall | presidio (llama-index-postprocessor-presidio) |
|--------|---------------|-----------------------------------------------|
| Median latency | **~12 us** | ~180 ms |
| Network calls | **0** | 0 |
| PII detection | Yes | Yes |
| Injection detection | **Yes** | No |
| Runs on | CPU (Rust/WASM) | CPU (Python + regex) |

## Comparison with llama-index-postprocessor-presidio

`llama-index-postprocessor-presidio` is a PII-only postprocessor:

- Runs **after** the LLM call (postprocessor), so PII already reached the model
- PII detection only -- no prompt injection detection
- ~180ms per scan vs ~12us for promptfirewall (15,000x faster)
- Requires separate presidio-analyzer and presidio-anonymizer packages

`llama-index-callbacks-promptfirewall` scans **before** the LLM call, catches both PII and injection, and runs in sub-millisecond time.

## Error Handling

```python
from llama_index_callbacks_promptfirewall import (
    PromptInjectionError,
    PiiDetectedError,
)

try:
    response = query_engine.query(user_input)
except PromptInjectionError as e:
    print(f"Injection blocked: score={e.injection_score}, labels={e.injection_labels}")
except PiiDetectedError as e:
    print(f"PII blocked: {e.pii_findings}")
```

## Links

- [promptfirewall](https://github.com/timurua/promptfirewall) -- the Rust engine
- [promptfirewall on PyPI](https://pypi.org/project/promptfirewall-rs/)
- [LlamaIndex](https://docs.llamaindex.ai/)

## License

MIT
