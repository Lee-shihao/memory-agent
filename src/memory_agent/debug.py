"""Debug logging for HTTP API calls. Enable via --debug flag."""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

_debug_enabled = False
_debug_file: Path | None = None
_lock = threading.Lock()

_SEPARATOR = "─" * 70


def enable(memory_dir: Path) -> None:
    """Enable debug logging to .agent-memory/debug.log.

    The log file is cleared at the start of each new conversation.
    """
    global _debug_enabled, _debug_file
    _debug_enabled = True
    memory_dir.mkdir(parents=True, exist_ok=True)
    _debug_file = memory_dir / "debug.log"
    # Truncate on new session
    with open(_debug_file, "w") as f:
        f.write(f"{'═' * 70}\n")
        f.write(f"  Debug session: {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"{'═' * 70}\n\n")


def disable() -> None:
    global _debug_enabled
    _debug_enabled = False


def is_enabled() -> bool:
    return _debug_enabled


def _write_raw(text: str) -> None:
    if not _debug_enabled or _debug_file is None:
        return
    with _lock:
        try:
            with open(_debug_file, "a") as f:
                f.write(text)
        except Exception:
            pass


def _pretty_json(obj) -> str:
    """Format an object as indented JSON, truncating very large strings."""
    if obj is None:
        return "(none)"
    try:
        formatted = json.dumps(obj, ensure_ascii=False, indent=2, default=str)
        # Truncate if excessively large (>16KB)
        if len(formatted) > 16384:
            formatted = formatted[:16384] + "\n... (truncated)"
        return formatted
    except Exception:
        return str(obj)


def _sanitize_headers(headers: dict) -> dict:
    """Redact auth tokens from headers."""
    if not headers:
        return {}
    sanitized = dict(headers)
    for key in ("authorization", "Authorization"):
        if key in sanitized:
            v = sanitized[key]
            if v.startswith("Bearer "):
                sanitized[key] = f"Bearer ...{v[-8:]}"
    return sanitized


# ── session token tracking ────────────────────────────────────────────────────

_session_stats = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "cached_tokens": 0,
    "prompt_cache_hit_tokens": 0,
    "prompt_cache_miss_tokens": 0,
    "llm_call_count": 0,
}


def accumulate_usage(usage: dict) -> None:
    """Accumulate token usage from a single LLM response. Thread-safe."""
    if not usage:
        return
    with _lock:
        _session_stats["prompt_tokens"] += usage.get("prompt_tokens", 0)
        _session_stats["completion_tokens"] += usage.get("completion_tokens", 0)
        _session_stats["total_tokens"] += usage.get("total_tokens", 0)
        details = usage.get("prompt_tokens_details", {})
        if isinstance(details, dict):
            _session_stats["cached_tokens"] += details.get("cached_tokens", 0)
        _session_stats["prompt_cache_hit_tokens"] += usage.get("prompt_cache_hit_tokens", 0)
        _session_stats["prompt_cache_miss_tokens"] += usage.get("prompt_cache_miss_tokens", 0)
        _session_stats["llm_call_count"] += 1


def get_session_stats() -> dict:
    """Return a copy of accumulated session token stats."""
    with _lock:
        return dict(_session_stats)


def reset_session_stats() -> None:
    """Zero all per-conversation token counters."""
    global _session_stats
    with _lock:
        for key in _session_stats:
            _session_stats[key] = 0


# ── public API ────────────────────────────────────────────────────────────────

def log_request(
    module: str,
    method: str,
    url: str,
    headers: dict | None = None,
    body: dict | None = None,
) -> str:
    """Log an outgoing HTTP request. Returns a request_id for matching the response."""
    request_id = datetime.now(timezone.utc).strftime("%H%M%S-%f")[:15]
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    lines = [
        _SEPARATOR,
        f"[{ts}]  REQUEST  {request_id}  module={module}",
        f"{method}  {url}",
    ]
    if headers:
        lines.append(f"Headers: {_pretty_json(_sanitize_headers(headers))}")
    if body:
        lines.append(f"Body:\n{_pretty_json(body)}")
    lines.append("")

    _write_raw("\n".join(lines))
    return request_id


def log_response(
    request_id: str,
    status_code: int,
    body: dict | str | None = None,
    error: str | None = None,
) -> None:
    """Log an HTTP response, matched to its request."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    status_icon = "✓" if 200 <= status_code < 300 else "✗"

    lines = [
        f"[{ts}]  RESPONSE  {request_id}  {status_icon} HTTP {status_code}",
    ]
    if error:
        lines.append(f"ERROR: {error}")
    if body is not None:
        lines.append(f"Body:\n{_pretty_json(body)}")
    lines.append(_SEPARATOR)
    lines.append("")  # blank line between pairs

    _write_raw("\n".join(lines))
