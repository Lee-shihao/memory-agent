"""Debug logging for HTTP API calls. Enable via --debug flag or DEBUG=true env."""

import json
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path


_debug_enabled = False
_debug_file: Path | None = None
_lock = threading.Lock()


def enable(memory_dir: Path) -> None:
    """Enable debug logging to .agent-memory/debug.log."""
    global _debug_enabled, _debug_file
    _debug_enabled = True
    memory_dir.mkdir(parents=True, exist_ok=True)
    _debug_file = memory_dir / "debug.log"


def disable() -> None:
    global _debug_enabled
    _debug_enabled = False


def is_enabled() -> bool:
    return _debug_enabled


def _write(entry: dict) -> None:
    """Append a JSON-line entry to the debug log."""
    if not _debug_enabled or _debug_file is None:
        return
    with _lock:
        try:
            line = json.dumps(entry, ensure_ascii=False, default=str)
            with open(_debug_file, "a") as f:
                f.write(line + "\n")
        except Exception:
            pass


def _sanitize_headers(headers: dict) -> dict:
    """Redact auth tokens from headers."""
    if not headers:
        return {}
    sanitized = dict(headers)
    if "authorization" in sanitized:
        v = sanitized["authorization"]
        if v.startswith("Bearer "):
            sanitized["authorization"] = f"Bearer ...{v[-8:]}"
    if "Authorization" in sanitized:
        v = sanitized["Authorization"]
        if v.startswith("Bearer "):
            sanitized["Authorization"] = f"Bearer ...{v[-8:]}"
    return sanitized


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
    _write({
        "ts": datetime.now(timezone.utc).isoformat(),
        "type": "request",
        "id": request_id,
        "module": module,
        "method": method,
        "url": url,
        "headers": _sanitize_headers(headers or {}),
        "body": body,
    })
    return request_id


def log_response(
    request_id: str,
    status_code: int,
    body: dict | str | None = None,
    error: str | None = None,
) -> None:
    """Log an HTTP response, matched to its request."""
    entry: dict = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "type": "response",
        "id": request_id,
        "status": status_code,
    }
    if body is not None:
        entry["body"] = body
    if error:
        entry["error"] = error
    _write(entry)
