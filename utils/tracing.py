"""JSONL tracer for node-level events.

The tracer is intentionally minimal: it appends one JSON object per call to
the trace file. Each entry is timestamped and gets a monotonically increasing
sequence number so that downstream analysis can recover total event order
even if multiple worker processes are writing (we serialize via fcntl on
POSIX -- best effort; a production setup might prefer SQLite).
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo


UK_TZ = ZoneInfo("Europe/London")


def _json_default(obj: Any) -> Any:
    """Best-effort JSON fallback for objects we don't normally serialize."""
    # ``set`` / ``frozenset`` / ``tuple`` are very common in metrics traces.
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    if hasattr(obj, "tolist"):  # numpy scalars / arrays
        return obj.tolist()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


class JsonlTracer:
    """Append-only JSONL writer with timestamps and sequence numbers.

    Thread-safe within a single process. Designed for write-once-read-many
    workloads: the file is opened lazily on the first ``write``.

    Examples
    --------
    >>> import tempfile, os
    >>> with tempfile.TemporaryDirectory() as d:
    ...     t = JsonlTracer(os.path.join(d, "trace.jsonl"))
    ...     t.write({"event": "ping"})
    ...     t.close()
    """

    def __init__(self, path: str | os.PathLike) -> None:
        """Create a tracer that will append events to ``path``.

        The file is created lazily, so constructing a tracer is cheap.
        """
        self._path = Path(path)
        self._lock = threading.Lock()
        self._fp = None  # type: ignore[assignment]
        self._seq = 0

    def _ensure_open(self) -> None:
        """Open the underlying file handle lazily."""
        if self._fp is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # Line-buffered append; UTF-8.
            self._fp = open(self._path, "a", buffering=1, encoding="utf-8")

    def write(self, event: Mapping[str, Any]) -> None:
        """Append a single event as one JSON line.

        The original mapping is *not* mutated; we copy and add ``ts`` and
        ``seq`` keys before serialization. Non-JSON-serializable values fall
        back to ``str(obj)``.
        """
        with self._lock:
            self._ensure_open()
            self._seq += 1
            from core.state import RUNTIME_ONLY_KEYS  # local import avoids cycle
            payload = dict(event)
            payload.setdefault("ts", datetime.now(UK_TZ).isoformat())
            payload.setdefault("seq", self._seq)
            # Drop runtime-only channels (RNGs, Task object, ...) and any
            # caller-introduced underscore-prefixed keys.
            payload = {
                k: v for k, v in payload.items()
                if not k.startswith("_") and k not in RUNTIME_ONLY_KEYS
            }
            self._fp.write(json.dumps(payload, default=_json_default, ensure_ascii=False))
            self._fp.write("\n")

    def write_many(self, events: list[Mapping[str, Any]]) -> None:
        """Append several events in one critical section."""
        for ev in events:
            self.write(ev)

    def close(self) -> None:
        """Close the underlying file handle if it was ever opened."""
        with self._lock:
            if self._fp is not None:
                self._fp.close()
                self._fp = None

    def __enter__(self) -> "JsonlTracer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
