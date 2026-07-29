"""Thread-safe access to python-tika's lazily started local server."""

import threading


_startup_lock = threading.Lock()
_startup_complete = False


def _from_buffer(blob):
    from tika import parser as tika_parser

    return tika_parser.from_buffer(blob)


def from_buffer(blob):
    """Parse a buffer while serializing only the first local Tika startup.

    python-tika starts a Java server lazily on port 9998. Concurrent first calls
    can launch multiple servers and fail with ``Address already in use``. Once
    one parse succeeds, later calls bypass the lock and retain normal Tika
    request concurrency.
    """
    global _startup_complete

    if _startup_complete:
        return _from_buffer(blob)

    with _startup_lock:
        if _startup_complete:
            return _from_buffer(blob)
        parsed = _from_buffer(blob)
        _startup_complete = True
        return parsed


def _reset_tika_startup_state_for_test():
    global _startup_complete

    with _startup_lock:
        _startup_complete = False
