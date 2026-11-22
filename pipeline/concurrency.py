"""
Bhasha Setu — Concurrency & Workload Queue Manager
Prevents server crashes, memory exhaustion, and FFmpeg CPU starvation
when multiple users submit dubbing requests simultaneously.
"""

import os
import time
import logging
import threading
from contextlib import contextmanager
from typing import Optional, Callable

log = logging.getLogger(__name__)

# Maximum simultaneous intensive dubbing pipelines (FFmpeg + TTS + Transcribe)
# Default is 3, configurable via MAX_CONCURRENT_JOBS environment variable
MAX_CONCURRENT_JOBS = max(1, int(os.environ.get("MAX_CONCURRENT_JOBS", "3")))

_SEMAPHORE = threading.BoundedSemaphore(MAX_CONCURRENT_JOBS)
_STATE_LOCK = threading.Lock()
_ACTIVE_JOBS = 0
_QUEUED_JOBS = 0


def get_queue_status() -> dict:
    """Return the current active and queued dubbing workloads."""
    with _STATE_LOCK:
        return {
            "active_jobs": _ACTIVE_JOBS,
            "queued_jobs": _QUEUED_JOBS,
            "max_slots": MAX_CONCURRENT_JOBS,
        }


@contextmanager
def acquire_dubbing_slot(
    job_id: str,
    status_cb: Optional[Callable[[str], None]] = None,
    timeout: float = 600.0,
):
    """
    Context manager to safely acquire a dubbing execution slot.
    If all slots are currently busy, queues the request and informs the user.
    Always releases the slot in finally block.
    """
    global _ACTIVE_JOBS, _QUEUED_JOBS

    acquired = False
    with _STATE_LOCK:
        if _ACTIVE_JOBS >= MAX_CONCURRENT_JOBS:
            _QUEUED_JOBS += 1
            is_queued = True
        else:
            is_queued = False

    try:
        if is_queued:
            msg = f"Server is processing other videos ({_ACTIVE_JOBS}/{MAX_CONCURRENT_JOBS} active). Your job is queued…"
            log.info(f"[{job_id}] {msg}")
            if status_cb:
                status_cb(msg)

        start_time = time.time()
        # Attempt to acquire slot with timeout
        while not acquired:
            acquired = _SEMAPHORE.acquire(blocking=True, timeout=1.0)
            if acquired:
                break
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Job {job_id} timed out waiting for an available server slot after {int(timeout)}s.")
            if is_queued and status_cb:
                waited = int(time.time() - start_time)
                status_cb(f"Waiting for available server slot… ({waited}s)")

        with _STATE_LOCK:
            if is_queued:
                _QUEUED_JOBS = max(0, _QUEUED_JOBS - 1)
            _ACTIVE_JOBS += 1
            log.info(f"[{job_id}] Slot acquired. Active jobs: {_ACTIVE_JOBS}/{MAX_CONCURRENT_JOBS}")

        yield

    finally:
        if acquired:
            with _STATE_LOCK:
                _ACTIVE_JOBS = max(0, _ACTIVE_JOBS - 1)
                log.info(f"[{job_id}] Slot released. Active jobs: {_ACTIVE_JOBS}/{MAX_CONCURRENT_JOBS}")
            _SEMAPHORE.release()
        else:
            with _STATE_LOCK:
                if is_queued:
                    _QUEUED_JOBS = max(0, _QUEUED_JOBS - 1)
