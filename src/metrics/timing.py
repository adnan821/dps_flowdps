"""CUDA-event-based wall-clock timing for per-image inference cost."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

import torch


class CUDATimer:
    """Context manager that measures elapsed GPU time in seconds via CUDA events.

    Falls back to perf_counter when CUDA is unavailable.
    """

    def __init__(self):
        self.elapsed_s: float = 0.0
        self._use_cuda = torch.cuda.is_available()
        self._start_evt = None
        self._end_evt = None
        self._t0 = None

    def __enter__(self) -> "CUDATimer":
        if self._use_cuda:
            self._start_evt = torch.cuda.Event(enable_timing=True)
            self._end_evt = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            self._start_evt.record()
        else:
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self._use_cuda:
            self._end_evt.record()
            torch.cuda.synchronize()
            self.elapsed_s = self._start_evt.elapsed_time(self._end_evt) / 1000.0
        else:
            self.elapsed_s = time.perf_counter() - self._t0
        return False


@contextmanager
def measure_time() -> Iterator[CUDATimer]:
    """Convenience: `with measure_time() as t: ...; print(t.elapsed_s)`."""
    timer = CUDATimer()
    with timer:
        yield timer
