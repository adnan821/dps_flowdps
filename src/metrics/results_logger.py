"""Incremental, resume-safe CSV logger for experiment runs.

Designed for the Colab 24-hour session cap: every row is appended and fsync'd
so an interrupted run can be resumed without losing prior measurements.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Iterable


REQUIRED_FIELDS = (
    "method",          # "pixel_dps" | "latent_dps" | "flowdps" | classical baselines
    "blur_type",       # "gaussian" | "motion"
    "sigma_blur",
    "sigma_noise",
    "motion_length",   # nullable for gaussian
    "motion_angle",    # nullable for gaussian
    "nfe",
    "zeta",
    "seed",
    "image_id",
    "psnr",
    "ssim",
    "lpips",
    "time_s",
    "resolution",      # 256 | 768
    "notes",
)


class ResultsLogger:
    """Append-only CSV logger with a fixed schema."""

    def __init__(self, csv_path: str | os.PathLike, fields: Iterable[str] = REQUIRED_FIELDS):
        self.path = Path(csv_path)
        self.fields = tuple(fields)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.path.exists() or self.path.stat().st_size == 0
        self._fh = self.path.open("a", newline="", buffering=1)  # line-buffered
        self._writer = csv.DictWriter(self._fh, fieldnames=self.fields)
        if is_new:
            self._writer.writeheader()
            self._fh.flush()

    def log(self, row: dict[str, Any]) -> None:
        # Fill missing keys with empty string for schema stability.
        full = {k: row.get(k, "") for k in self.fields}
        self._writer.writerow(full)
        self._fh.flush()
        try:
            os.fsync(self._fh.fileno())
        except OSError:
            pass  # fsync not supported on all filesystems (e.g., Colab Drive)

    def done_keys(self, key_fields: Iterable[str]) -> set[tuple]:
        """Return the set of (key_field tuples) already present in the CSV — for resume logic."""
        keys: set[tuple] = set()
        if not self.path.exists():
            return keys
        with self.path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                keys.add(tuple(row[k] for k in key_fields))
        return keys

    def close(self) -> None:
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass

    def __enter__(self) -> "ResultsLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
