"""MetricsLogger: CSV-based logger, opens and closes file per write."""

from __future__ import annotations

import csv
from pathlib import Path


class MetricsLogger:
    def __init__(self, path: Path, fields: list) -> None:
        self._path = path
        self._fields = fields

    def log(self, row: dict) -> None:
        file_exists = self._path.exists()

        if not file_exists:
            with open(self._path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._fields)
                writer.writeheader()
                writer.writerow({k: row.get(k, "") for k in self._fields})
        else:
            with open(self._path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._fields)
                writer.writerow({k: row.get(k, "") for k in self._fields})

    def close(self) -> None:
        pass
