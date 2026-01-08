"""Project configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    """Standard project paths."""

    project_root: Path
    data_raw: Path
    data_processed: Path
    models: Path
    reports: Path

    @staticmethod
    def from_here() -> "Paths":
        root = Path(__file__).resolve().parents[2]
        return Paths(
            project_root=root,
            data_raw=root / "data" / "raw",
            data_processed=root / "data" / "processed",
            models=root / "models",
            reports=root / "reports",
        )


NASA_NEOWS_BASE_URL = "https://api.nasa.gov/neo/rest/v1"
