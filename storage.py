"""Saving and loading the feature table without a hard parquet dependency.

Parquet is the right format here - columnar, compressed, fast - but it needs
pyarrow, which is not installable in every environment. Rather than let a
storage detail break the pipeline, this falls back to gzipped pickle and the
loader accepts either. The caller names a logical path; this decides the rest.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def _parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import fastparquet  # noqa: F401
        return True
    except ImportError:
        return False


def save_table(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a frame, preferring parquet. Returns the path actually written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if _parquet_available():
        target = path.with_suffix(".parquet")
        df.to_parquet(target, index=False)
        return target

    target = path.with_suffix(".pkl.gz")
    df.to_pickle(target, compression="gzip")
    log.info("parquet unavailable; wrote %s instead", target.name)
    return target


def load_table(path: str | Path) -> pd.DataFrame | None:
    """Read whichever format is present, or None if neither is."""
    path = Path(path)
    for candidate in (path.with_suffix(".parquet"), path.with_suffix(".pkl.gz"),
                      path):
        if not candidate.exists():
            continue
        try:
            if candidate.suffix == ".parquet":
                return pd.read_parquet(candidate)
            return pd.read_pickle(candidate, compression="gzip")
        except Exception as exc:  # noqa: BLE001 - try the next candidate
            log.warning("could not read %s: %s", candidate.name, exc)
    return None


def table_exists(path: str | Path) -> bool:
    path = Path(path)
    return (path.with_suffix(".parquet").exists()
            or path.with_suffix(".pkl.gz").exists())
