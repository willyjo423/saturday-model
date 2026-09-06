"""Open-Meteo weather, for both live forecasts and historical training labels.

Two things matter for correctness here:

1. **Same source for train and predict.** Historical weather comes from the
   Open-Meteo archive (ERA5 reanalysis) and upcoming weather from their
   forecast model. Different products, but the same variables on the same
   scales, which is what keeps the trained coefficients meaningful.

2. **Kickoff hour, not daily average.** A 40F night game in a 70F day is a
   different game. We interpolate to the hour nearest kickoff in UTC.

Efficiency: the archive is fetched per venue per season across the whole
Aug-Jan window in one call, then indexed by hour. That turns ~8,000 per-game
lookups into ~300 cached calls.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from . import config

log = logging.getLogger(__name__)

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_gusts_10m",
    "surface_pressure",
]

# Neutral indoor conditions substituted for domes and closed-roof stadiums.
DOME_CONDITIONS = {
    "temp_f": 70.0,
    "humidity": 50.0,
    "precip_in": 0.0,
    "wind_mph": 0.0,
    "gust_mph": 0.0,
    "pressure_hpa": 1013.0,
    "is_dome": 1,
}

NULL_CONDITIONS = {**DOME_CONDITIONS, "is_dome": 0}


def _cache_file(kind: str, params: dict) -> "config.Path":
    digest = hashlib.sha256(
        json.dumps(params, sort_keys=True).encode()).hexdigest()[:20]
    return config.CACHE / f"wx_{kind}__{digest}.json"


def _fetch(url: str, params: dict, kind: str, ttl: int | None) -> dict:
    cf = _cache_file(kind, params)
    if cf.exists():
        fresh = ttl is None or (time.time() - cf.stat().st_mtime) < ttl
        if fresh:
            try:
                return json.loads(cf.read_text())
            except json.JSONDecodeError:
                cf.unlink(missing_ok=True)

    delay = 2.0
    for _ in range(config.MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=config.REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            log.warning("weather request failed: %s", exc)
            time.sleep(delay)
            delay *= 2
            continue
        if resp.status_code == 200:
            body = resp.json()
            cf.write_text(json.dumps(body))
            return body
        if resp.status_code == 429:
            time.sleep(delay)
            delay *= 2
            continue
        log.warning("weather HTTP %s: %s", resp.status_code, resp.text[:200])
        break
    return {}


def _index_hourly(body: dict) -> dict[str, dict]:
    """Turn Open-Meteo's columnar hourly block into {iso_hour: {var: value}}."""
    hourly = body.get("hourly") or {}
    times = hourly.get("time") or []
    out: dict[str, dict] = {}
    for i, t in enumerate(times):
        row = {}
        for var in HOURLY_VARS:
            series = hourly.get(var)
            if series is not None and i < len(series):
                row[var] = series[i]
        out[t[:13]] = row  # key on 'YYYY-MM-DDTHH'
    return out


def _to_conditions(row: dict, is_dome: bool) -> dict:
    if is_dome:
        return dict(DOME_CONDITIONS)
    if not row:
        return dict(NULL_CONDITIONS)
    return {
        "temp_f": row.get("temperature_2m"),
        "humidity": row.get("relative_humidity_2m"),
        "precip_in": row.get("precipitation"),
        "wind_mph": row.get("wind_speed_10m"),
        "gust_mph": row.get("wind_gusts_10m"),
        "pressure_hpa": row.get("surface_pressure"),
        "is_dome": 0,
    }


class WeatherService:
    """Fetches and caches weather keyed by (lat, lon, kickoff hour)."""

    def __init__(self):
        self._archive_index: dict[tuple, dict] = {}

    # -- historical ---------------------------------------------------------
    def archive_season(self, lat: float, lon: float, year: int) -> dict[str, dict]:
        key = (round(lat, 3), round(lon, 3), year)
        if key in self._archive_index:
            return self._archive_index[key]

        params = {
            "latitude": round(lat, 3),
            "longitude": round(lon, 3),
            "start_date": f"{year}-08-01",
            "end_date": f"{year + 1}-01-25",
            "hourly": ",".join(HOURLY_VARS),
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "precipitation_unit": "inch",
            "timezone": "UTC",
        }
        body = _fetch(config.OPEN_METEO_ARCHIVE, params, "archive", ttl=None)
        index = _index_hourly(body)
        self._archive_index[key] = index
        return index

    # -- forecast -----------------------------------------------------------
    def forecast_hours(self, lat: float, lon: float) -> dict[str, dict]:
        params = {
            "latitude": round(lat, 3),
            "longitude": round(lon, 3),
            "hourly": ",".join(HOURLY_VARS),
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "precipitation_unit": "inch",
            "timezone": "UTC",
            "forecast_days": 16,
            "past_days": 1,
        }
        body = _fetch(config.OPEN_METEO_FORECAST, params, "forecast", ttl=3600)
        return _index_hourly(body)

    # -- unified lookup -----------------------------------------------------
    def at_kickoff(self, lat, lon, kickoff_utc, is_dome: bool = False,
                   historical: bool | None = None) -> dict:
        if is_dome:
            return dict(DOME_CONDITIONS)
        if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
            return dict(NULL_CONDITIONS)

        ts = pd.to_datetime(kickoff_utc, utc=True, errors="coerce")
        if pd.isna(ts):
            return dict(NULL_CONDITIONS)

        if historical is None:
            historical = ts < pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=2)

        hour_key = ts.strftime("%Y-%m-%dT%H")
        try:
            if historical:
                season = ts.year if ts.month >= 8 else ts.year - 1
                index = self.archive_season(float(lat), float(lon), season)
            else:
                index = self.forecast_hours(float(lat), float(lon))
        except Exception as exc:  # never let weather sink a prediction run
            log.warning("weather lookup failed (%s, %s): %s", lat, lon, exc)
            return dict(NULL_CONDITIONS)

        row = index.get(hour_key)
        if row is None and index:
            # nearest available hour within +/- 3h
            for offset in (1, -1, 2, -2, 3, -3):
                alt = (ts + pd.Timedelta(hours=offset)).strftime("%Y-%m-%dT%H")
                if alt in index:
                    row = index[alt]
                    break
        return _to_conditions(row or {}, is_dome=False)


def describe(conditions: dict) -> str:
    """Short human-readable summary for the dashboard."""
    if conditions.get("is_dome"):
        return "Indoors"
    bits = []
    t = conditions.get("temp_f")
    if t is not None:
        bits.append(f"{round(t)}°F")
    w = conditions.get("wind_mph")
    if w is not None:
        bits.append(f"{round(w)} mph wind")
    p = conditions.get("precip_in") or 0
    if p >= 0.02:
        bits.append(f"{p:.2f}\" precip")
    return ", ".join(bits) if bits else "Unknown"
