"""Open-Meteo weather, for both live forecasts and historical training labels.

Two things matter for correctness:

1. **Same source for train and predict.** Historical weather comes from the
   Open-Meteo archive (ERA5 reanalysis) and upcoming weather from their
   forecast model. Different products, same variables on the same scales,
   which is what keeps the trained model meaningful at prediction time.

2. **Kickoff hour, not daily average.** A 40F night game inside a 70F day is a
   different game. We take the hour nearest kickoff, in UTC.

Request strategy
----------------
Open-Meteo's free tier bills by *weight*, not by request count: roughly
variables x hours. Asking one venue for a whole Aug-Jan window is a single
request but a very heavy one, and a decade of venues blows the daily budget
long before it finishes - which shows up as a flood of 429s and silently
missing weather.

So we batch the other way round: **one request per game date, carrying every
venue playing that date as a comma-separated coordinate list, for that single
day**. Roughly 40 dates a season instead of 300 venues, and each request is
about a hundredth of the weight. Four variables rather than six for the same
reason - gusts track wind speed closely and pressure earns nothing.

When weather genuinely can't be fetched, the values are left as NaN rather
than filled with a plausible-looking constant. The gradient booster handles
NaN natively, so a missing reading stays missing instead of teaching the model
that every unfetchable game was a calm 70F afternoon.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import requests

import config

log = logging.getLogger(__name__)

# Four variables, not six. Weight is variables x hours, and gusts are ~1.4x
# wind speed while surface pressure adds nothing to a football score.
HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
]

# Indoor games get fixed conditions - known, not missing.
DOME_CONDITIONS = {
    "temp_f": 70.0,
    "humidity": 50.0,
    "precip_in": 0.0,
    "wind_mph": 0.0,
    "is_dome": 1,
}

# Outdoor game whose weather we could not retrieve. NaN, deliberately.
UNKNOWN_CONDITIONS = {
    "temp_f": np.nan,
    "humidity": np.nan,
    "precip_in": np.nan,
    "wind_mph": np.nan,
    "is_dome": 0,
}

# Free-tier pacing: 600 requests/minute is the documented ceiling.
MIN_SECONDS_BETWEEN_CALLS = 0.12
MAX_LOCATIONS_PER_CALL = 40

# Weather is an enrichment, never a blocker. If the provider is throttling us,
# we take what we can get inside a fixed budget and mark the rest unavailable,
# rather than stalling a prediction run for twenty minutes on retries.
WEATHER_MAX_ATTEMPTS = 3
MAX_BACKOFF_SECONDS = 20
DEFAULT_BUDGET_SECONDS = 900


def _cache_file(kind: str, params: dict) -> "config.Path":
    digest = hashlib.sha256(
        json.dumps(params, sort_keys=True).encode()).hexdigest()[:20]
    return config.CACHE / f"wx_{kind}__{digest}.json"


class _Pacer:
    def __init__(self, gap: float):
        self.gap = gap
        self.last = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self.last
        if elapsed < self.gap:
            time.sleep(self.gap - elapsed)
        self.last = time.monotonic()


_PACER = _Pacer(MIN_SECONDS_BETWEEN_CALLS)


def _fetch(url: str, params: dict, kind: str, ttl: int | None):
    """One request, cached on disk, with backoff that respects Retry-After."""
    cf = _cache_file(kind, params)
    if cf.exists():
        if ttl is None or (time.time() - cf.stat().st_mtime) < ttl:
            try:
                return json.loads(cf.read_text())
            except json.JSONDecodeError:
                cf.unlink(missing_ok=True)

    delay = 3.0
    for attempt in range(WEATHER_MAX_ATTEMPTS):
        _PACER.wait()
        try:
            resp = requests.get(url, params=params, timeout=config.REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            log.warning("weather request failed (%s): %s", kind, exc)
            time.sleep(min(delay, MAX_BACKOFF_SECONDS))
            delay *= 2
            continue

        if resp.status_code == 200:
            body = resp.json()
            cf.write_text(json.dumps(body))
            return body

        if resp.status_code == 429:
            wait = min(float(resp.headers.get("Retry-After", delay)),
                       MAX_BACKOFF_SECONDS)
            log.warning("weather rate-limited, waiting %.0fs (attempt %d/%d)",
                        wait, attempt + 1, WEATHER_MAX_ATTEMPTS)
            time.sleep(wait)
            delay *= 2
            continue

        # 400s are our fault (bad params) and will not improve on retry.
        log.warning("weather HTTP %s (%s): %s",
                    resp.status_code, kind, resp.text[:200])
        return None

    log.warning("weather: giving up on %s after %d attempts",
                kind, WEATHER_MAX_ATTEMPTS)
    return None


def _as_location_list(body) -> list[dict]:
    """Open-Meteo returns an object for one location, a list for several."""
    if body is None:
        return []
    if isinstance(body, list):
        return body
    return [body]


def _index_hourly(block: dict) -> dict[str, dict]:
    """Columnar hourly block -> {'YYYY-MM-DDTHH': {var: value}}."""
    hourly = (block or {}).get("hourly") or {}
    times = hourly.get("time") or []
    out: dict[str, dict] = {}
    for i, t in enumerate(times):
        row = {}
        for var in HOURLY_VARS:
            series = hourly.get(var)
            if series is not None and i < len(series):
                row[var] = series[i]
        out[t[:13]] = row
    return out


def _to_conditions(row: dict | None) -> dict:
    if not row:
        return dict(UNKNOWN_CONDITIONS)

    def val(key):
        v = row.get(key)
        return np.nan if v is None else float(v)

    return {
        "temp_f": val("temperature_2m"),
        "humidity": val("relative_humidity_2m"),
        "precip_in": val("precipitation"),
        "wind_mph": val("wind_speed_10m"),
        "is_dome": 0,
    }


def _key(lat, lon, hour: str) -> tuple:
    return (round(float(lat), 2), round(float(lon), 2), hour)


class WeatherService:
    """Weather keyed by (venue coordinates, kickoff hour).

    Call `prefetch(games)` first for bulk work; `at_kickoff` then reads from
    memory. Used without prefetching it still works, one location at a time.
    """

    def __init__(self, budget_seconds: float = DEFAULT_BUDGET_SECONDS):
        self._hours: dict[tuple, dict] = {}
        self._fetched_dates: set[tuple[str, bool]] = set()
        self.requests_made = 0
        self.dates_failed = 0
        self.budget_seconds = budget_seconds
        self._deadline: float | None = None
        self.gave_up_early = False

    def _out_of_time(self) -> bool:
        if self._deadline is None:
            return False
        if time.monotonic() < self._deadline:
            return False
        if not self.gave_up_early:
            log.warning(
                "weather: %.0fs budget spent after %d requests - the rest of "
                "the slate will be marked unavailable rather than holding up "
                "the run", self.budget_seconds, self.requests_made)
            self.gave_up_early = True
        return True

    # -- bulk ---------------------------------------------------------------
    def prefetch(self, games: pd.DataFrame, historical: bool = True) -> None:
        """Fetch every venue-hour the given games need, batched by date."""
        needed = games.loc[
            games["kickoff"].notna()
            & games["lat"].notna()
            & games["lon"].notna()
            & ~games.get("dome", pd.Series(False, index=games.index)).fillna(False)
        ]
        if needed.empty:
            return

        by_date: dict[str, set] = defaultdict(set)
        for _, g in needed.iterrows():
            ts = pd.to_datetime(g["kickoff"], utc=True)
            # Grouped by UTC day, so the kickoff hour always falls inside the
            # single day we request - a 7pm Eastern kickoff is 00:00 UTC the
            # next day, and lands in that day's bucket.
            by_date[ts.strftime("%Y-%m-%d")].add(
                (round(float(g["lat"]), 2), round(float(g["lon"]), 2)))

        total = len(by_date)
        log.info("weather: %d dates, %d venue-days to fetch",
                 total, sum(len(v) for v in by_date.values()))

        self._deadline = time.monotonic() + self.budget_seconds

        for i, (day, coords) in enumerate(sorted(by_date.items()), 1):
            if self._out_of_time():
                break
            if (day, historical) in self._fetched_dates:
                continue
            coord_list = sorted(coords)
            for chunk_start in range(0, len(coord_list), MAX_LOCATIONS_PER_CALL):
                chunk = coord_list[chunk_start:chunk_start + MAX_LOCATIONS_PER_CALL]
                self._fetch_date(day, chunk, historical)
            self._fetched_dates.add((day, historical))
            if i % 25 == 0 or i == total:
                log.info("weather: %d/%d dates (%d requests, %d failed)",
                         i, total, self.requests_made, self.dates_failed)

    def _fetch_date(self, day: str, coords: list[tuple], historical: bool) -> None:
        lats = ",".join(f"{lat}" for lat, _ in coords)
        lons = ",".join(f"{lon}" for _, lon in coords)

        # Dates are grouped by the kickoff's *UTC* day, so a single day covers
        # every hour we need. Asking for a second day would double the weight
        # for nothing.
        params = {
            "latitude": lats,
            "longitude": lons,
            "hourly": ",".join(HOURLY_VARS),
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "precipitation_unit": "inch",
            "timezone": "UTC",
            "start_date": day,
            "end_date": day,
        }
        url = (config.OPEN_METEO_ARCHIVE if historical
               else config.OPEN_METEO_FORECAST)

        body = _fetch(url, params, "archive" if historical else "forecast",
                      ttl=None if historical else 3600)
        self.requests_made += 1

        blocks = _as_location_list(body)
        if not blocks:
            self.dates_failed += 1
            return

        # Responses come back in the order the coordinates were sent.
        for (lat, lon), block in zip(coords, blocks):
            for hour, row in _index_hourly(block).items():
                self._hours[_key(lat, lon, hour)] = row

    # -- single lookup ------------------------------------------------------
    def at_kickoff(self, lat, lon, kickoff_utc, is_dome: bool = False,
                   historical: bool | None = None) -> dict:
        if is_dome:
            return dict(DOME_CONDITIONS)
        if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
            return dict(UNKNOWN_CONDITIONS)

        ts = pd.to_datetime(kickoff_utc, utc=True, errors="coerce")
        if pd.isna(ts):
            return dict(UNKNOWN_CONDITIONS)

        if historical is None:
            historical = ts < pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=2)

        row = self._lookup(lat, lon, ts)
        if row is None and not self._out_of_time():
            # Not prefetched - fetch this one date on demand.
            day = ts.strftime("%Y-%m-%d")
            key = (day, historical)
            if key not in self._fetched_dates:
                self._fetch_date(day, [(round(float(lat), 2), round(float(lon), 2))],
                                 historical)
                self._fetched_dates.add(key)
            row = self._lookup(lat, lon, ts)

        return _to_conditions(row)

    def _lookup(self, lat, lon, ts: pd.Timestamp) -> dict | None:
        for offset in (0, 1, -1, 2, -2, 3, -3):
            hour = (ts + pd.Timedelta(hours=offset)).strftime("%Y-%m-%dT%H")
            row = self._hours.get(_key(lat, lon, hour))
            if row:
                return row
        return None

    def coverage(self) -> str:
        return (f"{self.requests_made} requests, "
                f"{self.dates_failed} failed, "
                f"{len(self._hours):,} venue-hours cached")


def describe(conditions: dict) -> str:
    """Short human-readable summary for the dashboard."""
    if conditions.get("is_dome"):
        return "Indoors"
    bits = []
    t = conditions.get("temp_f")
    if t is not None and not pd.isna(t):
        bits.append(f"{round(t)}°F")
    w = conditions.get("wind_mph")
    if w is not None and not pd.isna(w):
        bits.append(f"{round(w)} mph wind")
    p = conditions.get("precip_in")
    if p is not None and not pd.isna(p) and p >= 0.02:
        bits.append(f'{p:.2f}" precip')
    return ", ".join(bits) if bits else "Weather unavailable"
