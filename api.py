"""CollegeFootballData API client.

Design notes
------------
* Every response is cached on disk keyed by (path, params). Historical seasons
  are treated as immutable so a rebuild costs zero API calls.
* CFBD moved from `division` to `classification` between API versions. Rather
  than guess, `_get` retries once with the alternate spelling on a 400.
* All list endpoints return `list[dict]`; callers get pandas frames via the
  typed wrappers below.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

import pandas as pd
import requests

import config

log = logging.getLogger(__name__)

# Param aliases we will transparently retry with if the server rejects a call.
_ALIASES = [("classification", "division"), ("division", "classification")]


class CFBDError(RuntimeError):
    pass


class MissingKeyError(CFBDError):
    pass


def _cache_path(path: str, params: dict) -> "config.Path":
    payload = json.dumps({"p": path, "q": params}, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:20]
    slug = path.strip("/").replace("/", "_") or "root"
    return config.CACHE / f"{slug}__{digest}.json"


def _is_immutable(params: dict) -> bool:
    year = params.get("year") or params.get("season")
    try:
        return int(year) < config.IMMUTABLE_CACHE_BEFORE_YEAR
    except (TypeError, ValueError):
        return False


class CFBDClient:
    def __init__(self, api_key: str | None = None, use_cache: bool = True):
        self.api_key = (api_key or config.CFBD_API_KEY or "").strip()
        self.use_cache = use_cache
        self.session = requests.Session()
        self.calls_made = 0
        if self.api_key:
            self.session.headers.update(
                {
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": "application/json",
                    "User-Agent": "cfb-predictor/1.0",
                }
            )

    # -- core ---------------------------------------------------------------
    def _require_key(self) -> None:
        if not self.api_key:
            raise MissingKeyError(
                "No CFBD API key. Set the CFBD_API_KEY environment variable "
                "(or repo secret). Get a free key at collegefootballdata.com/key"
            )

    def get(self, path: str, **params: Any) -> list[dict]:
        params = {k: v for k, v in params.items() if v is not None}
        cache_file = _cache_path(path, params)

        if self.use_cache and cache_file.exists():
            age = time.time() - cache_file.stat().st_mtime
            if _is_immutable(params) or age < config.CACHE_TTL_SECONDS:
                try:
                    return json.loads(cache_file.read_text())
                except json.JSONDecodeError:
                    cache_file.unlink(missing_ok=True)

        self._require_key()
        data = self._request_with_alias_retry(path, params)

        if self.use_cache:
            cache_file.write_text(json.dumps(data))
        return data

    def _request_with_alias_retry(self, path: str, params: dict) -> list[dict]:
        try:
            return self._request(path, params)
        except CFBDError as first_error:
            if "400" not in str(first_error):
                raise
            for old, new in _ALIASES:
                if old in params:
                    alt = dict(params)
                    alt[new] = alt.pop(old)
                    log.info("Retrying %s with %r -> %r", path, old, new)
                    try:
                        return self._request(path, alt)
                    except CFBDError:
                        continue
            raise first_error

    def _request(self, path: str, params: dict) -> list[dict]:
        url = f"{config.CFBD_BASE}/{path.lstrip('/')}"
        delay = 1.5
        last: Exception | None = None

        for attempt in range(config.MAX_RETRIES):
            try:
                resp = self.session.get(
                    url, params=params, timeout=config.REQUEST_TIMEOUT
                )
                self.calls_made += 1
            except requests.RequestException as exc:
                last = exc
                time.sleep(delay)
                delay *= 2
                continue

            if resp.status_code == 200:
                try:
                    body = resp.json()
                except ValueError as exc:
                    raise CFBDError(f"{path}: non-JSON response") from exc
                if isinstance(body, dict):
                    return [body]
                return body or []

            if resp.status_code in (401, 403):
                raise CFBDError(
                    f"{path}: HTTP {resp.status_code} - the API key was rejected, "
                    "or this endpoint needs a higher CFBD tier. "
                    f"Body: {resp.text[:200]}"
                )
            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", delay))
                log.warning("Rate limited on %s; sleeping %.1fs", path, wait)
                time.sleep(wait)
                delay *= 2
                continue
            if 500 <= resp.status_code < 600:
                last = CFBDError(f"{path}: HTTP {resp.status_code}")
                time.sleep(delay)
                delay *= 2
                continue

            raise CFBDError(f"{path}: HTTP {resp.status_code} {resp.text[:200]}")

        raise CFBDError(f"{path}: exhausted retries ({last})")

    def frame(self, path: str, **params: Any) -> pd.DataFrame:
        return pd.json_normalize(self.get(path, **params))

    # -- typed wrappers -----------------------------------------------------
    def games(self, year: int, week: int | None = None,
              season_type: str = "regular", classification: str = "fbs") -> pd.DataFrame:
        return self.frame("/games", year=year, week=week,
                          seasonType=season_type, classification=classification)

    def all_games(self, year: int, classification: str = "fbs") -> pd.DataFrame:
        """Regular season plus postseason, concatenated."""
        parts = []
        for st in ("regular", "postseason"):
            try:
                df = self.games(year, season_type=st, classification=classification)
                if not df.empty:
                    df["seasonType"] = st
                    parts.append(df)
            except CFBDError as exc:
                log.warning("games %s %s failed: %s", year, st, exc)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def venues(self) -> pd.DataFrame:
        return self.frame("/venues")

    def calendar(self, year: int) -> pd.DataFrame:
        return self.frame("/calendar", year=year)

    def lines(self, year: int, week: int | None = None,
              season_type: str = "regular") -> pd.DataFrame:
        return self.frame("/lines", year=year, week=week, seasonType=season_type)

    def sp_ratings(self, year: int) -> pd.DataFrame:
        return self.frame("/ratings/sp", year=year)

    def elo_ratings(self, year: int, week: int | None = None) -> pd.DataFrame:
        return self.frame("/ratings/elo", year=year, week=week)

    def talent(self, year: int) -> pd.DataFrame:
        return self.frame("/talent", year=year)

    def returning_production(self, year: int) -> pd.DataFrame:
        return self.frame("/player/returning", year=year)

    def advanced_season_stats(self, year: int, start_week: int | None = None,
                              end_week: int | None = None,
                              exclude_garbage_time: bool = True) -> pd.DataFrame:
        return self.frame(
            "/stats/season/advanced", year=year, startWeek=start_week,
            endWeek=end_week, excludeGarbageTime=exclude_garbage_time)

    def ppa_teams(self, year: int, exclude_garbage_time: bool = True) -> pd.DataFrame:
        return self.frame("/ppa/teams", year=year,
                          excludeGarbageTime=exclude_garbage_time)

    def advanced_game_stats(self, year: int, week: int | None = None,
                            season_type: str = "regular") -> pd.DataFrame:
        return self.frame("/stats/game/advanced", year=year, week=week,
                          seasonType=season_type, excludeGarbageTime=True)

    def fbs_teams(self, year: int) -> pd.DataFrame:
        return self.frame("/teams/fbs", year=year)

    def usage_info(self) -> list[dict]:
        """Remaining monthly call budget, if the tier exposes it."""
        try:
            return self.get("/info/usage")
        except CFBDError as exc:
            log.info("usage endpoint unavailable: %s", exc)
            return []
