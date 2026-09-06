"""Check the live API before trusting a run.

    python selftest.py

Verifies the key works, each endpoint we depend on responds, the fields we
need are actually present, and Open-Meteo is reachable. Run this first after
setup, and any time results look wrong - it turns a silent wrong answer into
a named failure.
"""
from __future__ import annotations

import sys

import pandas as pd

import config, schema
from api import CFBDClient, CFBDError, MissingKeyError
from weather import WeatherService

CHECKS: list[tuple[bool, str, str]] = []


def record(ok: bool, name: str, detail: str = "") -> bool:
    CHECKS.append((ok, name, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def main(argv=None) -> int:
    year = config.TRAIN_END_YEAR
    print(f"CFBD self-test (probe season {year})\n" + "-" * 46)

    try:
        client = CFBDClient(use_cache=False)
        client._require_key()
    except MissingKeyError as exc:
        print(f"  [FAIL] API key — {exc}")
        return 2

    # --- games ---
    try:
        games = client.games(year, week=3)
        record(len(games) > 0, "GET /games", f"{len(games)} games in week 3")
        missing = [f for f in schema.missing_report(games, schema.GAME_FIELDS)
                   if f in schema.REQUIRED_GAME_FIELDS]
        record(not missing, "games have every required field",
               f"missing {missing}" if missing else "all present")
    except CFBDError as exc:
        record(False, "GET /games", str(exc)[:140])
        games = pd.DataFrame()

    # --- venues ---
    try:
        venues = client.venues()
        record(len(venues) > 0, "GET /venues", f"{len(venues)} venues")
        missing = [f for f in schema.missing_report(venues, schema.VENUE_FIELDS)
                   if f in schema.REQUIRED_VENUE_FIELDS]
        record(not missing, "venues have coordinates",
               f"missing {missing}" if missing else "lat/lon present")
    except CFBDError as exc:
        record(False, "GET /venues", str(exc)[:140])
        venues = pd.DataFrame()

    # --- lines ---
    try:
        lines = client.lines(year, week=3)
        record(len(lines) > 0, "GET /lines", f"{len(lines)} games with lines")
    except CFBDError as exc:
        record(False, "GET /lines", str(exc)[:140])

    # --- optional enrichment: failures here degrade, they don't break ---
    for label, fn in (
        ("GET /ratings/sp", lambda: client.sp_ratings(year - 1)),
        ("GET /talent", lambda: client.talent(year)),
        ("GET /player/returning", lambda: client.returning_production(year)),
    ):
        try:
            df = fn()
            record(len(df) > 0, f"{label} (optional)", f"{len(df)} rows")
        except CFBDError as exc:
            record(False, f"{label} (optional)",
                   f"unavailable — the prior still works without it: {str(exc)[:90]}")

    # --- weather ---
    try:
        wx = WeatherService()
        sample = wx.at_kickoff(39.75, -84.19,
                               pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=2))
        ok = sample.get("temp_f") is not None
        record(ok, "Open-Meteo forecast",
               f"{sample.get('temp_f')}°F / {sample.get('wind_mph')} mph")
        hist = wx.at_kickoff(39.75, -84.19,
                             pd.Timestamp(f"{year - 1}-10-15T19:00", tz="UTC"),
                             historical=True)
        record(hist.get("temp_f") is not None, "Open-Meteo archive",
               f"{hist.get('temp_f')}°F")
    except Exception as exc:  # noqa: BLE001 - report anything at all
        record(False, "Open-Meteo", str(exc)[:140])

    # --- quota ---
    usage = client.usage_info()
    if usage:
        print(f"\n  API usage this period: {usage}")

    required_failed = [n for ok, n, _ in CHECKS if not ok and "optional" not in n]
    passed = sum(1 for ok, _, _ in CHECKS if ok)
    print("-" * 46)
    print(f"{passed}/{len(CHECKS)} checks passed; {client.calls_made} API calls used")

    if required_failed:
        print("\nBlocking failures:")
        for name in required_failed:
            print(f"  - {name}")
        return 1
    print("\nReady. Next: python build.py && python train.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
