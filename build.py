"""Build the historical training table. Run once, then commit the parquet.

    python build.py --start 2015 --end 2025

This is the only expensive step. After it, daily runs touch the API for just
the current week's schedule, lines, and ratings inputs.
"""
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

import config, dataset
from api import CFBDClient, CFBDError, MissingKeyError
from features import build_features
from ratings import PreseasonPriors, RatingsEngine
from weather import WeatherService

log = logging.getLogger(__name__)


def build_priors(client: CFBDClient, years: list[int],
                 engine_games: pd.DataFrame) -> PreseasonPriors:
    """Preseason priors for each season, using only prior-season information."""
    priors = PreseasonPriors()
    # A throwaway engine to compute each season's final ratings, which feed the
    # next season's prior. Its own priors are empty, which is fine: end-of-season
    # ratings are dominated by the season's games.
    seed_engine = RatingsEngine(engine_games, PreseasonPriors())

    for year in years:
        prev = year - 1
        prior_sp = talent = returning = None
        try:
            prior_sp = client.sp_ratings(prev)
        except CFBDError as exc:
            log.info("SP+ %s unavailable: %s", prev, exc)
        try:
            talent = client.talent(year)
        except CFBDError as exc:
            log.info("talent %s unavailable: %s", year, exc)
        try:
            returning = client.returning_production(year)
        except CFBDError as exc:
            log.info("returning production %s unavailable: %s", year, exc)

        prior_final = None
        if (engine_games["season"] == prev).any():
            prior_final = seed_engine.final_ratings(prev)

        priors.build(year, prior_sp, talent, returning, prior_final)
        log.info("priors %s: %d teams", year, len(priors.by_year.get(year, [])))
    return priors


def build_dataset(start: int, end: int, with_weather: bool = True,
                  api_key: str | None = None) -> pd.DataFrame:
    years = list(range(start, end + 1))
    client = CFBDClient(api_key=api_key)

    log.info("Loading venues...")
    venues = dataset.load_venues(client)
    log.info("  %d venues", len(venues))

    log.info("Loading games %s-%s...", start, end)
    raw_games = dataset.load_games(client, years)
    games = dataset.clean_games(raw_games, venues)
    log.info("  %d games", len(games))
    if games.empty:
        raise SystemExit("No games returned - check the API key and network.")

    log.info("Computing rest and travel...")
    games = dataset.add_rest_and_travel(games)

    log.info("Loading betting lines...")
    lines = dataset.load_lines(client, years)
    games = games.merge(lines, on="game_id", how="left")
    log.info("  %d games with a line", int(games["spread"].notna().sum()))

    log.info("Building preseason priors...")
    priors = build_priors(client, years, games)
    engine = RatingsEngine(games, priors)

    log.info("Building features%s...", " with weather" if with_weather else "")
    weather = WeatherService() if with_weather else None
    completed = games[games["completed"]].copy()
    feat = build_features(completed, engine, weather,
                          with_weather=with_weather, historical=True)

    log.info("Feature table: %d rows x %d cols", *feat.shape)
    return feat


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build the CFB training dataset")
    p.add_argument("--start", type=int, default=config.TRAIN_START_YEAR)
    p.add_argument("--end", type=int, default=config.TRAIN_END_YEAR)
    p.add_argument("--no-weather", action="store_true",
                   help="skip historical weather (much faster, slightly worse)")
    p.add_argument("--out", default=str(config.DATA / "training.parquet"))
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        feat = build_dataset(args.start, args.end, with_weather=not args.no_weather)
    except MissingKeyError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 2

    feat.to_parquet(args.out, index=False)
    print(f"Wrote {args.out}: {len(feat):,} games, "
          f"{feat['season'].min()}-{feat['season'].max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
