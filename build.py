"""Build the historical training table. Run once, then commit the parquet.

    python build.py --start 2015 --end 2025

This is the only expensive step. After it, daily runs touch the API for just
the current week's schedule, lines, and ratings inputs.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

import pandas as pd

import config, dataset
from api import CFBDClient, CFBDError, MissingKeyError
from comps import CompsEngine, attach_comps, learn_axis_weights
from efficiency import EfficiencyEngine, load_team_game_stats, pool_non_fbs
from features import SeasonContext, build_features, training_matrix
from ratings import PreseasonPriors, RatingsEngine
from storage import save_table
from weather import WeatherService

log = logging.getLogger(__name__)


def build_priors(client: CFBDClient, years: list[int],
                 engine_games: pd.DataFrame,
                 context: SeasonContext | None = None) -> PreseasonPriors:
    """Preseason priors for each season, using only prior-season information.

    Also fills `context` with the same preseason inputs (recruiting talent and
    returning production) so they can be used directly as features rather than
    only as a rating prior.
    """
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

        if context is not None:
            context.add_season(year, talent, returning)

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
    context = SeasonContext()
    priors = build_priors(client, years, games, context=context)
    engine = RatingsEngine(games, priors)

    log.info("Loading advanced play-level stats...")
    fbs_teams = set(games.loc[games["home_is_fbs"], "home_team"]) | \
                set(games.loc[games["away_is_fbs"], "away_team"])
    team_games = load_team_game_stats(client, years)
    team_games = pool_non_fbs(team_games, fbs_teams)
    efficiency = EfficiencyEngine(team_games)

    log.info("Building features%s...", " with weather" if with_weather else "")
    weather = WeatherService() if with_weather else None
    completed = games[games["completed"]].copy()

    if weather is not None:
        # One request per game date carrying every venue playing that date,
        # rather than one heavy request per venue-season. Same data, a small
        # fraction of the API weight, which is what keeps us inside the free
        # tier instead of drowning in 429s.
        weather.prefetch(completed, historical=True)
        log.info("Weather: %s", weather.coverage())

    feat = build_features(completed, engine, weather,
                          with_weather=with_weather, historical=True,
                          efficiency=efficiency, context=context)

    if with_weather:
        known = feat["temp_f"].notna().mean()
        log.info("Weather resolved for %.1f%% of games", known * 100)
        if known < 0.80:
            log.warning("Less than 80%% of games have weather - the weather "
                        "features will carry little signal. Check the log "
                        "above for rate limiting.")

    eff_known = feat["home_off_ppa"].notna().mean()
    log.info("Play-level efficiency resolved for %.1f%% of games", eff_known * 100)
    if eff_known < 0.50:
        log.warning("Advanced stats are mostly missing - the model will fall "
                    "back on scoring margin alone, which is noticeably weaker.")

    ctx_known = feat["home_talent"].notna().mean()
    log.info("Recruiting talent resolved for %.1f%% of games", ctx_known * 100)

    # --- historical comparables -------------------------------------------
    # Each game draws its comparables only from games played strictly before
    # it, so this adds no leakage. Axis weights come from a throwaway booster
    # that ranks which profile dimensions actually move a scoreline.
    log.info("Finding historical comparables...")
    X0, y0 = training_matrix(feat)
    weights = learn_axis_weights(X0, y0["margin"])
    comps_engine = CompsEngine(feat, weights=weights)
    feat = attach_comps(feat, comps_engine)

    if weights:
        (config.MODELS / "comp_weights.json").write_text(json.dumps(weights, indent=2))

    log.info("Feature table: %d rows x %d cols", *feat.shape)
    return feat


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build the CFB training dataset")
    p.add_argument("--start", type=int, default=config.TRAIN_START_YEAR)
    p.add_argument("--end", type=int, default=config.TRAIN_END_YEAR)
    p.add_argument("--no-weather", action="store_true",
                   help="skip historical weather (much faster, slightly worse)")
    p.add_argument("--out", default=str(config.DATA / "training"))
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        feat = build_dataset(args.start, args.end, with_weather=not args.no_weather)
    except MissingKeyError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 2

    written = save_table(feat, args.out)
    print(f"Wrote {written}: {len(feat):,} games, "
          f"{feat['season'].min()}-{feat['season'].max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
