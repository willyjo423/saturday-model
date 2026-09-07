"""A synthetic college football universe.

The pipeline can't be exercised against the live API from every environment
(and burning API quota on smoke tests is wasteful), so this builds a fake
world with known ground truth: teams have real hidden strengths, weather has
a real effect on scoring, and the "market" is a good-but-imperfect observer.

That gives the test suite something it can actually assert against - if the
model can't recover a signal we planted ourselves, the pipeline is broken.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RNG = np.random.default_rng(20260903)

N_TEAMS = 120
CONFERENCES = ["Big Ten", "SEC", "Big 12", "ACC", "Pac-12", "Mountain West",
               "Sun Belt", "MAC", "AAC", "Conference USA"]

# Ground truth the model is supposed to rediscover.
TRUE_HFA = 2.6
TRUE_WIND_EFFECT = -0.28      # points of total per mph over 8
TRUE_COLD_EFFECT = -0.05      # points of total per degree under 50
TRUE_TRAVEL_EFFECT = -0.0009  # points of margin per mile the away team flies


def make_teams() -> pd.DataFrame:
    names = [f"Team {i:03d}" for i in range(N_TEAMS)]
    latent = RNG.normal(0, 9.5, N_TEAMS)
    return pd.DataFrame({
        "team": names,
        "conference": [CONFERENCES[i % len(CONFERENCES)] for i in range(N_TEAMS)],
        "latent": latent,
        "pace": RNG.normal(55, 4, N_TEAMS),
    })


def make_venues(teams: pd.DataFrame) -> list[dict]:
    venues = []
    for i, row in teams.iterrows():
        venues.append({
            "id": 1000 + i,
            "name": f"{row['team']} Stadium",
            "city": "Somewhere",
            "state": "XX",
            "timezone": "America/New_York",
            "latitude": float(RNG.uniform(26.0, 47.5)),
            "longitude": float(RNG.uniform(-123.0, -71.0)),
            "elevation": float(abs(RNG.normal(300, 600))),
            "capacity": int(RNG.integers(25_000, 108_000)),
            "grass": bool(RNG.random() > 0.5),
            "dome": bool(RNG.random() < 0.08),
        })
    return venues


def season_strength(teams: pd.DataFrame, year: int) -> pd.Series:
    """Team strength drifts year to year but is autocorrelated, like reality."""
    drift = RNG.normal(0, 3.2, len(teams))
    return pd.Series(teams["latent"].to_numpy() + drift, index=teams["team"])


def simulate_season(teams: pd.DataFrame, venues: list[dict], year: int,
                    weeks: int = 13) -> tuple[list[dict], dict]:
    strength = season_strength(teams, year)
    venue_by_team = {t: v for t, v in zip(teams["team"], venues)}
    names = list(teams["team"])
    games = []
    gid = year * 100_000

    for week in range(1, weeks + 1):
        shuffled = list(RNG.permutation(names))
        for i in range(0, len(shuffled) - 1, 2):
            home, away = shuffled[i], shuffled[i + 1]
            neutral = bool(RNG.random() < 0.04)
            venue = venue_by_team[home]

            is_dome = venue["dome"]
            if is_dome:
                temp, wind = 70.0, 0.0
            else:
                # Colder and windier as the season goes on.
                temp = float(RNG.normal(78 - week * 2.4, 11))
                wind = float(abs(RNG.normal(7 + week * 0.25, 4)))
            precip = 0.0 if is_dome else float(max(0.0, RNG.normal(-0.03, 0.09)))

            travel = 0.0 if neutral else float(
                abs(venue["latitude"] - venue_by_team[away]["latitude"]) * 69 +
                abs(venue["longitude"] - venue_by_team[away]["longitude"]) * 53)

            hfa = 0.0 if neutral else TRUE_HFA
            true_margin = (strength[home] - strength[away] + hfa
                           + TRUE_TRAVEL_EFFECT * travel)

            base_total = 52.0 + (teams.set_index("team").loc[home, "pace"]
                                 + teams.set_index("team").loc[away, "pace"] - 110) * 0.35
            weather_hit = (TRUE_WIND_EFFECT * max(0.0, wind - 8)
                           + TRUE_COLD_EFFECT * max(0.0, 50 - temp)
                           - 6.0 * min(precip, 0.5))
            true_total = base_total + weather_hit

            margin = true_margin + RNG.normal(0, 14.5)
            total = max(10.0, true_total + RNG.normal(0, 12.0))
            home_pts = int(round((total + margin) / 2))
            away_pts = int(round((total - margin) / 2))
            home_pts, away_pts = max(0, home_pts), max(0, away_pts)

            gid += 1
            games.append({
                "id": gid,
                "season": year,
                "week": week,
                "seasonType": "regular",
                "startDate": (pd.Timestamp(f"{year}-08-28", tz="UTC")
                              + pd.Timedelta(days=7 * (week - 1))
                              + pd.Timedelta(hours=int(RNG.integers(16, 28)))).isoformat(),
                "neutralSite": neutral,
                "conferenceGame": bool(RNG.random() < 0.6),
                "venueId": venue["id"],
                "venue": venue["name"],
                "homeId": names.index(home),
                "homeTeam": home,
                "homeConference": teams.set_index("team").loc[home, "conference"],
                "homePoints": home_pts,
                "homeClassification": "fbs",
                "awayId": names.index(away),
                "awayTeam": away,
                "awayConference": teams.set_index("team").loc[away, "conference"],
                "awayPoints": away_pts,
                "awayClassification": "fbs",
                "completed": True,
                # Ground truth carried alongside for the weather join.
                "_temp": temp, "_wind": wind, "_precip": precip,
                "_true_margin": true_margin, "_true_total": true_total,
            })
    return games, {"strength": strength}


def make_lines(games: list[dict], sharpness: float = 3.1) -> list[dict]:
    """A market that knows the truth plus a little noise, rounded to halves."""
    out = []
    for g in games:
        market_margin = g["_true_margin"] + RNG.normal(0, sharpness)
        market_total = g["_true_total"] + RNG.normal(0, sharpness * 1.4)
        out.append({
            "id": g["id"],
            "season": g["season"],
            "week": g["week"],
            "homeTeam": g["homeTeam"],
            "awayTeam": g["awayTeam"],
            "lines": [{
                "provider": "consensus",
                "spread": round(-market_margin * 2) / 2,
                "overUnder": round(market_total * 2) / 2,
            }],
        })
    return out


# Advanced-stat scales, roughly matching real FBS distributions.
_STAT_SPEC = {
    #                league mean, strength coefficient, per-game noise
    "ppa":            (0.170, 0.0120, 0.090),
    "successRate":    (0.425, 0.0060, 0.055),
    "explosiveness":  (1.220, 0.0090, 0.180),
    "lineYards":      (2.800, 0.0180, 0.420),
    "stuffRate":      (0.190, -0.0035, 0.045),
    "powerSuccess":   (0.680, 0.0070, 0.110),
}
_HAVOC = (0.175, 0.0030, 0.035)


def make_advanced_stats(games: list[dict], teams: pd.DataFrame) -> list[dict]:
    """Per-team-per-game advanced stats, generated additively.

    Observed rate = league mean + this offence + that defence + noise, which is
    exactly the structure EfficiencyEngine tries to invert. The noise is set
    lower relative to signal than single-game scoring margin, mirroring the
    real reason these metrics are useful: they say more per game than the
    scoreboard does.
    """
    latent = teams.set_index("team")["latent"].to_dict()
    pace = teams.set_index("team")["pace"].to_dict()
    rows = []

    for g in games:
        for team, opp in ((g["homeTeam"], g["awayTeam"]),
                          (g["awayTeam"], g["homeTeam"])):
            if team not in latent or opp not in latent:
                continue
            off_q, def_q = latent[team], latent[opp]

            offense, defense = {}, {}
            for name, (mean, coef, noise) in _STAT_SPEC.items():
                offense[name] = float(
                    mean + coef * off_q - coef * def_q + RNG.normal(0, noise))
                defense[name] = float(
                    mean + coef * def_q - coef * off_q + RNG.normal(0, noise))

            m, c, n = _HAVOC
            offense["havoc"] = {"total": float(m - c * off_q + c * def_q
                                               + RNG.normal(0, n))}
            defense["havoc"] = {"total": float(m + c * def_q - c * off_q
                                               + RNG.normal(0, n))}

            tempo = (pace.get(team, 55) + pace.get(opp, 55)) / 2
            offense["plays"] = float(tempo + RNG.normal(0, 5))
            defense["plays"] = float(tempo + RNG.normal(0, 5))

            rows.append({
                "gameId": g["id"], "season": g["season"], "week": g["week"],
                "team": team, "opponent": opp,
                "offense": offense, "defense": defense,
            })
    return rows


def make_preseason_inputs(teams: pd.DataFrame, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recruiting talent and returning production, CFBD-shaped.

    Both are noisy reads on the same hidden strength - which is what they are
    in reality, and why they belong in the preseason prior rather than the
    in-season solve.
    """
    latent = teams["latent"].to_numpy()
    talent = pd.DataFrame({
        "year": year,
        "school": teams["team"],
        "talent": 700 + latent * 9 + RNG.normal(0, 45, len(teams)),
    })
    returning = pd.DataFrame({
        "season": year,
        "team": teams["team"],
        "totalPPA": 55 + latent * 0.6 + RNG.normal(0, 14, len(teams)),
        "passingPPA": 28 + latent * 0.4 + RNG.normal(0, 11, len(teams)),
    })
    return talent, returning


def build_world(years: list[int], with_advanced: bool = True) -> dict:
    teams = make_teams()
    venues = make_venues(teams)
    all_games, all_lines = [], []
    for year in years:
        games, _ = simulate_season(teams, venues, year)
        all_games.extend(games)
        all_lines.extend(make_lines(games))
    world = {"teams": teams, "venues": venues,
             "games": all_games, "lines": all_lines}
    world["advanced"] = (make_advanced_stats(all_games, teams)
                         if with_advanced else [])
    world["preseason"] = {y: make_preseason_inputs(teams, y) for y in years}
    return world


class FakeWeather:
    """Replays the weather that actually generated the simulated scores.

    Keyed the same way the real service is - venue coordinates plus kickoff
    hour - so it exercises the identical code path in `build_features`.
    """

    def __init__(self, games: list[dict], venues: list[dict]):
        venue_by_id = {v["id"]: v for v in venues}
        self.by_key: dict[tuple, dict] = {}
        for g in games:
            v = venue_by_id[g["venueId"]]
            key = (round(v["latitude"], 3), round(v["longitude"], 3),
                   pd.Timestamp(g["startDate"]).strftime("%Y-%m-%dT%H"))
            self.by_key[key] = g

    def prefetch(self, games, historical=True):
        """Nothing to warm - the index is built in __init__."""

    def coverage(self) -> str:
        return f"stub, {len(self.by_key):,} venue-hours"

    def at_kickoff(self, lat, lon, kickoff, is_dome=False, historical=None):
        neutral = {"temp_f": 65.0, "humidity": 60.0, "precip_in": 0.0,
                   "wind_mph": 6.0, "is_dome": int(bool(is_dome))}
        if lat is None or pd.isna(lat) or kickoff is None or pd.isna(kickoff):
            return neutral
        key = (round(float(lat), 3), round(float(lon), 3),
               pd.Timestamp(kickoff).strftime("%Y-%m-%dT%H"))
        g = self.by_key.get(key)
        if g is None:
            return neutral
        return {"temp_f": g["_temp"], "humidity": 60.0, "precip_in": g["_precip"],
                "wind_mph": g["_wind"], "is_dome": int(bool(is_dome))}
