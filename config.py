"""Central configuration. Everything tunable lives here."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE = DATA / "cache"
MODELS = ROOT / "models"
DOCS = ROOT / "docs"

for _p in (DATA, CACHE, MODELS, DOCS):
    _p.mkdir(parents=True, exist_ok=True)

# --- Credentials -----------------------------------------------------------
CFBD_API_KEY = os.environ.get("CFBD_API_KEY", "").strip()
CFBD_BASE = "https://api.collegefootballdata.com"

# Open-Meteo needs no key.
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

# --- Training window -------------------------------------------------------
# 2020 is included but flagged; COVID scheduling makes it an outlier season.
TRAIN_START_YEAR = int(os.environ.get("TRAIN_START_YEAR", 2015))
TRAIN_END_YEAR = int(os.environ.get("TRAIN_END_YEAR", 2025))
COVID_YEARS = {2020}

# Weeks 1-2 of a season have almost no in-season signal; the preseason prior
# carries them. We still train on them so the model learns to lean on the prior.
REGULAR_SEASON_WEEKS = 15

# --- Ratings engine --------------------------------------------------------
# Ridge penalty for the adjusted-margin solve. Higher = more shrinkage toward
# the preseason prior, which is what you want early in a season.
RIDGE_LAMBDA_BASE = 40.0
# Margin is capped before rating so blowouts don't dominate the fit.
MARGIN_CAP = 28.0
# Home field advantage seed (points). The ratings solve re-estimates this.
HFA_PRIOR = 2.4
# How much a prior-year rating carries into the next preseason.
YEAR_CARRYOVER = 0.60
# Games lose half their weight in the ratings solve every N weeks. A team in
# November is not the team that opened in September - injuries, freshmen
# developing, schemes settling - so recent evidence should count for more.
# Set to 0 to weight every game equally.
RECENCY_HALFLIFE_WEEKS = 6.0

# --- Model -----------------------------------------------------------------
RANDOM_SEED = 1729
# Edge thresholds (points) used to tier daily picks.
EDGE_TIERS = [(6.0, "Strong"), (3.5, "Lean"), (2.0, "Slight")]

# --- Runtime ---------------------------------------------------------------
REQUEST_TIMEOUT = 45
MAX_RETRIES = 4
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", 6 * 3600))
# Historical seasons never change, so their cache never expires.
IMMUTABLE_CACHE_BEFORE_YEAR = int(os.environ.get("IMMUTABLE_BEFORE", 2026))
