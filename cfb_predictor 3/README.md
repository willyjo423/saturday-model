# Saturday Model

A college football prediction system that runs itself. Every morning it works
out which games are being played, pulls each team's current form and the live
weather at each stadium, predicts a margin, a total and a win probability, and
publishes a dashboard. You never submit anything.

## What it predicts

For every game on the slate:

- **Margin / spread** — predicted points, compared against the market line
- **Total** — predicted combined score, compared against the posted over/under
- **Win probability** — calibrated, not just a margin dressed up as a percentage
- **Confidence tier** — Strong / Lean / Slight, based on how far the model sits
  from the market

## How it works

```
CFBD API ──┐
           ├─→ canonical game table ─→ leak-free power ratings ─┐
Open-Meteo ┘                                                    ├─→ features ─→ models ─→ dashboard
                                     preseason priors ──────────┘
```

**Power ratings.** A ridge-regularised least-squares fit on capped scoring
margin, solved fresh at every point in the season. The rating used to predict a
week-9 game is fit only on games played through week 8, plus a preseason prior
built from last season's finish, recruiting talent, and returning production.
A second solve splits the same games into offence and defence ratings, which is
what the totals model runs on.

**Weather.** Each venue's coordinates come from the CFBD venue table; the
forecast is Open-Meteo's for the *kickoff hour*, not a daily average. Historical
training weather comes from the same provider's ERA5 archive, so the variables
are on identical scales at train and predict time. Domes get fixed indoor
conditions.

**Models.** Gradient-boosted trees on absolute error for margin and total, plus
a classifier blended with a normal CDF over the predicted margin for win
probability, then isotonic-calibrated on held-out seasons.

**The betting line is never a model input.** If it were, the model would mostly
learn to repeat it and "model vs market" would be meaningless. Lines are used
only to evaluate the model and to compute edge at prediction time.

## Setup

You need a free CollegeFootballData API key: <https://collegefootballdata.com/key>

1. **Create a repository** and push these files to it.

2. **Add the key as a secret.** Repo → Settings → Secrets and variables →
   Actions → New repository secret. Name it exactly `CFBD_API_KEY`.

3. **Turn on Pages.** Repo → Settings → Pages → Source: *Deploy from a branch*,
   branch `main`, folder `/docs`. Your dashboard will live at
   `https://<you>.github.io/<repo>/`.

4. **Run the bootstrap once.** Actions → *Bootstrap (build dataset and train)* →
   Run workflow. This pulls roughly a decade of games, builds the training set,
   trains the models, prints an honest backtest, and commits the result. It
   takes 30–90 minutes, mostly waiting on the API. It is the only slow step.

5. **Done.** The *Daily predictions* workflow now runs every morning at 11:00
   UTC and updates the dashboard. It also retrains weekly so the model keeps
   absorbing completed games.

### Running it locally instead

```bash
pip install -r requirements.txt
export CFBD_API_KEY=your_key_here

python -m cfb.selftest    # confirm the key and every endpoint work
python -m cfb.build       # build the training set (slow, once)
python -m cfb.train       # train + walk-forward backtest
python -m cfb.daily       # today's slate -> docs/index.html
```

Useful variations:

```bash
python -m cfb.daily --date 2026-11-28    # a specific day
python -m cfb.daily --days 3             # today plus the next two
python -m cfb.build --no-weather         # much faster, slightly worse
python -m tests.test_pipeline            # full verification, no API needed
```

## Reading the dashboard

Each game shows one **spread line**: a single scale with the market's number and
the model's number placed on it and the disagreement between them shaded amber.
Games where the model and market disagree meaningfully are sorted to the top and
carry a tier chip; everything else falls into "Rest of the slate".

## About accuracy

The honest benchmark is not "does it pick winners" — favourites win most games,
so a model can look good doing nothing. The benchmark is the **closing line**,
which is very hard to beat, and `python -m cfb.train` prints exactly how the
model does against it: margin MAE versus the market's, and an against-the-spread
record by confidence tier with 52.4% marked as break-even at −110.

Take that number seriously in both directions. A backtest that shows 53–54% at
the top tier is a plausible real edge. One that shows 58% almost always means
something leaked, and the first place to look is whether a feature encodes
information that wasn't available before kickoff.

`models/metrics.json` carries the full picture including per-season MAE and the
win-probability calibration table.

## Layout

```
cfb/
  api.py         CFBD client: caching, retries, tolerant of field renames
  weather.py     Open-Meteo forecast + archive, keyed to kickoff hour
  schema.py      canonical field names; degrades instead of crashing
  dataset.py     game table, venues, rest, travel, consensus lines
  ratings.py     leak-free ridge power ratings + preseason priors
  features.py    feature matrix (no betting lines)
  model.py       margin/total/win-prob models, walk-forward evaluation
  build.py       one-off dataset construction
  train.py       training + backtest CLI
  daily.py       the zero-input daily run
  dashboard.py   standalone HTML output
  selftest.py    live API verification
tests/
  simulate.py       synthetic universe with known ground truth
  test_pipeline.py  33 end-to-end checks, no API key needed
```

## Notes and limits

- The free CFBD tier has a monthly call budget. Responses are cached on disk and
  past seasons are treated as immutable, so a rebuild is nearly free. `selftest`
  reports remaining usage where the tier exposes it.
- Injuries, suspensions and transfers are not modelled. They matter, and their
  absence is the most likely reason a specific prediction looks wrong.
- Lines come from CFBD's aggregated providers, so they lag a live sportsbook.
  Treat the edge as directional, not as a price you can actually get.
- This is a forecasting tool. Nothing here is betting advice.
