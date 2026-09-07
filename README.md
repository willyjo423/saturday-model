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

**Play-level efficiency.** Opponent-adjusted EPA per play, success rate,
explosiveness, line yards, stuff rate, power success, havoc and tempo, solved
week by week from per-game advanced stats. Scoring margin alone cannot tell a
dominant team from a lucky one; this can, and it stabilises by about week 4
rather than week 8.

**Models.** Gradient-boosted trees predict the *residual* from a rescaled
ratings baseline rather than the margin outright. That matters: trees can only
average training leaves, so predicting margin directly meant the model could
never express a game more lopsided than ones it had already seen, and a
45-point mismatch came back as a 20-point favourite. The linear baseline
supplies the level and extrapolates without limit; the trees correct it.

Win probability is a two-parameter logistic in the predicted margin, fitted on
out-of-sample predictions. Isotonic regression was tried first and was the
wrong tool - on a few hundred held-out games it produced flat plateaus, so a
coin flip came out at 61% and a near-certainty was dragged down to 90%.

**Historical comparables.** Every game is matched against the closest matchups
of the last decade, on a dozen weighted profile axes, drawing only on games
played before it. This is what puts a *distribution* on the dashboard - the
middle half of comparable outcomes, how often the favourite actually won, and
four named precedents you can eyeball. Measured honestly, the comps add almost
nothing to point accuracy (the booster already partitions the feature space in
a similar way); they earn their place by showing you the spread of plausible
outcomes rather than a single confident number.

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

python selftest.py    # confirm the key and every endpoint work
python build.py       # build the training set (slow, once)
python train.py       # train + walk-forward backtest
python daily.py       # today's slate -> docs/index.html
```

Useful variations:

```bash
python daily.py --date 2026-11-28    # a specific day
python daily.py --days 3             # today plus the next two
python build.py --no-weather         # much faster, slightly worse
python test_pipeline.py            # full verification, no API needed
```

## Reading the dashboard

Each game shows one **spread line**: a single scale with the market's number and
the model's number placed on it and the disagreement between them shaded amber.
Games where the model and market disagree meaningfully are sorted to the top and
carry a tier chip; everything else falls into "Rest of the slate".

## Edge calibration — why big disagreements are usually wrong

Sort games by how far the model sits from the closing line and the biggest
disagreements tend to go *against* the model. That is arithmetic, not luck.
Writing model = truth + our error and market = truth + a much smaller error,
the edge between them is close to *our own error*. Ranking by edge size is
therefore close to ranking our predictions by how wrong they are, then backing
the worst ones hardest.

It is worse than that, because large edges cluster on the games where the
inputs are weakest: a quarterback out that the market knows about and the model
does not, a team three games into a season, a stale line that has not moved.

So `train.py` measures, on out-of-sample predictions only, what fraction of a
claimed edge actually materialises at each edge size, and prints it:

```
|edge|        n     ATS       95% band      keeps
 1.5-3.0      333   49.5%   44.2-54.9 %   41.1%
 3.0-4.5      321   54.8%   49.4-60.3 %   47.1%
 6.0-9.0      421   54.4%   49.6-59.2 %    9.6%
13.0-99.0     139   54.0%   45.7-62.2 %    9.6%
```

The daily run then shrinks new edges by that measured relationship, and tiers
plays by **what each bucket historically did** rather than by assuming bigger
is better. A large disagreement the backtest does not trust is labelled
"no play" with the historical rate attached, instead of being promoted.

Each card also carries named reasons to distrust it - few games played, missing
efficiency, books far apart, a line that has moved a long way since opening.
Those come from the market metadata (provider dispersion, opening number)
which is deliberately kept out of the model and used only for confidence.

## About accuracy

The honest benchmark is not "does it pick winners" — favourites win most games,
so a model can look good doing nothing. The benchmark is the **closing line**,
which is very hard to beat, and `python train.py` prints exactly how the
model does against it: margin MAE versus the market's, and an against-the-spread
record by confidence tier with 52.4% marked as break-even at −110.

Take that number seriously in both directions. A backtest that shows 53–54% at
the top tier is a plausible real edge. One that shows 58% almost always means
something leaked, and the first place to look is whether a feature encodes
information that wasn't available before kickoff.

`models/metrics.json` carries the full picture including per-season MAE and the
win-probability calibration table.

## Layout

Every file sits at the repository root, which keeps uploading simple.

```
api.py         CFBD client: caching, retries, tolerant of field renames
weather.py     Open-Meteo, batched by date, with a hard time budget
schema.py      canonical field names; degrades instead of crashing
dataset.py     game table, venues, rest, travel, consensus lines
ratings.py     leak-free ridge power ratings, recency-weighted
efficiency.py  opponent-adjusted play-level efficiency, week by week
comps.py       nearest historical matchups and their outcome distributions
features.py    feature matrix (no betting lines, ever)
model.py       baseline + residual models, walk-forward evaluation
storage.py     table IO that works with or without pyarrow
config.py      every tunable setting
build.py       one-off dataset construction
train.py       training + backtest CLI
daily.py       the zero-input daily run
dashboard.py   standalone HTML output
selftest.py    live API verification

simulate.py       synthetic universe with known ground truth
test_pipeline.py  54 end-to-end checks, no API key needed
test_daily.py     20 checks on the daily run, no API key needed
```

`data/`, `models/` and `docs/` are created automatically on first run.

## Notes and limits

- The free CFBD tier has a monthly call budget. Responses are cached on disk and
  past seasons are treated as immutable, so a rebuild is nearly free. `selftest`
  reports remaining usage where the tier exposes it.
- Injuries, suspensions and transfers are not modelled. They matter, and their
  absence is the most likely reason a specific prediction looks wrong.
- Lines come from CFBD's aggregated providers, so they lag a live sportsbook.
  Treat the edge as directional, not as a price you can actually get.
- This is a forecasting tool. Nothing here is betting advice.
