"""Render the daily slate as a standalone HTML dashboard.

Design notes
------------
The page answers one question per game: *when games like this have been played
before, what happened?* So each card leads with the side and the price, the
rate at which comparable games covered it, and - crucially - what that rate has
actually been worth once measured out of sample. Then five real precedents,
named and scored, so a reader can check the reasoning by eye rather than
trusting it.

The model still does the work of deciding which games are alike. It just does
not front the card any more. Its numbers stay in predictions.json.
"""
from __future__ import annotations

import html
import json
from datetime import datetime

CSS = """
:root {
  color-scheme: light dark;
  --ground:  #F4F5F2;
  --surface: #FFFFFF;
  --sunken:  #EBEEE9;
  --ink:     #131A15;
  --muted:   #5A6860;
  --faint:   #8B978F;
  --line:    #E0E4DD;
  --field:   #1B5E3F;
  --field-soft: #E4EFE8;
  --flag:    #B8790C;
  --flag-soft:  #FBF1DC;
  --over:    #0F7B6C;
  --under:   #A63A2B;
  --shadow:  0 1px 2px rgba(19,26,21,.06), 0 4px 14px rgba(19,26,21,.05);
  --radius:  10px;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground:  #0D1210;
    --surface: #151C18;
    --sunken:  #1D2620;
    --ink:     #E9EEE9;
    --muted:   #94A59A;
    --faint:   #6C7C72;
    --line:    #27322C;
    --field:   #58B183;
    --field-soft: #1A2B22;
    --flag:    #E3A93C;
    --flag-soft:  #2C2415;
    --over:    #3FBFA8;
    --under:   #E0705C;
    --shadow:  0 1px 2px rgba(0,0,0,.4), 0 4px 16px rgba(0,0,0,.3);
  }
}
:root[data-theme="dark"] {
  --ground:  #0D1210;
  --surface: #151C18;
  --sunken:  #1D2620;
  --ink:     #E9EEE9;
  --muted:   #94A59A;
  --faint:   #6C7C72;
  --line:    #27322C;
  --field:   #58B183;
  --field-soft: #1A2B22;
  --flag:    #E3A93C;
  --flag-soft:  #2C2415;
  --over:    #3FBFA8;
  --under:   #E0705C;
  --shadow:  0 1px 2px rgba(0,0,0,.4), 0 4px 16px rgba(0,0,0,.3);
}

* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--ground);
  color: var(--ink);
  font-family: "IBM Plex Sans", ui-sans-serif, system-ui, -apple-system, sans-serif;
  font-size: 15px;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 940px; margin: 0 auto; padding: 28px 20px 64px; }

/* ---------- masthead ---------- */
.masthead {
  display: flex; flex-wrap: wrap; align-items: baseline; gap: 10px 18px;
  padding-bottom: 16px; border-bottom: 2px solid var(--field);
}
.masthead h1 {
  font-family: "Barlow Condensed", ui-sans-serif, sans-serif;
  font-weight: 700; font-size: clamp(30px, 5vw, 44px);
  letter-spacing: .01em; text-transform: uppercase;
  margin: 0; line-height: 1; text-wrap: balance;
}
.masthead .date { color: var(--muted); font-size: 14px; }
.masthead .stamp {
  margin-left: auto; font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 11px; color: var(--faint); letter-spacing: .04em;
}

.lede {
  margin: 18px 0 0; max-width: 62ch;
  font-size: 13.5px; color: var(--muted); line-height: 1.55;
}
.lede b { color: var(--ink); font-weight: 600; }

/* ---------- summary strip ---------- */
.strip {
  display: grid; gap: 1px; background: var(--line);
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  border: 1px solid var(--line); border-radius: var(--radius);
  overflow: hidden; margin: 20px 0 26px;
}
.strip div { background: var(--surface); padding: 12px 14px; }
.strip dt {
  font-size: 10.5px; text-transform: uppercase; letter-spacing: .09em;
  color: var(--faint); margin: 0 0 3px;
}
.strip dd {
  margin: 0; font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 19px; font-weight: 500; font-variant-numeric: tabular-nums;
}

h2.sec {
  font-family: "Barlow Condensed", ui-sans-serif, sans-serif;
  text-transform: uppercase; letter-spacing: .06em; font-size: 17px;
  font-weight: 600; color: var(--muted);
  margin: 34px 0 12px; display: flex; align-items: center; gap: 10px;
}
h2.sec::after { content: ""; flex: 1; height: 1px; background: var(--line); }

/* ---------- game card ---------- */
.game {
  background: var(--surface);
  border: 1px solid var(--line);
  border-left: 3px solid transparent;
  border-radius: var(--radius);
  padding: 16px 18px 14px;
  margin-bottom: 12px;
}
.game.play { border-left-color: var(--field); box-shadow: var(--shadow); }

.head {
  display: flex; flex-wrap: wrap; align-items: baseline;
  gap: 4px 14px; margin-bottom: 12px;
}
.head .teams {
  font-family: "Barlow Condensed", ui-sans-serif, sans-serif;
  font-size: 23px; font-weight: 600; line-height: 1.1; letter-spacing: .005em;
}
.head .at { color: var(--faint); font-weight: 400; padding: 0 3px; }
.head .when { font-size: 12.5px; color: var(--muted); }
.head .wx {
  margin-left: auto;
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 11.5px; color: var(--muted);
  background: var(--sunken); border-radius: 999px; padding: 2px 9px;
  white-space: nowrap;
}

/* ---------- the verdict ---------- */
.verdict {
  display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 12px;
  margin-bottom: 6px;
}
.pick {
  font-family: "Barlow Condensed", ui-sans-serif, sans-serif;
  font-size: 30px; font-weight: 700; line-height: 1.05;
  letter-spacing: .01em; color: var(--field);
}
.pick .num { font-family: "IBM Plex Mono", ui-monospace, monospace;
             font-size: 25px; font-weight: 600; }
.pick.none { color: var(--faint); font-size: 24px; font-weight: 600; }
.tag {
  font-size: 11px; font-weight: 600; letter-spacing: .04em;
  padding: 3px 10px; border-radius: 999px;
  background: var(--field-soft); color: var(--field);
  border: 1px solid var(--field);
}
.tag.lean   { background: var(--flag-soft); color: var(--flag); border-color: var(--flag); }
.tag.slight { background: var(--sunken); color: var(--muted); border-color: var(--line); }

.because { font-size: 14px; color: var(--ink); margin-bottom: 3px; }
.because b {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-variant-numeric: tabular-nums; font-weight: 600;
}
.track-record { font-size: 12.5px; color: var(--muted); }
.track-record b {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  color: var(--ink); font-weight: 600; font-variant-numeric: tabular-nums;
}

/* ---------- outcome band ---------- */
.band-row {
  display: flex; align-items: center; gap: 14px;
  margin: 14px 0 4px; flex-wrap: wrap;
}
.band { position: relative; height: 22px; flex: 1 1 240px; min-width: 180px; }
.band .track {
  position: absolute; top: 10px; left: 0; right: 0; height: 3px;
  background: var(--sunken); border-radius: 2px;
}
.band .iqr {
  position: absolute; top: 7px; height: 9px;
  background: var(--field-soft); border: 1px solid var(--field);
  border-radius: 5px;
}
.band .med {
  position: absolute; top: 3px; width: 3px; height: 17px;
  background: var(--field); border-radius: 2px;
}
.band .zero { position: absolute; top: 1px; width: 1px; height: 21px;
              background: var(--faint); opacity: .5; }
.band-label {
  font-size: 12px; color: var(--muted); white-space: nowrap;
}
.band-label b {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  color: var(--ink); font-weight: 600; font-variant-numeric: tabular-nums;
}

.market {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 12px; color: var(--faint);
  font-variant-numeric: tabular-nums; margin-top: 2px;
}

/* ---------- precedents ---------- */
details.prec { margin-top: 12px; }
details.prec > summary {
  cursor: pointer; list-style: none;
  font-size: 10.5px; text-transform: uppercase; letter-spacing: .09em;
  color: var(--faint); padding: 6px 0 4px;
  border-top: 1px dashed var(--line);
}
details.prec > summary::-webkit-details-marker { display: none; }
details.prec > summary::after { content: "  ▾"; }
details.prec[open] > summary::after { content: "  ▴"; }
.prec-table {
  border-collapse: collapse; margin-top: 4px;
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 11.5px; font-variant-numeric: tabular-nums;
}
.prec-table td { padding: 3px 18px 3px 0; color: var(--muted); }
.prec-table tr.hd th {
  text-align: left; padding: 0 18px 5px 0;
  font-family: "IBM Plex Sans", ui-sans-serif, sans-serif;
  font-size: 10px; font-weight: 600; text-transform: uppercase;
  letter-spacing: .07em; color: var(--faint);
  border-bottom: 1px solid var(--line); white-space: nowrap;
}
.prec-table tr.hd th:first-child { padding-left: 0; }
.prec-table tr:not(.hd) td:first-child { padding-top: 6px; }
.prec-key {
  margin: 8px 0 2px; font-size: 11.5px; color: var(--muted);
  line-height: 1.5; max-width: 70ch;
}
.prec-key b { color: var(--ink); font-weight: 600; }
.prec-table td.yr { color: var(--faint); width: 3.5em; }
.prec-table td.gm { color: var(--ink); }
.prec-table td.sc { color: var(--ink); white-space: nowrap; }
.prec-table td.ln { white-space: nowrap; }
.prec-table td.rs { white-space: nowrap; font-weight: 600; }
.prec-table td.rs.yes { color: var(--field); }
.prec-table td.rs.no  { color: var(--under); }
.prec-wrap { overflow-x: auto; }

.flags {
  margin-top: 10px; display: flex; flex-wrap: wrap; gap: 4px 14px;
  font-size: 11.5px; color: var(--faint);
}
.flags .lead { text-transform: uppercase; letter-spacing: .08em; }
.flags i { font-style: normal; color: var(--under); opacity: .9; }

.totals {
  margin-top: 8px; font-size: 12.5px; color: var(--muted);
}
.totals b {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  color: var(--ink); font-weight: 600;
}

.note {
  margin-top: 40px; padding-top: 18px; border-top: 1px solid var(--line);
  font-size: 12.5px; color: var(--muted); max-width: 66ch;
}
.note strong { color: var(--ink); }
.empty {
  background: var(--surface); border: 1px dashed var(--line);
  border-radius: var(--radius); padding: 32px; text-align: center;
  color: var(--muted);
}
.banner {
  margin: 20px 0 0; padding: 11px 14px; border-radius: var(--radius);
  background: var(--flag-soft); border: 1px solid var(--flag);
  color: var(--flag); font-size: 13px; line-height: 1.45;
}
.banner strong { color: var(--flag); }

@media (max-width: 640px) {
  .head .wx { margin-left: 0; }
  .pick { font-size: 26px; }
}
@media (prefers-reduced-motion: reduce) {
  * { animation: none !important; transition: none !important; }
}
"""

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
         '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
         'family=Barlow+Condensed:wght@400;600;700&'
         'family=IBM+Plex+Mono:wght@400;500;600&'
         'family=IBM+Plex+Sans:wght@400;500;600&display=swap">')


def _e(x) -> str:
    return html.escape(str(x if x is not None else ""))


def _pct(v, digits=0) -> str:
    return "—" if v is None else f"{float(v) * 100:.{digits}f}%"


# ---------------------------------------------------------------- card parts
def _verdict(g: dict) -> str:
    """The single thing this card is telling you."""
    play = g.get("play")
    comps = g.get("comps") or {}

    if not play:
        n = comps.get("n")
        reason = ("comparable games split too evenly to favour a side"
                  if n else "not enough comparable games")
        return (f'<div class="verdict"><span class="pick none">No play</span>'
                f'</div><div class="because">{_e(reason)}.</div>')

    tier = play.get("tier") or "Slight"
    cls = {"Strong": "", "Lean": "lean", "Slight": "slight"}.get(tier, "slight")
    tag = f'<span class="tag {cls}">{_e(tier)}</span>'

    pick = (f'<span class="pick">{_e(play["team"])} '
            f'<span class="num">{_e(play["line"])}</span></span>')

    rate = comps.get("cover_rate")
    side = comps.get("side")
    shown = rate if side == "home" else (None if rate is None else 1 - rate)
    n = comps.get("n")

    because = (f'<div class="because">This side covered in '
               f'<b>{_pct(shown)}</b> of <b>{n}</b> similar games.</div>')

    hist_rate = play.get("expected_rate")
    hist_n = play.get("sample")
    if hist_rate is not None and hist_n:
        track = (f'<div class="track-record">When the comparables have said '
                 f'this before, the side actually won <b>{_pct(hist_rate, 1)}</b> '
                 f'of the time, over <b>{hist_n:,}</b> graded games. '
                 f'Break-even is 52.4%.</div>')
    else:
        track = ('<div class="track-record">No measured track record for a '
                 'signal this strong yet — treat it as untested.</div>')

    return f'<div class="verdict">{pick}{tag}</div>{because}{track}'


def _band(g: dict) -> str:
    """Where comparable games actually finished, on the home team's scale."""
    c = g.get("comps") or {}
    p25, p75 = c.get("margin_p25"), c.get("margin_p75")
    if p25 is None or p75 is None:
        return ""
    med = c.get("margin_median") or 0.0
    lo = min(p25, med, 0.0) - 7
    hi = max(p75, med, 0.0) + 7
    span = max(hi - lo, 1.0)

    def pos(v):
        return max(0.0, min(100.0, (v - lo) / span * 100.0))

    a, b = pos(p25), pos(p75)
    home = g["home_team"]
    win = c.get("home_win_rate")
    win_txt = (f' · {_e(home)} won <b>{_pct(win)}</b>' if win is not None else "")

    return (
        '<div class="band-row">'
        f'<div class="band" title="Middle half of comparable final margins">'
        f'<div class="track"></div>'
        f'<div class="iqr" style="left:{a:.1f}%;width:{max(b - a, 1.0):.1f}%"></div>'
        f'<div class="zero" style="left:{pos(0.0):.1f}%"></div>'
        f'<div class="med" style="left:{pos(med):.1f}%"></div></div>'
        f'<div class="band-label">Half finished <b>{p25:+.0f} to {p75:+.0f}</b>'
        f'{win_txt}</div>'
        '</div>')


def _market_line(g: dict) -> str:
    bits = []
    if g.get("market_spread") is not None:
        bits.append(f'{_e(g["home_team"])} {g["market_spread"]:+.1f}')
    if g.get("market_total") is not None:
        bits.append(f'total {g["market_total"]:.1f}')
    if not bits:
        return '<div class="market">No market line available</div>'
    return f'<div class="market">Market &nbsp;{" &nbsp;·&nbsp; ".join(bits)}</div>'


def _totals_line(g: dict) -> str:
    tp = g.get("total_play")
    if not tp:
        return ""
    cls = "over" if tp["side"] == "Over" else "under"
    return (f'<div class="totals">Total: <b>{_e(tp["side"])} {tp["line"]:.1f}</b> '
            f'— comparable games went {_e(tp["side"].lower())} '
            f'<b>{_pct(tp["rate"] if tp["side"] == "Over" else 1 - tp["rate"])}</b> '
            f'of the time.</div>')


def _precedents(g: dict) -> str:
    """The five nearest real games, with the role mapping spelled out.

    Comparables are built from the home team's point of view, so in every
    precedent the home team stands in for this game's home team and the
    visitor stands in for the visitor. Without saying that plainly the table
    is unreadable - and the spread column is the *home* team's line, which is
    the other thing nobody can guess.
    """
    rows = g.get("comp_examples") or []
    if not rows:
        return ""

    home, away = g["home_team"], g["away_team"]
    has_play = bool(g.get("play"))
    side = (g.get("comps") or {}).get("side") if has_play else None
    total_n = (g.get("comps") or {}).get("n")

    body = []
    for e in rows[:5]:
        # Winner first and named, so nobody has to work out the orientation.
        hp, ap = e["home_points"], e["away_points"]
        if hp > ap:
            final = f'{_e(e["home_team"])} {hp}–{ap}'
        elif ap > hp:
            final = f'{_e(e["away_team"])} {ap}–{hp}'

        else:
            final = f'tied {hp}–{ap}'

        covered = e.get("home_covered")
        if covered is None:
            res, cls = "push", ""
        else:
            res = f'{_e(e["home_team"] if covered else e["away_team"])}'
            if side is None:
                cls = ""
            else:
                cls = "yes" if (covered if side == "home" else not covered) else "no"

        body.append(
            f'<tr><td class="yr">{e["season"]}</td>'
            f'<td class="gm">{_e(e["away_team"])} at {_e(e["home_team"])}</td>'
            f'<td class="sc">{final}</td>'
            f'<td class="ln">{e["home_spread"]:+.1f}</td>'
            f'<td class="rs {cls}">{res}</td></tr>')

    this_line = g.get("market_spread")
    line_head = ("Home line" if this_line is None
                 else f"Home line (here {this_line:+.1f})")

    header = ('<tr class="hd"><th>Season</th><th>Game</th><th>Final</th>'
              f'<th>{_e(line_head)}</th><th>Covered</th></tr>')

    key = (f'<p class="prec-key">In each of these the <b>home team stands in '
           f'for {_e(home)}</b> and the visitor stands in for {_e(away)}. '
           f'The line shown is the home team’s.'
           + (' Green means the game went the way this pick needs.'
              if side else '')
           + '</p>')

    label = ("Five closest precedents" if not total_n
             else f"Five closest of {total_n} comparable games")

    return ('<details class="prec">'
            f'<summary>{_e(label)}</summary>{key}'
            f'<div class="prec-wrap"><table class="prec-table">'
            f'{header}{"".join(body)}</table></div></details>')


def _flags(g: dict) -> str:
    flags = g.get("confidence_flags") or []
    if not flags or not g.get("play"):
        return ""
    items = "".join(f"<span><i>▲</i> {_e(f)}</span>" for f in flags[:3])
    return f'<div class="flags"><span class="lead">Watch</span>{items}</div>'


def _game_row(g: dict) -> str:
    has_play = bool(g.get("play"))
    meta = []
    if g.get("kickoff_et"):
        meta.append(_e(g["kickoff_et"]))
    if g.get("neutral_site"):
        meta.append("Neutral site")

    return f"""
    <article class="game{' play' if has_play else ''}">
      <div class="head">
        <span class="teams">{_e(g["away_team"])} <span class="at">at</span> {_e(g["home_team"])}</span>
        <span class="when">{" · ".join(meta)}</span>
        <span class="wx">{_e(g.get("weather_text", ""))}</span>
      </div>
      {_verdict(g)}
      {_band(g)}
      {_market_line(g)}
      {_totals_line(g)}
      {_flags(g)}
      {_precedents(g)}
    </article>"""


# ---------------------------------------------------------------- page parts
def _summary_strip(payload: dict) -> str:
    games = payload["games"]
    plays = [g for g in games if g.get("play")]
    cells = [("Games", str(len(games))), ("Plays", str(len(plays)))]

    strong = sum(1 for g in plays if (g["play"].get("tier") == "Strong"))
    if plays:
        cells.append(("Strong", str(strong)))

    m = payload.get("model_metrics") or {}
    cc = m.get("comps_calibration") or {}
    if cc.get("n_fitted"):
        cells.append(("Signal tested on", f'{cc["n_fitted"]:,}'))
    if m.get("market_margin_mae"):
        cells.append(("Market MAE", f'{m["market_margin_mae"]:.1f}'))

    inner = "".join(f"<div><dt>{_e(k)}</dt><dd>{_e(v)}</dd></div>" for k, v in cells)
    return f'<dl class="strip">{inner}</dl>'


def _lede(payload: dict) -> str:
    m = payload.get("model_metrics") or {}
    cc = m.get("comps_calibration") or {}
    tested = (f' Every rate below was checked against <b>{cc["n_fitted"]:,}</b> '
              f'games the model had never seen.' if cc.get("n_fitted") else "")
    return (f'<p class="lede">Each game is matched against the most similar '
            f'matchups of the last decade — similar teams, and a similar price '
            f'— and the card reports what those games actually did.{tested}</p>')


def _honesty_note(payload: dict) -> str:
    m = payload.get("model_metrics") or {}
    bits = []
    cc = m.get("comps_calibration") or {}
    if cc.get("slope") is not None and cc.get("n_fitted"):
        bits.append(
            f"The raw rate from comparable games overstates its own accuracy — "
            f"200 nearest neighbours are nowhere near 200 independent games — "
            f"so it is corrected against {cc['n_fitted']:,} out-of-sample "
            f"results before anything is called a play.")
    bits.append("Comparables only ever come from games played earlier than the "
                "one being predicted, and the betting line is never an input to "
                "the model that decides which games are alike.")
    bits.append("This is a forecasting tool, not advice.")
    return f'<p class="note">{" ".join(bits)}</p>'


def render(payload: dict, standalone: bool = True, banner: str = "") -> str:
    games = payload["games"]
    generated = payload.get("generated_at", "")
    try:
        stamp = datetime.fromisoformat(generated).strftime("%b %-d, %Y · %-I:%M %p ET")
    except (ValueError, TypeError):
        stamp = generated

    dates = payload.get("dates") or []
    try:
        pretty = datetime.strptime(dates[0], "%Y-%m-%d").strftime("%A, %B %-d, %Y")
        if len(dates) > 1:
            end = datetime.strptime(dates[-1], "%Y-%m-%d").strftime("%B %-d")
            pretty = f"{pretty} – {end}"
    except (ValueError, IndexError):
        pretty = ", ".join(dates)

    plays = [g for g in games if g.get("play")]
    rest = [g for g in games if not g.get("play")]

    if games:
        body = ""
        if plays:
            body += '<h2 class="sec">Where the precedents point somewhere</h2>'
            body += "".join(_game_row(g) for g in plays)
        if rest:
            body += '<h2 class="sec">Rest of the slate</h2>'
            body += "".join(_game_row(g) for g in rest)
    else:
        body = ('<div class="empty">No games scheduled for this date. '
                'The next run will pick up the following slate automatically.</div>')

    banner_html = f'<div class="banner">{banner}</div>' if banner else ""

    content = f"""<div class="wrap">
  <header class="masthead">
    <h1>Saturday Model</h1>
    <span class="date">{_e(pretty)}</span>
    <span class="stamp">Built {_e(stamp)}</span>
  </header>
  {banner_html}
  {_lede(payload)}
  {_summary_strip(payload)}
  {body}
  {_honesty_note(payload)}
</div>"""

    head = f"<title>Saturday Model</title>\n{FONTS}\n<style>{CSS}</style>"

    if not standalone:
        return f"{head}\n{content}"

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
{head}
</head>
<body>
{content}
</body>
</html>"""


def render_from_file(path: str) -> str:
    with open(path) as fh:
        return render(json.load(fh))
