"""Render the daily slate as a standalone HTML dashboard.

Design notes: the page is scanned, not read. The signature element is the
spread line - one scale per game with the market's number and the model's
number placed on it and the gap between them shaded. That is the whole
argument of the page in one glyph, and it is the thing only this subject has.
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
.wrap { max-width: 1080px; margin: 0 auto; padding: 28px 20px 64px; }

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

/* ---------- summary strip ---------- */
.strip {
  display: grid; gap: 1px; background: var(--line);
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  border: 1px solid var(--line); border-radius: var(--radius);
  overflow: hidden; margin: 22px 0 26px;
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

/* ---------- section headings ---------- */
h2.sec {
  font-family: "Barlow Condensed", ui-sans-serif, sans-serif;
  text-transform: uppercase; letter-spacing: .06em; font-size: 17px;
  font-weight: 600; color: var(--muted);
  margin: 34px 0 12px; display: flex; align-items: center; gap: 10px;
}
h2.sec::after {
  content: ""; flex: 1; height: 1px; background: var(--line);
}

/* ---------- game row ---------- */
.game {
  background: var(--surface);
  border: 1px solid var(--line);
  border-left: 3px solid transparent;
  border-radius: var(--radius);
  padding: 14px 16px;
  margin-bottom: 10px;
  display: grid;
  grid-template-columns: minmax(200px, 1.15fr) minmax(220px, 1.35fr) minmax(150px, .9fr);
  gap: 16px 20px;
  align-items: center;
}
.game.flagged { border-left-color: var(--flag); box-shadow: var(--shadow); }

.matchup .teams {
  font-family: "Barlow Condensed", ui-sans-serif, sans-serif;
  font-size: 21px; font-weight: 600; line-height: 1.18;
  letter-spacing: .005em;
}
.matchup .at { color: var(--faint); font-weight: 400; padding: 0 3px; }
.matchup .meta {
  font-size: 12px; color: var(--muted); margin-top: 4px;
  display: flex; flex-wrap: wrap; gap: 4px 10px; align-items: center;
}
.wx {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 11.5px; color: var(--muted);
  background: var(--sunken); border-radius: 999px; padding: 2px 8px;
  white-space: nowrap;
}

/* ---------- spread line ---------- */
.line-wrap { min-width: 0; }
.line-head {
  display: flex; justify-content: space-between; align-items: baseline;
  font-size: 10.5px; text-transform: uppercase; letter-spacing: .08em;
  color: var(--faint); margin-bottom: 9px;
}
/* Model reads above the rule, market below, so the two numbers can never
   collide however close the marks sit. */
.axis { position: relative; height: 54px; }
.axis .rule {
  position: absolute; top: 26px; left: 0; right: 0; height: 2px;
  background: var(--line); border-radius: 2px;
}
.axis .gap {
  position: absolute; top: 26px; height: 2px; background: var(--flag);
  border-radius: 2px;
}
.mark { position: absolute; top: 0; height: 54px; }
.mark .tick {
  position: absolute; top: 20px; left: 0; width: 2px; height: 14px;
  border-radius: 2px; transform: translateX(-50%);
}
.mark.market .tick { background: var(--faint); }
.mark.model  .tick { background: var(--field); width: 3px; }
.mark .val {
  position: absolute; left: 0; transform: translateX(-50%);
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 11.5px; font-variant-numeric: tabular-nums; white-space: nowrap;
}
.mark.model  .val { top: 2px;  color: var(--field); font-weight: 600; }
.mark.market .val { top: 36px; color: var(--muted); }
.axis-legend {
  display: flex; flex-wrap: wrap; gap: 6px 14px; margin-top: 6px;
  font-size: 11px; color: var(--faint);
}
.axis-legend span { display: inline-flex; align-items: center; gap: 5px; }
.swatch { width: 9px; height: 3px; border-radius: 2px; display: inline-block; }

/* ---------- right column ---------- */
.readout { display: flex; flex-direction: column; gap: 9px; }
.prob-row {
  display: flex; justify-content: space-between; gap: 10px; font-size: 11.5px;
  color: var(--muted); font-variant-numeric: tabular-nums;
}
.prob-row .pct { font-weight: 600; color: var(--ink); }
.prob-bar {
  height: 8px; border-radius: 4px; background: var(--sunken);
  overflow: hidden; display: flex; margin: 4px 0 3px;
}
.prob-bar i { display: block; height: 100%; }
.prob-bar i.away { background: var(--faint); }
.prob-bar i.home { background: var(--field); }
.score {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 13px; font-variant-numeric: tabular-nums; color: var(--ink);
}
.score .lbl { color: var(--faint); font-size: 11px; }

.chips { display: flex; flex-wrap: wrap; gap: 6px; }
.chip {
  font-size: 11px; font-weight: 600; letter-spacing: .03em;
  padding: 3px 9px; border-radius: 999px; white-space: nowrap;
  border: 1px solid transparent;
}
.chip.tier   { background: var(--flag-soft); color: var(--flag); border-color: var(--flag); }
.chip.over   { background: var(--field-soft); color: var(--over); }
.chip.under  { background: var(--field-soft); color: var(--under); }
.chip.quiet  { background: var(--sunken); color: var(--muted); }

/* ---------- footer ---------- */
.note {
  margin-top: 40px; padding-top: 18px; border-top: 1px solid var(--line);
  font-size: 12.5px; color: var(--muted); max-width: 68ch;
}
.note strong { color: var(--ink); }
.note code {
  font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 11.5px;
  background: var(--sunken); padding: 1px 5px; border-radius: 4px;
}
.empty {
  background: var(--surface); border: 1px dashed var(--line);
  border-radius: var(--radius); padding: 32px; text-align: center;
  color: var(--muted);
}

@media (max-width: 760px) {
  .game { grid-template-columns: 1fr; gap: 14px; }
  .masthead .stamp { margin-left: 0; width: 100%; }
}
@media (prefers-reduced-motion: reduce) {
  * { animation: none !important; transition: none !important; }
}
"""

BANNER_CSS = """
.banner {
  margin: 20px 0 0; padding: 11px 14px; border-radius: var(--radius);
  background: var(--flag-soft); border: 1px solid var(--flag);
  color: var(--flag); font-size: 13px; line-height: 1.45;
}
.banner strong { color: var(--flag); }
"""

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
         '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
         'family=Barlow+Condensed:wght@400;600;700&'
         'family=IBM+Plex+Mono:wght@400;500;600&'
         'family=IBM+Plex+Sans:wght@400;500;600&display=swap">')


def _e(x) -> str:
    return html.escape(str(x if x is not None else ""))


def _fmt_signed(v) -> str:
    return "—" if v is None else f"{v:+.1f}"


def _spread_axis(game: dict) -> str:
    """One scale, both numbers on it, the disagreement shaded."""
    model_margin = game["pred_margin"]
    market_spread = game.get("market_spread")

    if market_spread is None:
        centre, span = model_margin, 10.0
        market_margin = None
    else:
        market_margin = -float(market_spread)
        centre = (model_margin + market_margin) / 2
        span = max(7.0, abs(model_margin - market_margin) * 1.9)

    lo, hi = centre - span, centre + span

    def pos(v: float) -> float:
        # Clamped well inside the track so the number labels stay on the page.
        return max(15.0, min(85.0, (v - lo) / (hi - lo) * 100.0))

    home, away = game["home_team"], game["away_team"]

    def quote(margin: float) -> str:
        """Express a home-margin as a spread on the favourite."""
        if margin >= 0:
            return f"{home} −{abs(margin):.1f}"
        return f"{away} −{abs(margin):.1f}"

    parts = ['<div class="axis"><div class="rule"></div>']

    if market_margin is not None:
        a, b = sorted([pos(model_margin), pos(market_margin)])
        parts.append(f'<div class="gap" style="left:{a:.2f}%;width:{b - a:.2f}%"></div>')
        parts.append(
            f'<div class="mark market" style="left:{pos(market_margin):.2f}%">'
            f'<div class="tick"></div><span class="val">{_e(quote(market_margin))}</span></div>')

    parts.append(
        f'<div class="mark model" style="left:{pos(model_margin):.2f}%">'
        f'<div class="tick"></div><span class="val">{_e(quote(model_margin))}</span></div>')
    parts.append("</div>")

    legend = ['<div class="axis-legend">',
              '<span><i class="swatch" style="background:var(--field)"></i>Model</span>']
    if market_margin is not None:
        legend.append('<span><i class="swatch" style="background:var(--faint)"></i>Market</span>')
        edge = game.get("spread_edge")
        if edge is not None:
            legend.append(f'<span><i class="swatch" style="background:var(--flag)"></i>'
                          f'{abs(edge):.1f} pt gap</span>')
    legend.append("</div>")

    head = ('<div class="line-head"><span>Spread</span>'
            f'<span>Week {_e(game["week"])}</span></div>')
    return f'<div class="line-wrap">{head}{"".join(parts)}{"".join(legend)}</div>'


def _game_row(g: dict) -> str:
    flagged = bool(g.get("spread_tier") or g.get("total_tier"))
    home_prob = int(round(g["home_win_prob"] * 100))

    meta = []
    if g.get("kickoff_et"):
        meta.append(_e(g["kickoff_et"]))
    if g.get("neutral_site"):
        meta.append("Neutral site")
    meta_html = " · ".join(meta)

    chips = []
    if g.get("spread_tier"):
        chips.append(f'<span class="chip tier">{_e(g["spread_tier"])} · '
                     f'{_e(g["spread_play"])}</span>')
    if g.get("total_tier") and g.get("total_play"):
        cls = "over" if g["total_play"].startswith("Over") else "under"
        chips.append(f'<span class="chip {cls}">{_e(g["total_play"])} '
                     f'({_fmt_signed(g.get("total_edge"))})</span>')
    if not chips:
        chips.append('<span class="chip quiet">No edge</span>')

    market_total = g.get("market_total")
    total_line = (f'{g["pred_total"]:.1f}' if market_total is None
                  else f'{g["pred_total"]:.1f} <span class="lbl">vs</span> {market_total:.1f}')

    return f"""
    <article class="game{' flagged' if flagged else ''}">
      <div class="matchup">
        <div class="teams">{_e(g["away_team"])} <span class="at">at</span> {_e(g["home_team"])}</div>
        <div class="meta">{meta_html}<span class="wx">{_e(g.get("weather_text", ""))}</span></div>
      </div>
      {_spread_axis(g)}
      <div class="readout">
        <div>
          <div class="prob-row"><span>Win probability</span></div>
          <div class="prob-bar">
            <i class="away" style="width:{100 - home_prob}%"></i>
            <i class="home" style="width:{home_prob}%"></i>
          </div>
          <div class="prob-row">
            <span><span class="pct">{100 - home_prob}%</span> {_e(g["away_team"])}</span>
            <span>{_e(g["home_team"])} <span class="pct">{home_prob}%</span></span>
          </div>
        </div>
        <div class="score"><span class="lbl">Proj score</span>
          {g["pred_away_points"]:.0f}–{g["pred_home_points"]:.0f}
        </div>
        <div class="score"><span class="lbl">Total</span> {total_line}</div>
        <div class="chips">{"".join(chips)}</div>
      </div>
    </article>"""


def _summary_strip(payload: dict) -> str:
    games = payload["games"]
    m = payload.get("model_metrics") or {}
    cells = [("Games today", str(len(games)))]

    flagged = [g for g in games if g.get("spread_tier")]
    cells.append(("Flagged edges", str(len(flagged))))

    if flagged:
        biggest = max(flagged, key=lambda g: abs(g["spread_edge"]))
        cells.append(("Largest gap", f'{abs(biggest["spread_edge"]):.1f} pts'))

    if m.get("margin_mae"):
        cells.append(("Margin MAE", f'{m["margin_mae"]:.1f}'))
    if m.get("market_margin_mae"):
        cells.append(("Market MAE", f'{m["market_margin_mae"]:.1f}'))
    if m.get("ats_win_pct"):
        cells.append(("Backtest ATS", f'{m["ats_win_pct"] * 100:.1f}%'))

    inner = "".join(f"<div><dt>{_e(k)}</dt><dd>{_e(v)}</dd></div>" for k, v in cells)
    return f'<dl class="strip">{inner}</dl>'


def _honesty_note(payload: dict) -> str:
    m = payload.get("model_metrics") or {}
    bits = []
    if "mae_vs_market" in m:
        d = m["mae_vs_market"]
        if d < 0:
            bits.append(f"In walk-forward backtesting the model's margin error was "
                        f"<strong>{abs(d):.2f} points lower</strong> than the closing line's.")
        else:
            bits.append(f"In walk-forward backtesting the model's margin error was "
                        f"<strong>{d:.2f} points higher</strong> than the closing line's — "
                        f"the market is still the better estimator overall.")
    if m.get("ats_win_pct") is not None:
        bits.append(f"Against the spread it hit {m['ats_win_pct'] * 100:.1f}% on "
                    f"{m.get('ats_n', 0):,} graded games; 52.4% is break-even at −110.")
    bits.append("Ratings are fit only on games played before each kickoff, and the "
                "betting line is never a model input — that is what keeps this "
                "comparison meaningful.")
    return (f'<p class="note">{" ".join(bits)} '
            f'Weather is the Open-Meteo forecast at each venue\'s coordinates for the '
            f'kickoff hour. Generated automatically — nothing was submitted by hand.</p>')


def render(payload: dict, standalone: bool = True, banner: str = "") -> str:
    """Render the dashboard.

    `standalone=False` emits just the title, styles and content, for hosts
    that supply their own document skeleton.
    """
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

    flagged = [g for g in games if g.get("spread_tier") or g.get("total_tier")]
    quiet = [g for g in games if g not in flagged]

    if games:
        body = ""
        if flagged:
            body += '<h2 class="sec">Where the model disagrees</h2>'
            body += "".join(_game_row(g) for g in flagged)
        if quiet:
            body += '<h2 class="sec">Rest of the slate</h2>'
            body += "".join(_game_row(g) for g in quiet)
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
  {_summary_strip(payload)}
  {body}
  {_honesty_note(payload)}
</div>"""

    head = f"<title>Saturday Model</title>\n{FONTS}\n<style>{CSS}{BANNER_CSS}</style>"

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
