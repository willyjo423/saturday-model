"""Render the daily slate as a standalone HTML dashboard.

Design notes
------------
This is a forecasting page, not a tipsheet. Each card leads with a projected
score and a win probability, then shows how comparable historical games
finished, then the market alongside the model for orientation, then five real
precedents so the reasoning can be checked by eye.

It deliberately makes no recommendation. Measured across 5,000+ out-of-sample
games, neither the model's disagreement with the closing line nor the
comparables' cover rate predicted covering - every band came back at 50%.
Reporting an edge that does not exist would be the one genuinely dishonest
thing this page could do, so it reports forecasts instead.
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
.game.close { border-left-color: var(--flag); box-shadow: var(--shadow); }

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
.band-wrap { flex: 1 1 260px; min-width: 200px; }
.band { position: relative; height: 22px; }
.band-ends {
  display: flex; justify-content: space-between;
  font-size: 10.5px; color: var(--faint); letter-spacing: .02em;
  margin-top: 1px;
}
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
  font-size: 12.5px; color: var(--muted); flex: 0 1 auto;
}
.band-label b {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  color: var(--ink); font-weight: 600; font-variant-numeric: tabular-nums;
}

.prob-bar {
  height: 8px; border-radius: 4px; background: var(--sunken);
  overflow: hidden; display: flex; margin: 8px 0 4px; max-width: 460px;
}
.prob-bar i { display: block; height: 100%; }
.prob-bar i.lead  { background: var(--field); }
.prob-bar i.trail { background: var(--faint); opacity: .55; }
.prob-ends {
  display: flex; justify-content: space-between; max-width: 460px;
  font-size: 11.5px; color: var(--muted); font-variant-numeric: tabular-nums;
}
.cmp-row { display: flex; gap: 16px; align-items: baseline; }
.cmp-k {
  font-family: "IBM Plex Sans", ui-sans-serif, sans-serif;
  font-size: 10px; text-transform: uppercase; letter-spacing: .08em;
  color: var(--faint); width: 4.5em;
}
.cmp-v { color: var(--muted); min-width: 11em; }

/* ---------- comparables summary ---------- */
.comps-sum { margin: 10px 0 6px; }
.comps-hd {
  font-size: 10.5px; text-transform: uppercase; letter-spacing: .09em;
  color: var(--faint); margin-bottom: 5px;
}
.sum-table {
  border-collapse: collapse;
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 12.5px; font-variant-numeric: tabular-nums;
}
.sum-table th {
  text-align: left; padding: 3px 16px 3px 0;
  font-family: "IBM Plex Sans", ui-sans-serif, sans-serif;
  font-size: 11px; font-weight: 600; color: var(--muted);
  white-space: nowrap;
}
.sum-table td { padding: 3px 0; }
.sum-table td.sd { color: var(--muted); padding-right: 8px; white-space: nowrap; }
.sum-table td.rt {
  color: var(--ink); font-weight: 600; padding-right: 22px;
  text-align: right; width: 3.4em;
}
.sum-table td.ct { color: var(--faint); font-size: 11px; white-space: nowrap; }
.magnitude {
  margin-top: 7px; font-size: 12.5px; color: var(--muted);
  line-height: 1.5; max-width: 74ch;
}
.magnitude b {
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
def _forecast(g: dict) -> str:
    """The projected result, which is what this tool actually does well."""
    f = g.get("forecast") or {}
    if not f:
        return ""

    home, away = g["home_team"], g["away_team"]
    hp, ap = f.get("home_points"), f.get("away_points")
    prob = f.get("home_win_prob")
    home_pct = int(round((prob or 0.5) * 100))

    if hp is None or ap is None:
        line = ""
    elif hp >= ap:
        line = f'<span class="pick">{_e(home)} <span class="num">{hp}–{ap}</span></span>'
    else:
        line = f'<span class="pick">{_e(away)} <span class="num">{ap}–{hp}</span></span>'

    fav = home if home_pct >= 50 else away
    conf = home_pct if home_pct >= 50 else 100 - home_pct
    prob_txt = (f'<div class="because">{_e(fav)} to win — '
                f'<b>{conf}%</b></div>')

    # Colour the favoured side, not the home side - green on the 19% end
    # reads as an endorsement of the wrong team.
    home_cls = "lead" if home_pct >= 50 else "trail"
    away_cls = "trail" if home_pct >= 50 else "lead"
    bar = (f'<div class="prob-bar">'
           f'<i class="{away_cls}" style="width:{100 - home_pct}%"></i>'
           f'<i class="{home_cls}" style="width:{home_pct}%"></i></div>'
           f'<div class="prob-ends">'
           f'<span>{_e(away)} {100 - home_pct}%</span>'
           f'<span>{_e(home)} {home_pct}%</span></div>')

    return f'<div class="verdict">{line}</div>{prob_txt}{bar}'


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
    home, away = g["home_team"], g["away_team"]

    # Never a signed number. A minus sign means "the home team lost by" here,
    # and "is favoured by" on the market row three lines below - the same
    # symbol with opposite meanings. And when the home team happens to be the
    # favourite, as it usually is, the two readings agree and the ambiguity
    # stays invisible until the one game where they don't.
    if p25 >= 0 and p75 >= 0:
        label = (f'Half finished with <b>{_e(home)}</b> winning by '
                 f'<b>{p25:.0f} to {p75:.0f}</b>')
    elif p25 < 0 and p75 < 0:
        label = (f'Half finished with <b>{_e(away)}</b> winning by '
                 f'<b>{abs(p75):.0f} to {abs(p25):.0f}</b>')
    else:
        label = (f'Half finished from <b>{_e(away)} winning by '
                 f'{abs(p25):.0f}</b> to <b>{_e(home)} winning by '
                 f'{p75:.0f}</b>')

    # End labels, so the direction of the bar is readable without the sentence.
    ends = (f'<div class="band-ends">'
            f'<span>&#8592; {_e(away)} wins</span>'
            f'<span>{_e(home)} wins &#8594;</span></div>')

    return (
        '<div class="band-row">'
        f'<div class="band-wrap">'
        f'<div class="band" title="Middle half of comparable final margins">'
        f'<div class="track"></div>'
        f'<div class="iqr" style="left:{a:.1f}%;width:{max(b - a, 1.0):.1f}%"></div>'
        f'<div class="zero" style="left:{pos(0.0):.1f}%"></div>'
        f'<div class="med" style="left:{pos(med):.1f}%"></div></div>'
        f'{ends}</div>'
        f'<div class="band-label">{label}</div>'
        '</div>')


def _comps_table(g: dict) -> str:
    """What all the comparable games did, on each of the three markets.

    The five precedents below are the closest, not a summary - this is the
    summary. Rates are of graded games, so pushes are excluded rather than
    counted as losses, which is why the counts differ slightly per row.
    """
    c = g.get("comps") or {}
    n = c.get("n")
    if not n:
        return ""

    home, away = g["home_team"], g["away_team"]
    rows = []

    cover = c.get("cover_rate")
    if cover is not None and g.get("market_spread") is not None:
        hs, as_ = g["market_spread"], -g["market_spread"]
        rows.append(("Spread",
                     f'{_e(home)} {hs:+.1f}', _pct(cover),
                     f'{_e(away)} {as_:+.1f}', _pct(1 - cover),
                     c.get("cover_n")))

    over = c.get("over_rate")
    if over is not None and g.get("market_total") is not None:
        t = g["market_total"]
        rows.append(("Total", f'Over {t:.1f}', _pct(over),
                     f'Under {t:.1f}', _pct(1 - over), c.get("over_n")))

    win = c.get("home_win_rate")
    if win is not None:
        rows.append(("Moneyline", f'{_e(home)} win', _pct(win),
                     f'{_e(away)} win', _pct(1 - win), n))

    if not rows:
        return ""

    body = "".join(
        f'<tr><th>{_e(k)}</th>'
        f'<td class="sd">{a}</td><td class="rt">{ar}</td>'
        f'<td class="sd">{b}</td><td class="rt">{br}</td>'
        f'<td class="ct">{"" if cnt is None else f"of {cnt}"}</td></tr>'
        for k, a, ar, b, br, cnt in rows)

    return ('<div class="comps-sum">'
            f'<div class="comps-hd">Across all {n} comparable games</div>'
            f'<table class="sum-table">{body}</table>'
            f'{_magnitude(g)}</div>')


def _magnitude(g: dict) -> str:
    """Size of the outcomes, not just how often they happened.

    Two matchups can both cover 52% of the time while one wins narrowly and
    the other occasionally runs away with it. The cover rate is blind to that
    difference; this is where it shows up.
    """
    c = g.get("comps") or {}
    home, away = g["home_team"], g["away_team"]
    lines = []

    hb, ab = c.get("home_cover_by"), c.get("away_cover_by")
    if hb is not None and ab is not None:
        # Only name a side when the gap is big enough to mean something.
        # Calling 11.7 against 11.5 an advantage is reporting noise.
        if abs(hb - ab) >= 1.5:
            verdict = (f' — the bigger outcomes belong to '
                       f'<b>{_e(home if hb > ab else away)}</b>.')
        else:
            verdict = " — neither side's wins were notably larger."
        lines.append(
            f'Covers came by <b>{hb:.1f}</b> for {_e(home)} and '
            f'<b>{ab:.1f}</b> for {_e(away)} on average{verdict}')

    hbl, abl = c.get("home_blowout"), c.get("away_blowout")
    if hbl is not None and abl is not None:
        lines.append(
            f'Won by 14 or more: {_e(home)} <b>{_pct(hbl)}</b>, '
            f'{_e(away)} <b>{_pct(abl)}</b>.')

    if not lines:
        return ""
    return '<div class="magnitude">' + " ".join(lines) + '</div>' 


def _market_line(g: dict) -> str:
    """Market and model side by side, as orientation rather than advice.

    The difference between them predicted nothing across 5,000+ out-of-sample
    games, so it is shown plainly and left alone.
    """
    f = g.get("forecast") or {}
    rows = []

    if g.get("market_spread") is not None:
        rows.append(("Market",
                     f'{_e(g["home_team"])} {g["market_spread"]:+.1f}',
                     f'total {g["market_total"]:.1f}'
                     if g.get("market_total") is not None else ""))
    if f.get("margin") is not None:
        m = f["margin"]
        rows.append(("Model",
                     f'{_e(g["home_team"])} {-m:+.1f}',
                     f'total {f["total"]:.1f}' if f.get("total") else ""))

    if not rows:
        return '<div class="market">No market line available</div>'

    body = "".join(
        f'<div class="cmp-row"><span class="cmp-k">{_e(k)}</span>'
        f'<span class="cmp-v">{a}</span><span class="cmp-v">{b}</span></div>'
        for k, a, b in rows)
    return f'<div class="market">{body}</div>'


def _totals_line(g: dict) -> str:
    return ""


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
    # No side is being recommended, so results are reported, not scored.
    side = None
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

    # The five nearest can easily lean one way while the full set does not -
    # five games is a tiny sample, and they were picked for being closest, not
    # for being representative. Say so with numbers rather than leaving the
    # reader to wonder why the table contradicts the headline.
    shown = [e for e in rows[:5] if e.get("home_covered") is not None]
    shown_cov = sum(1 for e in shown if e["home_covered"])
    full_rate = (g.get("comps") or {}).get("cover_rate")
    contrast = ""
    if shown and full_rate is not None:
        contrast = (f' In these five, {_e(home)} covered '
                    f'<b>{shown_cov} of {len(shown)}</b>; across all '
                    f'{(g.get("comps") or {}).get("n")} comparable games it was '
                    f'<b>{_pct(full_rate)}</b> — the five are the closest, '
                    f'not a summary.')

    key = (f'<p class="prec-key">In each of these the <b>home team stands in '
           f'for {_e(home)}</b> and the visitor stands in for {_e(away)}. '
           f'The line shown is the home team’s.'
           + (' Green means the game went the way this pick needs.'
              if side else '')
           + contrast
           + '</p>')

    label = ("Five closest precedents" if not total_n
             else f"Five closest of {total_n} comparable games")

    return ('<details class="prec">'
            f'<summary>{_e(label)}</summary>{key}'
            f'<div class="prec-wrap"><table class="prec-table">'
            f'{header}{"".join(body)}</table></div></details>')


def _flags(g: dict) -> str:
    flags = g.get("confidence_flags") or []
    if not flags:
        return ""
    items = "".join(f"<span><i>▲</i> {_e(f)}</span>" for f in flags[:3])
    return f'<div class="flags"><span class="lead">Watch</span>{items}</div>'


def _game_row(g: dict) -> str:
    # Highlight the genuinely uncertain games - those are the interesting ones
    # to watch, and it is a claim the model can actually support.
    prob = ((g.get("forecast") or {}).get("home_win_prob") or 0.5)
    close = abs(prob - 0.5) <= 0.10
    meta = []
    if g.get("kickoff_et"):
        meta.append(_e(g["kickoff_et"]))
    if g.get("neutral_site"):
        meta.append("Neutral site")

    return f"""
    <article class="game{' close' if close else ''}">
      <div class="head">
        <span class="teams">{_e(g["away_team"])} <span class="at">at</span> {_e(g["home_team"])}</span>
        <span class="when">{" · ".join(meta)}</span>
        <span class="wx">{_e(g.get("weather_text", ""))}</span>
      </div>
      {_forecast(g)}
      {_band(g)}
      {_comps_table(g)}
      {_market_line(g)}
      {_totals_line(g)}
      {_flags(g)}
      {_precedents(g)}
    </article>"""


# ---------------------------------------------------------------- page parts
def _summary_strip(payload: dict) -> str:
    games = payload["games"]
    cells = [("Games", str(len(games)))]

    close = [g for g in games
             if abs(((g.get("forecast") or {}).get("home_win_prob") or .5) - .5) <= 0.10]
    cells.append(("Toss-ups", str(len(close))))

    m = payload.get("model_metrics") or {}
    if m.get("margin_mae"):
        cells.append(("Model error", f'{m["margin_mae"]:.1f} pts'))
    if m.get("market_margin_mae"):
        cells.append(("Market error", f'{m["market_margin_mae"]:.1f} pts'))
    if m.get("win_accuracy"):
        cells.append(("Winners called", f'{m["win_accuracy"] * 100:.0f}%'))

    inner = "".join(f"<div><dt>{_e(k)}</dt><dd>{_e(v)}</dd></div>" for k, v in cells)
    return f'<dl class="strip">{inner}</dl>'


def _lede(payload: dict) -> str:
    m = payload.get("model_metrics") or {}
    acc = (f' Across seasons it had never seen, it called '
           f'<b>{m["win_accuracy"] * 100:.0f}%</b> of winners correctly.'
           if m.get("win_accuracy") else "")
    return (f'<p class="lede">A projected score and win probability for every '
            f'game, with the outcomes of the most similar matchups of the last '
            f'decade for context.{acc} '
            f'No picks: the model does not beat the closing line, and the page '
            f'says so rather than pretending otherwise.</p>')


def _honesty_note(payload: dict) -> str:
    m = payload.get("model_metrics") or {}
    cc = m.get("comps_calibration") or {}
    bits = []

    if m.get("margin_mae") and m.get("market_margin_mae"):
        gap = m["margin_mae"] - m["market_margin_mae"]
        bits.append(
            f"<strong>On accuracy:</strong> out of sample the model's margin "
            f"error is {m['margin_mae']:.1f} points against the closing line's "
            f"{m['market_margin_mae']:.1f} — it trails the market by "
            f"{abs(gap):.1f}.")

    if cc.get("n_fitted"):
        bits.append(
            f"<strong>On picks:</strong> across {cc['n_fitted']:,} games the "
            f"model had never seen, neither its disagreement with the line nor "
            f"the comparables' cover rate predicted covering — every band came "
            f"back at roughly 50%. So this page makes no recommendations.")

    bits.append("Ratings, efficiency and comparables are all built only from "
                "games played before the one being forecast, and the betting "
                "line is never an input to the model itself.")
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

    if games:
        body = '<h2 class="sec">The slate, by kickoff</h2>'
        body += "".join(_game_row(g) for g in games)
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
