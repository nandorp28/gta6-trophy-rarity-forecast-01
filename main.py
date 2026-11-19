"""Forecast GTA VI's PS5 trophy earn rates a week after launch (make.fun).

Method: per-trophy logit drift with band-pooled shrinkage and damping.

An earn rate is a share of the players Exophase tracks, and launch week
it moves like a saturating curve: fast while the first wave plays through,
then flattening as the tracked base matures. In log time,
x = ln(1 + hours since release), that curve is close to a straight line,
so each trophy gets an OLS fit of logit(rate) on x over the case's
distinct reads (consecutive identical reads collapsed: Exophase refreshes
in jumps, so a flat stretch is one point, not many):

  * Window: the last 48 hours of reads. Launch night's explosion is a
    different regime from the steady week-one climb, and the house curve
    baseline's last-24h window is the same instinct; 48h keeps more
    refreshes while still dropping the launch-night step.
  * Slope shrinkage: a trophy's own slope is shrunk toward the pooled
    slope of trophies in its current-rate band (trophies at a similar rate
    today are at a similar point of the game, so their earners grow alike).
    Rare trophies have a handful of earners, so their own slopes are
    mostly noise; the shrink weight grows with the trophy's distinct
    reads. This is the same family as the shipped
    gta6-twitch-ccu-forecast-01 entry (per-unit OLS in log space, slope
    shrinkage, floored spread), adapted to earn rates: here the "analogs"
    are the game's own trophies, grouped by rate, because the case
    carries no cross-game comparables, only this game's own reads.
  * Damping: none. A saturating curve is concave in log time in principle,
    so a straight line fitted over the window could overstate the slope at
    the target — but in week one these curves are far from saturation (rates
    0.1-20% against eventual shares of 10-50%), and damping was tested at
    0-0.4/day on a cohort simulator calibrated to the real launch-night
    refresh and the live page: undamped was best by a clear margin (mean
    loss 0.10 vs 0.15 at 0.15/day), so the drift is extrapolated whole.
    The median is anchored on the last read's rate, not on the fitted
    line's level, so a stale last read still moves with the fitted drift.
  * Spread: the fit's residual scatter (refresh jumps included) plus the
    slope's standard error times the distance to the target, plus a floor
    that grows with the horizon. On launch night the page sat unchanged
    for hours and then one refresh cut every rate several-fold, so a
    history of identical reads earns a wide band, not a point: with no
    distinct reads the forecast is persistence with a 0.9-logit spread.

Fallbacks keep every case valid: a trophy with too few distinct reads
takes the pooled slope at its last rate; a case with no usable reads at
all forecasts persistence. Everything is deterministic: no randomness, no
clock, no network, stdlib only.
"""

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Rate bands (percent) for pooling drift slopes: trophies at a similar
# rate today are at a similar point of the game.
BAND_EDGES = [0.0, 0.5, 1.0, 2.0, 100.0]
# Fit window: only reads this recent (hours before the checkpoint).
WINDOW_HOURS = 48.0
# Shrinkage strength: a trophy's own slope counts as this many extra reads.
SHRINK_K = 4.0
# Damping of the extrapolated drift per day of horizon. Set to 0 after
# testing 0-0.4/day on the calibrated cohort simulator (undamped best,
# mean loss 0.10 vs 0.15 at 0.15/day): in week one the curves are far
# from saturation, so damping is pure bias. Kept as a named constant so
# the choice is visible, not buried.
DAMP_PER_DAY = 0.0
# Narrowest logit spread any trophy is allowed.
SIGMA_FLOOR = 0.35
# Spread when the history never moved: persistence with a wide band, so a
# refresh the case has not seen yet still lands inside it.
FLAT_SIGMA = 0.90
# Extra spread per sqrt-day of horizon, for dynamics the fits cannot see.
HORIZON_SIGMA = 0.10


def _parse_ts(ts):
    """'2026-11-19T03:00:00Z' -> epoch seconds, or None."""
    if not isinstance(ts, str):
        return None
    try:
        from datetime import datetime, timezone
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def _ols(xs, ys):
    """Least squares y = a + b*x. Returns (a, b, resid_sd, se_b, x_mean)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx > 0 else 0.0
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    var = sum(r * r for r in resid) / max(n - 2, 1) if n > 2 else 0.0
    se_b = math.sqrt(var / sxx) if sxx > 0 and n > 2 else 0.0
    return a, b, math.sqrt(var), se_b, mx


def _median(values):
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _band(rate):
    for i in range(len(BAND_EDGES) - 1):
        if BAND_EDGES[i] <= rate < BAND_EDGES[i + 1]:
            return i
    return len(BAND_EDGES) - 2


def _logit(pct):
    p = min(99.99, max(0.01, pct)) / 100.0
    return math.log(p / (1.0 - p))


def _inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-max(-30.0, min(30.0, lg))))


def _round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def distinct_reads(case):
    """Collapse consecutive identical reads: a flat stretch is one point."""
    out = []
    prev = None
    for read in case.get("history") or []:
        key = (read.get("players_tracked"),
               tuple(read.get("rates") or []))
        if key != prev:
            out.append(read)
            prev = key
    return out


def forecast_case(case):
    trophies = case.get("trophies") or []
    history = distinct_reads(case)
    horizon_days = (case.get("horizon_minutes") or 0) / 1440.0

    release = _parse_ts((case.get("game") or {}).get("release_date")
                        and (case["game"]["release_date"] + "T00:00:00Z"))
    if release is None and history:
        release = _parse_ts(history[0].get("slot")) or 0.0
    if release is None:
        release = 0.0
    target_ts = _parse_ts(case.get("target"))
    if target_ts is None:
        target_ts = release
    x_target = math.log1p(max(0.0, (target_ts - release) / 3600.0))
    at_ts = _parse_ts(case.get("at"))
    if at_ts is None and history:
        at_ts = _parse_ts(history[-1].get("slot") or history[-1].get("at"))
    if at_ts is None:
        at_ts = target_ts

    last = (case.get("history") or [None])[-1]
    last_rates = (last or {}).get("rates") or []
    last_n = last.get("players_tracked") if last else None
    last_ts = _parse_ts(last.get("slot") or last.get("at")) if last else None
    x_last = (math.log1p(max(0.0, (last_ts - release) / 3600.0))
              if last_ts is not None else 0.0)

    # Reads inside the fit window (by slot time); all history if the
    # window holds too few distinct reads to fit on.
    window_start = at_ts - WINDOW_HOURS * 3600.0
    window = [r for r in history
              if (_parse_ts(r.get("slot") or r.get("at")) or 0.0)
              >= window_start]
    if len(window) < 3:
        window = history

    def x_of(read):
        ts = _parse_ts(read.get("slot") or read.get("at"))
        if ts is None:
            return None
        return math.log1p(max(0.0, (ts - release) / 3600.0))

    # --- Per-trophy logit fits, then pool slopes by rate band. ---
    per_trophy = []
    band_slopes = {}
    for i, trophy in enumerate(trophies):
        xs, ys = [], []
        for read in window:
            rates = read.get("rates") or []
            x = x_of(read)
            if i >= len(rates) or rates[i] is None or x is None:
                continue
            xs.append(x)
            ys.append(_logit(rates[i]))
        fit = _ols(xs, ys) if len(xs) >= 2 and len(set(xs)) > 1 else None
        last_rate = (last_rates[i] if i < len(last_rates)
                     and isinstance(last_rates[i], (int, float))
                     else 0.0)
        last_e = (max(last_rate * last_n / 100.0, 0.25)
                  if isinstance(last_n, (int, float)) and last_n > 0 else 1.0)
        per_trophy.append({
            "id": trophy["id"], "fit": fit, "npts": len(xs),
            "last_rate": last_rate, "last_e": last_e,
            "band": _band(last_rate),
        })
        if fit is not None:
            band_slopes.setdefault(_band(last_rate), []).append(
                (fit[1], last_e))

    pooled = {}
    for band, slopes in band_slopes.items():
        wsum = sum(w for _, w in slopes)
        pooled[band] = (sum(s * w for s, w in slopes) / wsum
                        if wsum > 0 else _median([s for s, _ in slopes]))

    all_slopes = [(s, w) for slopes in band_slopes.values()
                  for s, w in slopes]
    if all_slopes:
        wsum = sum(w for _, w in all_slopes)
        global_pool = sum(s * w for s, w in all_slopes) / wsum
    else:
        global_pool = 0.0

    # Damping: the further the target, the more the fitted drift is
    # pulled back toward persistence.
    damp = 1.0 / (1.0 + DAMP_PER_DAY * horizon_days)

    rows = []
    for t in per_trophy:
        band_pool = pooled.get(t["band"], global_pool)
        lg_last = _logit(t["last_rate"])
        if t["fit"] is not None:
            _, b, resid, se_b, x_mean = t["fit"]
            w = t["npts"] / (t["npts"] + SHRINK_K)
            b_star = w * b + (1.0 - w) * band_pool
            drift = damp * b_star * (x_target - x_last)
            sigma = math.sqrt(resid ** 2
                              + (se_b * abs(x_target - x_mean)) ** 2)
        else:
            drift = damp * band_pool * max(0.0, x_target - x_last)
            sigma = FLAT_SIGMA
        sigma = max(SIGMA_FLOOR,
                    math.sqrt(sigma ** 2
                              + (HORIZON_SIGMA * math.sqrt(horizon_days)) ** 2))
        qs = [_round4(min(100.0, max(0.0, _inv_logit(lg_last + drift + z * sigma))))
              for z in Z]
        qs = [max(qs[i], qs[i - 1] if i else 0.0)
              for i in range(len(qs))]
        rows.append({"id": t["id"], "q": qs})

    if not rows:  # degenerate case: no trophies listed at all
        rows = [{"id": 1, "q": [0.01, 0.01, 0.01, 0.01, 0.01]}]
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
