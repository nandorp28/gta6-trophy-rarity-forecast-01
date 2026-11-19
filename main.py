"""Forecast GTA VI PS5 trophy earn rates one week after launch (make.fun).

Each trophy's earn rate is the share of Exophase-tracked players who hold
it. Over launch week that share climbs like a saturating curve, and in log
time, x = ln(1 + hours since release), the climb is close to a straight
line on the logit scale. So for every trophy this program fits, by least
squares, logit(rate) against x over the case's recent reads, then reads the
fitted drift forward to the target:

  * Distinct reads only. Exophase refreshes its counts in jumps, so a
    stretch of identical reads is one observation, not many: consecutive
    reads with the same players-tracked count and the same rates collapse
    into one point before any fitting.
  * Window: reads from the last 48 hours before the checkpoint. Launch
    night's explosion is a different regime from the steady week-one
    climb; 48 hours keeps several refreshes while dropping that step.
    If the window holds fewer than three distinct reads, the whole
    history is used instead.
  * Slope shrinkage. A trophy's own slope is noisy when it has few
    earners, so it is shrunk toward the pooled slope of the trophies in
    its current-rate band (trophies at a similar rate today are at a
    similar point of the game, so their earners grow alike). The pooled
    slope weights each trophy by its current earner count, and the shrink
    weight grows with the trophy's own number of distinct reads.
  * No damping. In week one the curves sit far from saturation (rates of
    0.1-20% against eventual shares of 10-50%), so the fitted drift is
    extrapolated whole. A cohort simulator calibrated to the real
    launch-night refresh and the live page showed undamped clearly best
    (mean loss 0.10 vs 0.15 at 0.15/day damping).
  * Median anchored on the last read. The forecast's centre is the last
    read's rate moved by the shrunk drift, not the fitted line's level,
    so a stale last read still moves with the fitted drift.
  * Spread. The fit's residual scatter plus the slope's standard error
    times the distance from the fit's centre to the target, plus a term
    that grows with the horizon, floored at 0.35 logit. A history that
    never moved gets a wide 0.9-logit band: a refresh the case has not
    seen yet still lands inside it.

Fallbacks keep every case valid: a trophy with too few distinct reads to
fit takes its band's pooled slope at its last rate; a case with no usable
reads at all forecasts persistence. Deterministic, stdlib only, no
network, no clock reads of its own: everything comes from the case file.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Rate bands (percent) for pooling drift slopes.
BAND_EDGES = [0.0, 0.5, 1.0, 2.0, 100.0]
# Fit window: only reads this recent (hours before the checkpoint).
WINDOW_HOURS = 48.0
# Shrinkage strength: a trophy's own slope counts as this many extra reads.
SHRINK_K = 4.0
# Damping of the extrapolated drift per day of horizon. 0 after testing
# 0-0.4/day on the calibrated cohort simulator (undamped best): in week
# one the curves are far from saturation, so damping is pure bias. Kept
# as a named constant so the choice is visible, not buried.
DAMP_PER_DAY = 0.0
# Narrowest logit spread any trophy is allowed.
SIGMA_FLOOR = 0.35
# Spread when the history never moved: persistence with a wide band.
FLAT_SIGMA = 0.90
# Extra spread per sqrt-day of horizon, for dynamics the fits cannot see.
HORIZON_SIGMA = 0.10


def _epoch(ts):
    """ISO '2026-11-19T03:00:00Z' -> epoch seconds, or None."""
    if not isinstance(ts, str):
        return None
    try:
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def _fit(xs, ys):
    """OLS y = a + b*x. Returns (intercept, slope, resid_sd, slope_se, x_bar)."""
    n = len(xs)
    x_bar = sum(xs) / n
    y_bar = sum(ys) / n
    sxx = sum((x - x_bar) ** 2 for x in xs)
    sxy = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx > 0 else 0.0
    intercept = y_bar - slope * x_bar
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    var = sum(r * r for r in resid) / max(n - 2, 1) if n > 2 else 0.0
    slope_se = math.sqrt(var / sxx) if sxx > 0 and n > 2 else 0.0
    return intercept, slope, math.sqrt(var), slope_se, x_bar


def _band_index(rate):
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
    """Round to 4 decimals, half up (matches the house baseline's rounding)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def _distinct(history):
    """Collapse consecutive identical reads into one point each."""
    points, prev_key = [], None
    for read in history or []:
        key = (read.get("players_tracked"),
               tuple(read.get("rates") or []))
        if key != prev_key:
            points.append(read)
            prev_key = key
    return points


def _weighted_mean(pairs):
    """Mean of values weighted by weights; plain median if weights sum to 0."""
    total = sum(w for _, w in pairs)
    if total > 0:
        return sum(v * w for v, w in pairs) / total
    vals = sorted(v for v, _ in pairs)
    n = len(vals)
    return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2.0


def forecast_case(case):
    trophies = case.get("trophies") or []
    history = _distinct(case.get("history") or [])
    horizon_days = (case.get("horizon_minutes") or 0) / 1440.0

    game = case.get("game") or {}
    release = _epoch(game.get("release_date") + "T00:00:00Z"
                     if game.get("release_date") else None)
    if release is None:
        release = _epoch(history[0].get("slot")) if history else 0.0
    if release is None:
        release = 0.0

    def log_hours(ts):
        if ts is None:
            return None
        return math.log1p(max(0.0, (ts - release) / 3600.0))

    target_ts = _epoch(case.get("target"))
    x_target = log_hours(target_ts) if target_ts is not None else 0.0

    at_ts = _epoch(case.get("at"))
    if at_ts is None and history:
        at_ts = _epoch(history[-1].get("slot") or history[-1].get("at"))
    if at_ts is None:
        at_ts = target_ts if target_ts is not None else 0.0

    last = (case.get("history") or [None])[-1] or {}
    last_rates = list(last.get("rates") or [])
    last_n = last.get("players_tracked")
    # Pad to the trophy count so a short last read does not silently
    # drop trophies from the anchor lookup below.
    if len(last_rates) < len(trophies):
        last_rates.extend([None] * (len(trophies) - len(last_rates)))
    # A trophy missing from the last read (null rate) anchors on its most
    # recent non-null rate instead of 0: the median is the last *seen*
    # rate moved by the drift, and 0.01% would be a false anchor.
    for i in range(len(trophies)):
        if i < len(last_rates) and last_rates[i] is None:
            for read in reversed(case.get("history") or []):
                rates = read.get("rates") or []
                if i < len(rates) and rates[i] is not None:
                    last_rates[i] = rates[i]
                    break
            else:
                last_rates[i] = 0.0
    x_last = log_hours(_epoch(last.get("slot") or last.get("at")))
    if x_last is None:
        x_last = 0.0

    # Reads inside the fit window; the whole history if the window holds
    # too few distinct reads to fit on.
    window_start = at_ts - WINDOW_HOURS * 3600.0
    window = [r for r in history
              if (_epoch(r.get("slot") or r.get("at")) or 0.0) >= window_start]
    if len(window) < 3:
        window = history

    # --- Per-trophy logit fits, then pool slopes by rate band. ---
    info = []
    band_members = {}
    for i, trophy in enumerate(trophies):
        xs, ys = [], []
        for read in window:
            rates = read.get("rates") or []
            if i >= len(rates) or rates[i] is None:
                continue
            x = log_hours(_epoch(read.get("slot") or read.get("at")))
            if x is None:
                continue
            xs.append(x)
            ys.append(_logit(rates[i]))
        fit = _fit(xs, ys) if len(xs) >= 2 and len(set(xs)) > 1 else None
        last_rate = (last_rates[i] if i < len(last_rates)
                     and isinstance(last_rates[i], (int, float)) else 0.0)
        # Current earners weight this trophy in its band's pooled slope.
        earners = (max(last_rate * last_n / 100.0, 0.25)
                   if isinstance(last_n, (int, float)) and last_n > 0 else 1.0)
        band = _band_index(last_rate)
        info.append({"id": trophy["id"], "fit": fit, "npts": len(xs),
                     "last_rate": last_rate, "band": band})
        if fit is not None:
            band_members.setdefault(band, []).append((fit[1], earners))

    pooled = {band: _weighted_mean(members)
              for band, members in band_members.items()}
    all_members = [m for members in band_members.values() for m in members]
    global_pool = _weighted_mean(all_members) if all_members else 0.0

    # Damping: the further the target, the more the fitted drift is pulled
    # back toward persistence. DAMP_PER_DAY is 0, so this is a no-op kept
    # visible; the drift below is extrapolated whole.
    damp = 1.0 / (1.0 + DAMP_PER_DAY * horizon_days)
    horizon_term = HORIZON_SIGMA * math.sqrt(horizon_days)

    rows = []
    for t in info:
        band_pool = pooled.get(t["band"], global_pool)
        lg_last = _logit(t["last_rate"])
        if t["fit"] is not None:
            _, slope, resid_sd, slope_se, x_bar = t["fit"]
            weight = t["npts"] / (t["npts"] + SHRINK_K)
            slope_star = weight * slope + (1.0 - weight) * band_pool
            drift = damp * slope_star * max(0.0, x_target - x_last)
            sigma = math.sqrt(resid_sd ** 2
                              + (slope_se * abs(x_target - x_bar)) ** 2)
        else:
            drift = damp * band_pool * max(0.0, x_target - x_last)
            sigma = FLAT_SIGMA
        sigma = max(SIGMA_FLOOR,
                    math.sqrt(sigma ** 2 + horizon_term ** 2))
        qs = [_round4(min(100.0, max(0.0, _inv_logit(lg_last + drift + z * sigma))))
              for z in Z]
        for j in range(1, len(qs)):
            qs[j] = max(qs[j], qs[j - 1])
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
