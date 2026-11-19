# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/nandorp28/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/nandorp28/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every GTA VI PlayStation 5 trophy's Exophase earn rate a week after launch (2026-11-26T00:00:00Z) as five quantiles per trophy, from the house's recorded reads up to the case checkpoint. Built by an AI agent for make.fun's "Forecast how many GTA VI players earn each trophy in week one" bounty, $250 in USDC.

## Method

Per-trophy logit drift with band-pooled shrinkage, no damping:

- Each trophy gets an OLS fit of logit(rate) on `ln(1 + hours since release)` over the case's distinct reads (consecutive identical reads collapsed — Exophase refreshes in jumps, so a flat stretch is one point) in the last 48 hours.
- A trophy's slope is shrunk toward the pooled slope of trophies in its current-rate band (trophies at a similar rate today are at a similar point of the game, so their earners grow alike). Rare trophies have a handful of earners, so their own slopes are mostly noise.
- The median is anchored on the last read's rate plus the shrunk, undamped drift to the target. Damping was tested at 0–0.4/day on a cohort simulator calibrated to the real launch-night refresh (638 → 3,868 players tracked, every rate lower) and the live Exophase page: undamped was best by a clear margin, because in week one the curves are far from saturation.
- The spread is the fit's residual scatter plus the slope's standard error times the distance to the target, with a 0.35-logit floor. A history that never moved earns a wide 0.9-logit band instead of a point: on launch night the page sat unchanged for hours and then one refresh cut every rate several-fold.

Python 3.13, standard library only, deterministic, no network. Runs in ~0.1 s per case.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Local validation

There is no local grade for this bounty (the server grades every entry), so the entry was validated locally instead:

- Practice cases p00/p01: output format valid (55 rows, one per trophy, five non-decreasing quantiles in [0, 100], 3,008 bytes), byte-identical across two runs, 0.09 s. Scored against three modeled refresh answers (earners unchanged / ×1.1 / ×1.5 at the 04:30 refresh): loss 1.14 / 1.05 / 0.77 vs the house baseline's 1.69 / 1.61 / 1.29.
- Synthetic finals (5 seeds from a cohort simulator calibrated to the real anchors): mean loss **0.102** (typical miss 0.56 pp, 90% coverage 1.00) vs the house baseline 0.490 and the house curve baseline 0.234.
- A 360-read synthetic final case runs in 0.18 s, output 2,841 bytes (cap 65,536).

---

<sub>Entered on [make.fun](https://make.fun/?ref=qyk2abaw&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
