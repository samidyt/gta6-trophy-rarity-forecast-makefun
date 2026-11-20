# GTA VI Trophy Rarity Forecaster

Entry for the [make.fun bounty "Forecast how many GTA VI players earn each
trophy in week one"](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
(`gta6-trophy-rarity-forecast-01`).

`main.py` reads a case (`/input/<id>.in`: the game's 55 PS5 trophies plus
the house's recording of Exophase's trophy page up to the checkpoint) and
writes `/output/<id>.out`: five quantiles (5/25/50/75/95%) of each trophy's
earn rate at the target, scored by pinball loss on the logit scale. Python
3.13, standard library only, offline, deterministic.

## Model: earners / players decomposition with tier priors

A trophy's earn rate is `r = E / P`, earners over players tracked. During a
launch week both move fast for different reasons, so the program models
them separately and divides at the target instead of extrapolating the
ratio directly (the house trend baseline extrapolates the ratio, which
conflates player progression with denominator dilution).

1. **Clean.** Exophase refreshes on its own schedule (hours apart), so the
   reads are a step function: long flat stretches punctuated by jumps.
   Consecutive duplicate reads are dropped; only distinct reads are fitted.
2. **Players P(t).** `log P` fit against `ln(1 + hours since launch)` over
   the last 24h of distinct reads, with a concave quadratic term — launch-
   week player growth decelerates, and a straight line overshoots a
   saturating curve. Growth bounded to a doubling per day.
3. **Post-refresh reads only.** The launch-night reads are a different
   regime (638 mostly hardcore early-access players, rates far above the
   refreshed population's). Fitting through them flattens every slope. All
   fits use only reads with players ≥ 10% of the last read's count, within
   48h of the cutoff.
4. **Earners E_i(t).** Per trophy, `log(E + 0.5)` fit against
   `ln(1 + hours)` with a concave quadratic term (progression saturates
   too), recency-weighted; the effective slope at the last read is shrunk
   toward the players slope plus a category adjustment (story trophies
   catch up faster; grinds grow slower). Extrapolation is **logistically
   damped** toward a category day-7 cap (tanh soft cap: S-curve
   saturation), floored at the last observed earners.
5. **Level prior.** When a trophy is early in its expected journey — few
   earners relative to what its category's day-7 prior implies at the
   target player count — the median is blended in logit space toward the
   category's day-7 prior rate (calibrated from GTA V lifetime rarities
   scaled to one week, and from the real day-1 Exophase rates). The
   prologue is set to 99.9% structurally.
6. **Spread.** Horizon term + fit residual + small-count + cap proximity
   + prior reliance, combined in quadrature.
7. **Stale history.** With fewer than 3 distinct reads there is no trend
   to fit, so the program expects a refresh: non-prologue rates cut to
   22% of the last read (the spec's published first refresh multiplied
   players ×6.1 while non-prologue earners only ×1.34, so rates fell to
   ~22%), with a wide spread. With no history at all, category priors
   with a wide spread.

## Local validation

All numbers below were measured with a self-built harness (pinball scorer
verified against the spec's worked example, 0.403404 exactly).

- **Practice cases** (vs answers estimated from the spec's refresh facts:
  3,868 players, 4,276 trophies earned; non-prologue earners scaled
  ×408/305 uniformly, prologue kept at 100%): pinball loss **0.22** vs
  the persistence baseline's **1.40** (trend baseline also 1.40: every
  pre-refresh read is flat, so its line is flat).
- **Synthetic final-like cases** (realistic 2.5-day history, 4.5-day
  horizon, S-curve progression, step-function refreshes, three random
  seeds): loss **0.21–0.27** (scenario 1) and **0.26–0.34** (scenario 2:
  faster player growth, lower day-7 rates, steeper curves), vs the trend
  baseline's **0.97–1.18** and persistence's **0.83–0.91**.
- Output format validated (55 rows, 5 non-decreasing quantiles each,
  ≤ 64 KiB); two runs byte-identical; runtime ~0.05–0.20 s per case
  (8 s cap); edge cases pass (single read, two reads, null rates,
  10-minute horizon, all-zero rates, no history).

## Run locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## License

MIT (see LICENSE).
