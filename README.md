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
   Exact-duplicate reads are dropped; only distinct reads are fitted.
2. **Players P(t).** `log P` fit against `ln(1 + hours since launch)` over
   the last 24h of distinct reads; growth bounded to a doubling per day.
3. **Post-refresh reads only.** The launch-night reads are a different
   regime (638 mostly hardcore early-access players, rates 0.63–2.04% —
   far above the refreshed population's). Fitting through them flattens
   every slope. All fits use only reads with players ≥ 10% of the last
   read's count, within 48h of the cutoff.
4. **Earners E_i(t).** Per trophy, `log(E + 0.5)` fit against
   `ln(1 + hours)`, recency-weighted; the slope is shrunk toward the
   players slope plus a category adjustment (story trophies catch up
   faster; grinds grow slower). Extrapolation is **logistically damped**
   toward a category day-7 cap (S-curve saturation), floored at the last
   observed earners.
5. **Level prior.** When few earners were ever observed, the median is
   blended in logit space toward the category's day-7 prior rate
   (calibrated from GTA V lifetime rarities scaled to one week, and from
   the real day-1 Exophase rates). The prologue is set to 99.9%
   structurally.
6. **Spread.** Horizon term + fit residual + small-count + cap proximity
   + prior reliance, combined in quadrature.
7. **Stale history.** With fewer than 3 distinct reads there is no trend
   to fit, so the program expects a refresh: non-prologue rates cut to
   ~30% of the last read (the observed first refresh multiplied players
   ×6.1 while non-prologue earners only ×1.3, so rates fell ~78%), with
   a wide spread.

## Local validation

- **Practice cases** (vs answers estimated from the spec's refresh facts:
  3,868 players, 4,276 trophies earned): pinball loss **0.34** vs the
  persistence baseline's **1.41**.
- **Synthetic final-like case** (realistic 2-day history, 4.5-day horizon,
  S-curve progression, step-function refreshes): loss **0.34** vs the
  trend baseline's **0.62** and persistence's **1.41**. A second,
  harsher simulation (faster player growth, lower day-7 rates, steeper
  curves): **0.51** vs trend **0.88** and persistence **2.27**.
- Output format validated (55 rows, 5 non-decreasing quantiles each,
  ≤ 64 KiB); two runs byte-identical; runtime ~0.1–0.3 s per case
  (8 s cap); edge cases pass (single read, two reads, null rates,
  90-minute horizon, all-zero rates).

## Run locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## License

MIT (see LICENSE).
