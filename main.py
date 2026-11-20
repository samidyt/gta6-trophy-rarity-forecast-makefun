"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Forecasts each trophy's Exophase earn rate at the case's target as five
quantiles (5/25/50/75/95%), scored by pinball loss on the logit scale.

MODEL -- earners / players decomposition with tier priors
----------------------------------------------------------
A trophy's earn rate is r = E / P, earners over players tracked. During a
launch week both move fast and for different reasons, so the program models
them separately and divides at the target instead of extrapolating the
ratio directly (the house trend baseline extrapolates the ratio, which
conflates progression with denominator dilution).

1. CLEAN. Exophase refreshes on its own schedule (hours apart), so the
   recorder's reads are a step function: long flat stretches punctuated by
   jumps. Exact-duplicate consecutive reads carry no information and are
   dropped; only distinct reads are fitted.

2. PLAYERS P(t). log P is fit against ln(1 + hours since launch) over the
   last 24 hours of distinct reads (all post-refresh, real data), by
   least squares. Extrapolated to the target, with growth bounded to at
   most a doubling per day (launch-week growth is fast but not infinite).

   FITS USE POST-REFRESH READS ONLY. The launch-night reads are a
   different regime: 638 mostly hardcore early-access players, whose
   rates (0.63-2.04%) are far above the refreshed population's. Fitting
   through them flattens every slope. So every fit uses only distinct
   reads with players >= 10% of the last read's count, within 48 hours
   of the cutoff (relaxed only when that leaves too few reads).

3. EARNERS E_i(t). For each trophy, log(E_i + 0.5) is fit against
   ln(1 + hours) on the distinct reads. The fitted slope is shrunk
   toward the players slope plus a category adjustment (story trophies
   catch up faster than the player base grows; grind/rare trophies grow
   slower), with shrinkage weight rising in the number of reads. The
   extrapolation is LOGISTIC-damped toward the cap: earners grow
   exponentially while far below it, then saturate as they approach
   P(target) * cap_i, where cap_i is a day-7 plausible-maximum rate
   for the trophy's category, calibrated from GTA V's lifetime
   rarities scaled down to one week of play (prologue ~73% lifetime
   -> ~99.9% week 1; story-complete ~17% -> ~8% week 1; 100% /
   platinum ~0.1-0.2% -> ~0.3% week 1). Earners are floored at the
   last observed count (earners never fall).

4. RATE. r_i = E_i(target) / P(target), clamped to [0.01, 99.99]. When
   few earners were ever observed (a rate printed as 0.00% hides
   whether 0 or 3 players earned it), the extrapolation says almost
   nothing about the LEVEL, so the median is blended in logit space
   toward the trophy category's day-7 prior rate, with weight on the
   data growing in the observed earner count. The prologue is set to
   99.9% structurally: everyone who plays earns it.

5. SPREAD. Quantiles are the median on the logit scale plus/minus
   z_tau * s_i, where s_i combines a horizon term (random-walk growth),
   the earners fit's residual spread, a small-count term (few earners =
   noisy), a cap-proximity term, and a prior-reliance term (the more
   the median leans on the prior, the wider the honest band). Honest,
   wider spreads for the trophies we know least about.

6. STALE HISTORY. When the history holds fewer than three distinct
   reads (launch night: every read identical), there is no trend to fit.
   The program then expects a refresh: non-prologue rates are cut to
   ~30% of their last value (the observed first refresh multiplied
   players x6.1 while non-prologue earners only x1.3, so rates fell
   ~78%), the prologue stays ~99.9%, and the spread is widened to cover
   both "refresh lands" and "no refresh yet".

Runs offline, standard library only, deterministic (no randomness).
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

# GTA VI release day, 2026-11-19T00:00:00Z: the clock every fit is measured on.
LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Base spread growth, logit units per sqrt(day). Slightly wider than the
# house baseline's 0.5: a 4.5-day launch-week horizon is genuinely uncertain.
SIGMA_BASE = 0.55
# Shrinkage strength for earners slopes: weight on the fitted slope is
# n / (n + SHRINK_K), n = number of distinct reads used.
SHRINK_K = 3.0
# Max player growth per day, as a multiple (a doubling a day, sustained,
# is already an extreme launch-week pace).
MAX_GROWTH_PER_DAY = 2.0
# Spread bounds, logit units.
SPREAD_MIN = 0.30
SPREAD_MAX = 2.50


def clamp(x: float, lo: float = 0.01, hi: float = 99.99) -> float:
    return min(hi, max(lo, x))


def logit(pct: float) -> float:
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """Round to 4 decimals, half up, as JavaScript's Math.round does."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_launch(t: datetime) -> float:
    return max(0.0, (t - LAUNCH).total_seconds() / 3600.0)


def ols(xs, ys, ws=None):
    """Weighted least squares y ~ a + b*x. Returns (a, b, resid_sd, n).

    With < 2 points or zero x-variance, returns the mean as intercept and
    slope 0. resid_sd is the residual standard deviation (0 when the fit
    is exact or n <= 2).
    """
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0
    if ws is None:
        ws = [1.0] * n
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 0:
        return my, 0.0, 0.0, n
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n <= 2:
        return a, b, 0.0, n
    sse = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys))
    dof = max(1, n - 2)
    return a, b, math.sqrt(sse / dof / (sw / n)), n


# ---------------------------------------------------------------------------
# Trophy categories: priors for the day-7 cap and the earners-slope adjustment.
# Calibrated from GTA V lifetime rarities (PocketPSN/PSNProfiles) scaled to
# one week of play, and from the observed day-1 Exophase rates.
# ---------------------------------------------------------------------------

def categorize(trophy):
    """Return (cap, slope_delta) for a trophy.

    cap: plausible-maximum week-1 earn rate as a fraction (0..1).
    slope_delta: added to the players slope as the earners-slope prior.
    """
    tier = trophy.get("tier", "")
    name = (trophy.get("name") or "").lower()
    desc = (trophy.get("description") or "").lower()

    if tier == "platinum":
        return 0.003, -0.50
    if "100%" in desc or "100% completion" in desc:
        return 0.003, -0.50
    if "gone fishing" in desc or "paradise" in name and "just like" in name:
        return 0.999, 0.0  # prologue: everyone who plays gets it
    if "complete chapter 1" in desc:
        return 0.60, 0.20
    if "complete chapter 2" in desc:
        return 0.35, 0.30
    if "complete chapter 3" in desc:
        return 0.25, 0.35
    if "complete chapter 4" in desc:
        return 0.18, 0.40
    if "final mission" in desc and tier == "gold":
        return 0.08, 0.40
    # Grinds: big counts, every-location / every-spot completions.
    if any(k in desc for k in ("250 stunts", "100 locations", "3 collectible",
                               "50 challenges", "every spot", "every location",
                               "all shooting range", "gold medal in all")):
        return 0.02, -0.50
    if "final mission for" in desc or "mission strands" in desc:
        return 0.08, 0.0
    if "explore all" in desc:
        return 0.04, -0.30
    if tier == "gold":
        return 0.04, -0.30
    if tier == "silver":
        return 0.08, -0.20
    return 0.20, 0.0  # misc bronze one-offs


def clean_history(case):
    """Drop exact-duplicate consecutive reads; parse times.

    Returns a list of (hours_since_launch, players, rates) with at least
    one entry, oldest first. A read is a duplicate when its players count
    and every rate equal the previous read's.
    """
    out = []
    prev_key = None
    for read in case["history"]:
        t = parse_time(read.get("slot") or read["at"])
        h = hours_since_launch(t)
        players = read.get("players_tracked")
        rates = list(read.get("rates") or [])
        key = (players, tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        out.append((h, players, rates))
    return out


def fit_reads(clean):
    """Distinct reads usable for fitting: post-refresh, real-sample regime.

    Drops reads with players < 10% of the last read's count (the stale
    launch-night regime and the first tiny refresh) and reads older than
    48 hours before the cutoff. Relaxes stepwise when that leaves fewer
    than 3 reads; always keeps the last read.
    """
    if not clean:
        return []
    last_h, last_p, _ = clean[-1]
    if not last_p or last_p <= 0:
        return clean
    floor = 0.10 * last_p
    reads = [(h, p, r) for (h, p, r) in clean
             if p and p >= floor and h >= last_h - 48.0]
    if len(reads) < 3:
        reads = [(h, p, r) for (h, p, r) in clean if p and p >= floor]
    if len(reads) < 3:
        reads = list(clean)
    return reads


def forecast_players(clean, target_hours, horizon_days):
    """Forecast players tracked at the target; returns (P_target, slope)."""
    if not clean:
        return 0.0, 0.0
    last_h, last_p, _ = clean[-1]
    if last_p is None or last_p <= 0:
        return 0.0, 0.0
    # Fit on the last 24 hours of fit reads (post-refresh, real data);
    # fall back to all fit reads when fewer than 3 sit in the window.
    usable = fit_reads(clean)
    window = [(h, p) for (h, p, _) in usable if h >= last_h - 24.0 and p and p > 0]
    if len(window) < 3:
        window = [(h, p) for (h, p, _) in usable if p and p > 0]
    xs = [math.log1p(h) for (h, _) in window]
    ys = [math.log(p) for (_, p) in window]
    # Recency weights: the latest read counts double.
    ws = [1.0 + (i / max(1, len(window) - 1)) for i in range(len(window))]
    a, b, _, _ = ols(xs, ys, ws)
    b = max(0.0, b)  # players tracked never shrinks in a launch week
    x_t = math.log1p(target_hours)
    log_p = a + b * x_t
    p_target = math.exp(log_p)
    # Bound growth: at most a doubling per day over the horizon.
    cap = last_p * (MAX_GROWTH_PER_DAY ** horizon_days)
    p_target = min(max(p_target, last_p), cap)
    return p_target, b


def level_prior(trophy):
    """Day-7 prior earn rate in percent, by trophy category.

    Used only when the observed earners are too few for the data-driven
    extrapolation to say anything (a rate printed as 0.00% hides whether
    0 or 3 players earned it). Calibrated from GTA V lifetime rarities
    scaled to one week and from the real day-1 Exophase rates
    (grinds already at 0.03-0.04% on day 1).
    """
    tier = trophy.get("tier", "")
    desc = (trophy.get("description") or "").lower()
    if tier == "platinum" or "100%" in desc:
        return 0.05
    if "gone fishing" in desc:
        return 99.9
    if "complete chapter 1" in desc:
        return 35.0
    if "complete chapter 2" in desc:
        return 18.0
    if "complete chapter 3" in desc:
        return 10.0
    if "complete chapter 4" in desc:
        return 6.0
    if "final mission" in desc and tier == "gold":
        return 3.0
    if any(k in desc for k in ("250 stunts", "100 locations", "3 collectible",
                               "50 challenges", "every spot", "every location",
                               "all shooting range", "gold medal in all")):
        return 0.06
    if "final mission for" in desc or "mission strands" in desc:
        return 1.0
    if "explore all" in desc:
        return 0.3
    if tier == "gold":
        return 0.15
    if tier == "silver":
        return 0.3
    return 3.0  # misc bronze one-offs


def extrapolate_earners(last_e, b, x_last, x_t, k_cap):
    """Extrapolate earners from x_last to x_t at log-growth rate b,
    logistically damped toward carrying capacity k_cap (S-curve
    saturation: exponential while far below the cap, flattening as it
    approaches it). Falls back to plain exponential when b <= 0 or the
    cap is not finite."""
    e_s = last_e + 0.5
    gap = max(0.0, x_t - x_last)
    if b > 0 and k_cap is not None and math.isfinite(k_cap) and k_cap > e_s:
        k_s = k_cap + 0.5
        ratio = (k_s - e_s) / e_s
        return k_s / (1.0 + ratio * math.exp(-b * gap)) - 0.5
    return math.exp(math.log(e_s) + b * gap) - 0.5


def forecast_trophy(idx, trophy, clean, p_target, b_players, target_hours,
                    horizon_days):
    """Forecast one trophy's five quantiles at the target."""
    last_rate = clean[-1][2][idx]
    if last_rate is None:
        last_rate = 0.0
    cap, slope_delta = categorize(trophy)
    is_prologue = last_rate >= 90.0

    # Earners series on fit reads only (post-refresh regime), recency
    # weighted: E = rate * P / 100.
    usable = fit_reads(clean)
    xs, ys = [], []
    for (h, p, rates) in usable:
        r = rates[idx]
        if r is None or p is None or p <= 0:
            continue
        xs.append(math.log1p(h))
        ys.append(math.log(max(0.0, r * p / 100.0) + 0.5))
    n = len(xs)
    last_e = max(0.0, last_rate * (clean[-1][1] or 0) / 100.0)

    if n >= 3:
        ws = [1.0 + (i / max(1, n - 1)) for i in range(n)]
        a, b_fit, resid_sd, _ = ols(xs, ys, ws)
        # Shrink the fitted slope toward the players slope + category delta.
        w = n / (n + SHRINK_K)
        b = w * b_fit + (1.0 - w) * (b_players + slope_delta)
        x_t = math.log1p(target_hours)
        # Re-anchor at the last observation so level errors don't compound.
        x_last = math.log1p(clean[-1][0])
        k_cap = cap * p_target if p_target > 0 else None
        e_target = extrapolate_earners(last_e, b, x_last, x_t, k_cap)
    elif n >= 1:
        # Too few points for a fit: grow earners with the players slope.
        b = b_players + slope_delta
        x_t = math.log1p(target_hours)
        x_last = xs[-1] if xs else math.log1p(clean[-1][0])
        k_cap = cap * p_target if p_target > 0 else None
        e_target = extrapolate_earners(last_e, b, x_last, x_t, k_cap)
        resid_sd = 0.5
    else:
        e_target = last_e
        resid_sd = 0.8

    e_target = max(e_target, last_e)  # earners never fall
    e_target = min(e_target, p_target * cap) if p_target > 0 else e_target
    median = (e_target / p_target * 100.0) if p_target > 0 else last_rate
    if is_prologue:
        # Structural: the prologue is earned by ~everyone who plays, so
        # its rate sits at the top of the scale whatever the fits say.
        median = 99.9
    else:
        # Bayesian level blend: when few earners were ever observed, the
        # extrapolation has almost no information about the LEVEL, so
        # shrink toward the category's day-7 prior rate. Weight on the
        # data grows with the observed earner count.
        w_data = last_e / (last_e + 10.0)
        prior = level_prior(trophy)
        lg_blend = w_data * logit(median) + (1.0 - w_data) * logit(prior)
        median = inv_logit(lg_blend)
    median = clamp(median)

    # Spread: horizon term + fit residual + small-count + cap proximity
    # + prior reliance (the more the median leans on the prior, the
    # wider the honest band).
    s_h = SIGMA_BASE * math.sqrt(max(horizon_days, 0.0))
    s_fit = min(1.0, resid_sd)
    s_small = 0.30 * max(0.0, 1.0 - last_e / 20.0) if last_e < 20 else 0.0
    near_cap = p_target > 0 and e_target >= 0.8 * p_target * cap
    s_cap = 0.30 if near_cap else 0.0
    w_data = 1.0 if is_prologue else last_e / (last_e + 10.0)
    s_prior = 0.60 * (1.0 - w_data)
    s = math.sqrt(s_h ** 2 + s_fit ** 2 + s_small ** 2 + s_cap ** 2
                  + s_prior ** 2)
    s = min(SPREAD_MAX, max(SPREAD_MIN, s))
    if is_prologue:
        s = min(s, 1.0)

    lg = logit(median)
    qs = [round4(clamp(inv_logit(lg + z * s), 0.0, 100.0)) for z in Z]
    # Keep quantiles non-decreasing after rounding.
    for i in range(1, 5):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return {"id": trophy["id"], "q": qs}


def forecast_stale(trophy, last_rate, horizon_days):
    """Fallback for histories with < 3 distinct reads: expect a refresh.

    Exophase's first refresh multiplied players x6.1 while non-prologue
    earners only x1.3, so non-prologue rates fell ~78%. The fallback cuts
    them to ~30% of the last read and widens the spread to cover both
    "refresh lands" and "no refresh yet". The prologue stays ~99.9%.
    """
    if last_rate is None:
        last_rate = 0.0
    if last_rate >= 90.0:
        median = 99.9
        s = 1.0
    else:
        median = clamp(last_rate * 0.30)
        s = 1.20 + 0.30 * math.sqrt(max(horizon_days, 0.0))
        s = min(SPREAD_MAX, s)
    lg = logit(median)
    qs = [round4(clamp(inv_logit(lg + z * s), 0.0, 100.0)) for z in Z]
    for i in range(1, 5):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return {"id": trophy["id"], "q": qs}


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    horizon_days = case["horizon_minutes"] / 1440.0
    target_hours = hours_since_launch(parse_time(case["target"]))
    clean = clean_history(case)
    last_rates = clean[-1][2]
    rows = []
    if len(clean) < 3:
        for i, trophy in enumerate(case["trophies"]):
            rows.append(forecast_stale(trophy, last_rates[i], horizon_days))
        return {"trophies": rows}
    p_target, b_players = forecast_players(clean, target_hours, horizon_days)
    for i, trophy in enumerate(case["trophies"]):
        rows.append(forecast_trophy(i, trophy, clean, p_target, b_players,
                                    target_hours, horizon_days))
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
