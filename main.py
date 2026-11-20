"""GTA VI trophy rarity forecaster for make.fun bounty gta6-trophy-rarity-forecast-01.

Reads one case per *.in file (the game's 55 PS5 trophies plus the house's
recording of Exophase's trophy page up to the checkpoint) and writes, for each
trophy, five quantiles (5/25/50/75/95%) of its earn rate at the target,
scored by pinball loss on the logit scale.

Model: earners / players decomposition with tier priors.

A trophy's earn rate is r = E / P, earners over players tracked. In a launch
week both move fast for different reasons, so the program models them
separately and divides at the target instead of extrapolating the ratio
directly (extrapolating the ratio conflates player progression with
denominator dilution).

  1. Clean. Exophase refreshes on its own schedule (hours apart), so the
     reads are a step function: long flat stretches punctuated by jumps.
     Consecutive duplicate reads are dropped; only distinct reads are fitted.
  2. Players P(t). log P is fit against ln(1 + hours since launch) over the
     last 24h of distinct reads, with a concave quadratic term (launch-week
     player growth decelerates; a straight line overshoots a saturating
     curve); growth is bounded to a doubling per day.
  3. Post-refresh reads only. The launch-night reads are a different regime
     (638 mostly hardcore early-access players, rates far above the refreshed
     population's). Fitting through them flattens every slope. All fits use
     only reads with players >= 10% of the last read's count, within 48h of
     the cutoff.
  4. Earners E_i(t). Per trophy, log(E + 0.5) is fit against ln(1 + hours)
     with a concave quadratic term (progression also saturates), recency-
     weighted; the linear slope is shrunk toward the players slope plus a
     category adjustment (story trophies catch up faster; grinds grow
     slower). Extrapolation is damped with a tanh soft cap toward a category
     day-7 cap (S-curve saturation), and floored at the last observed
     earners.
  5. Level prior. When a trophy is early in its expected journey (few
     earners relative to what its category's day-7 prior implies at the
     target player count), the median is blended in logit space toward the
     category's day-7 prior rate (calibrated from GTA V lifetime rarities
     scaled to one week, and from the real day-1 Exophase rates). A trophy
     at 100% in the last read (the prologue) is set to 99.9% structurally.
  6. Spread. Horizon term + fit residual + small-count + cap proximity +
     prior reliance, combined in quadrature.
  7. Stale history. With fewer than 3 distinct reads there is no trend to
     fit, so the program expects a refresh: non-prologue rates are cut to
     22% of the last read (the spec's published first refresh multiplied
     players x6.1 while non-prologue earners only x1.34, so rates fell to
     ~22%), with a wide spread.

Python 3.13, standard library only, offline, deterministic.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at the scored levels.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]
LAUNCH = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)

# --- model constants -------------------------------------------------------
# Fit window: reads with players >= this fraction of the last read's count,
# within this many hours of the checkpoint.
MIN_PLAYER_FRAC = 0.10
FIT_WINDOW_HOURS = 48.0
PLAYERS_FIT_HOURS = 24.0
# Recency half-life (hours) for the earners-fit weights.
RECENCY_HALFLIFE_H = 12.0
# Prior weight: w = 1 / (1 + PRIOR_K * progress), where progress is the
# trophy's observed earners over the earners its day-7 prior implies at the
# target player count. A trophy that has covered little of its expected
# journey leans on the category prior; one nearly there trusts its own data.
PRIOR_K = 3.0
# Concave-quadratic fits: curvature clamped to this range (never convex:
# growth does not accelerate forever) and used only with >= this many points.
MAX_CURVATURE = 0.0
MIN_CURVATURE = -1.5
MIN_QUAD_POINTS = 4
# Stale-history rule (fewer than 3 distinct reads): expected refresh.
STALE_MIN_READS = 3
STALE_FACTOR = 0.22
STALE_SIGMA = 0.9
# Spread pieces (logit units).
BASE_SIGMA = 0.20
HORIZON_SIGMA = 0.18
MAX_SIGMA = 1.6
MIN_SIGMA = 0.2
PROLOGUE_RATE = 99.9
PROLOGUE_SIGMA = 0.15

# Category table: day-7 prior rate (%), day-7 cap (%), slope adjustment
# relative to the players slope. Priors are calibrated from GTA V lifetime
# rarities scaled down to one week (a launch week's player base is far more
# casual than a lifetime one) and from the real day-1 Exophase rates.
CATEGORIES = {
    "prologue": (99.5, 99.9, 0.0),
    "story_early": (50.0, 80.0, 0.25),
    "story_mid": (32.0, 62.0, 0.15),
    "story_late": (11.0, 32.0, 0.0),
    "side": (14.0, 38.0, 0.0),
    "grind": (2.0, 9.0, -0.25),
    "misc": (9.0, 32.0, -0.05),
    "platinum": (0.15, 0.6, -0.3),
    "completion": (0.3, 1.2, -0.3),
}
GRIND_WORDS = (
    "250 stunts", "100 locations", "100 vehicles", "50 challenges",
    "20 different races", "gold medal", "collectible card", "every spot",
    "explore all", "purchase all", "earn $", "spend $", "win a top fun",
)


def parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_launch(ts):
    """Hours from launch (2026-11-19T00:00Z); reads just before launch count
    as slightly negative, clamped so ln(1 + h) stays defined."""
    h = (ts - LAUNCH).total_seconds() / 3600.0
    return max(h, -0.9)


def x_of(ts):
    return math.log1p(hours_since_launch(ts))


def clamp_pct(value):
    return min(99.99, max(0.01, value))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid_to_pct(logit_value):
    return 100.0 / (1.0 + math.exp(-logit_value))


def round4(value):
    """Round to 4 decimals, half up (matching JavaScript's Math.round)."""
    scaled = value * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    result = whole / 10000
    return int(result) if result.is_integer() else result


def classify(trophy):
    """Assign a trophy to a progression category from its description."""
    desc = trophy.get("description", "")
    if trophy.get("tier") == "platinum":
        return "platinum"
    if "100% Completion" in desc:
        return "completion"
    if "Complete Gone Fishing" in desc:
        return "prologue"
    if "Complete Chapter 1" in desc or "Complete Chapter 2" in desc:
        return "story_early"
    if "Complete Chapter" in desc:
        return "story_mid"
    if desc == "Complete the final mission.":
        return "story_late"
    if "final mission for" in desc or "mission strands" in desc:
        return "side"
    lowered = desc.lower()
    if any(word in lowered for word in GRIND_WORDS):
        return "grind"
    if desc.startswith("Complete ") and trophy.get("secret"):
        return "story_mid"
    return "misc"


def distinct_reads(history):
    """Drop consecutive reads identical in players and rates (the page had
    not refreshed between them); keep each step's first read."""
    out = []
    prev_key = None
    for read in history:
        key = (read.get("players_tracked"),
               tuple(read.get("rates") or []))
        if key != prev_key:
            out.append(read)
            prev_key = key
    return out


def weighted_line(points):
    """Weighted least squares for y ~ a + b*x. points are (x, y, w).
    Returns (a, b, rms_residual)."""
    sw = sum(p[2] for p in points)
    mx = sum(p[2] * p[0] for p in points) / sw
    my = sum(p[2] * p[1] for p in points) / sw
    sxx = sum(p[2] * (p[0] - mx) ** 2 for p in points)
    sxy = sum(p[2] * (p[0] - mx) * (p[1] - my) for p in points)
    slope = sxy / sxx if sxx > 1e-12 else 0.0
    intercept = my - slope * mx
    resid = math.sqrt(
        sum(p[2] * (p[1] - (intercept + slope * p[0])) ** 2 for p in points) / sw
    )
    return intercept, slope, resid


def _solve_3x3(mat, vec):
    """Gaussian elimination for a 3x3 system; None if singular."""
    rows = [list(mat[i]) + [vec[i]] for i in range(3)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(rows[r][col]))
        if abs(rows[piv][col]) < 1e-12:
            return None
        rows[col], rows[piv] = rows[piv], rows[col]
        for r in range(col + 1, 3):
            factor = rows[r][col] / rows[col][col]
            for c in range(col, 4):
                rows[r][c] -= factor * rows[col][c]
    out = [0.0, 0.0, 0.0]
    for r in (2, 1, 0):
        out[r] = (rows[r][3] - sum(rows[r][c] * out[c] for c in range(r + 1, 3))
                  ) / rows[r][r]
    return out


def weighted_quad(points):
    """Weighted least squares for y ~ a + b*x + c*x^2, with the curvature
    clamped to [MIN_CURVATURE, MAX_CURVATURE] (concave only) and the line
    re-fit at the clamped curvature. Returns (a, b, c, rms_residual), or
    None if the system is degenerate."""
    # Centre x for conditioning; the fit is shift-invariant.
    sw = sum(p[2] for p in points)
    mx = sum(p[2] * p[0] for p in points) / sw
    centered = [(p[0] - mx, p[1], p[2]) for p in points]
    # Normal equations in the basis (1, u, u^2).
    mat = [[0.0] * 3 for _ in range(3)]
    vec = [0.0] * 3
    for u, y, w in centered:
        basis = (1.0, u, u * u)
        for i in range(3):
            vec[i] += w * basis[i] * y
            for j in range(3):
                mat[i][j] += w * basis[i] * basis[j]
    sol = _solve_3x3(mat, vec)
    if sol is None:
        return None
    a0, b0, c0 = sol
    c = min(max(c0, MIN_CURVATURE), MAX_CURVATURE)
    if abs(c - c0) > 1e-12:
        # Re-fit the line at the clamped curvature: y - c*u^2 ~ a + b*u.
        adj = [(u, y - c * u * u, w) for u, y, w in centered]
        a0, b0, _ = weighted_line(adj)
    # Shift back to the original x: y = a0 + b0*(x-mx) + c*(x-mx)^2.
    a = a0 - b0 * mx + c * mx * mx
    b = b0 - 2.0 * c * mx
    resid = math.sqrt(sum(
        p[2] * (p[1] - (a + b * p[0] + c * p[0] * p[0])) ** 2
        for p in points) / sw)
    return a, b, c, resid


def fit_players(reads, cutoff, x_target):
    """Fit log players against ln(1 + hours) over the last 24h of distinct
    reads, with a concave quadratic term when there are enough points
    (launch-week player growth decelerates); extrapolate to the target with
    growth bounded to a doubling per day. Returns (players_at_target, slope)
    where slope is the effective log-slope at the last read."""
    last_players = reads[-1].get("players_tracked") or 1
    points = []
    for read in reads:
        ts = parse_ts(read["at"])
        if (cutoff - ts).total_seconds() > PLAYERS_FIT_HOURS * 3600:
            continue
        players = read.get("players_tracked") or 0
        if players <= 0:
            continue
        points.append((x_of(ts), math.log(players), 1.0))
    if len(points) < 2:
        return float(last_players), 0.0
    # Bound: at most a doubling over the day ending at the target.
    h_target = math.expm1(x_target)
    day_span = math.log1p(h_target) - math.log1p(max(h_target - 24.0, -0.9))
    max_slope = math.log(2.0) / day_span if day_span > 1e-9 else 0.0
    x_last = points[-1][0]
    if len(points) >= MIN_QUAD_POINTS:
        quad = weighted_quad(points)
    else:
        quad = None
    if quad is not None:
        intercept, slope, curv, _ = quad
        # Effective slope at the target, bounded; re-anchor the curve at the
        # last point so the bound bends the path, not the level.
        slope_tgt = slope + 2.0 * curv * x_target
        slope_tgt = min(max(slope_tgt, 0.0), max_slope)
        log_last = intercept + slope * x_last + curv * x_last * x_last
        # Integrate the (linearly decaying) slope from x_last to x_target.
        slope_last = slope + 2.0 * curv * x_last
        avg_slope = 0.5 * (min(max(slope_last, 0.0), max_slope) + slope_tgt)
        log_target = log_last + avg_slope * (x_target - x_last)
        players_target = math.exp(log_target)
        eff_slope = min(max(slope_last, 0.0), max_slope)
        return max(players_target, float(last_players)), eff_slope
    intercept, slope, _ = weighted_line(points)
    slope = min(max(slope, 0.0), max_slope)
    # Re-anchor the intercept at the weighted centroid with the bounded slope.
    sw = sum(p[2] for p in points)
    mx = sum(p[2] * p[0] for p in points) / sw
    my = sum(p[2] * p[1] for p in points) / sw
    intercept = my - slope * mx
    players_target = math.exp(intercept + slope * x_target)
    return max(players_target, float(last_players)), slope


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_days = case["horizon_minutes"] / 1440.0
    cutoff = parse_ts(case["at"])
    target = parse_ts(case["target"])
    x_target = x_of(target)

    def row_for(trophy, median_pct, sigma):
        center = logit(median_pct)
        return {"id": trophy["id"],
                "q": [round4(sigmoid_to_pct(center + z * sigma)) for z in Z]}

    if not history:
        # No reads at all: fall back to category priors, wide spread.
        return {"trophies": [
            row_for(t, CATEGORIES[classify(t)][0], 1.2) for t in trophies]}

    reads = distinct_reads(history)
    last = history[-1]
    last_players = last.get("players_tracked") or 1
    last_rates = last.get("rates") or []

    def last_rate(i):
        if i < len(last_rates) and last_rates[i] is not None:
            return last_rates[i]
        for read in reversed(history):
            rates = read.get("rates") or []
            if i < len(rates) and rates[i] is not None:
                return rates[i]
        return None

    if len(reads) < STALE_MIN_READS:
        # Stale history: no trend to fit, so expect a refresh before the
        # target. The spec's published first refresh took non-prologue rates
        # to ~22% of their pre-refresh values; the spread stays wide because
        # the refresh's per-trophy effect is unseen.
        sigma = math.sqrt(STALE_SIGMA ** 2 +
                          (HORIZON_SIGMA * math.sqrt(horizon_days)) ** 2)
        rows = []
        for i, trophy in enumerate(trophies):
            rate = last_rate(i)
            if rate is not None and rate >= 99.99:
                rows.append(row_for(trophy, PROLOGUE_RATE, PROLOGUE_SIGMA))
            elif rate is None:
                rows.append(row_for(trophy, CATEGORIES[classify(trophy)][0],
                                    sigma))
            elif rate <= 0:
                rows.append(row_for(trophy, 0.05, sigma))
            else:
                rows.append(row_for(trophy, rate * STALE_FACTOR, sigma))
        return {"trophies": rows}

    # Fit window: post-refresh reads only, within 48h of the cutoff.
    window = []
    for read in reads:
        ts = parse_ts(read["at"])
        players = read.get("players_tracked") or 0
        if players < MIN_PLAYER_FRAC * last_players:
            continue
        if (cutoff - ts).total_seconds() > FIT_WINDOW_HOURS * 3600:
            continue
        window.append((ts, read))
    if len(window) < 2:
        window = [(parse_ts(r["at"]), r) for r in reads[-2:]]
    window_reads = [r for _, r in window]
    last_ts = window[-1][0]

    players_target, players_slope = fit_players(window_reads, cutoff, x_target)

    rows = []
    for i, trophy in enumerate(trophies):
        category = classify(trophy)
        prior_pct, cap_pct, slope_adj = CATEGORIES[category]
        rate_last = last_rate(i)

        if rate_last is not None and rate_last >= 99.99:
            rows.append(row_for(trophy, PROLOGUE_RATE, PROLOGUE_SIGMA))
            continue

        # Earners series for this trophy over the fit window.
        points = []
        earners_last = 0.0
        for ts, read in window:
            rates = read.get("rates") or []
            if i >= len(rates) or rates[i] is None:
                continue
            players = read.get("players_tracked") or 0
            earners = rates[i] / 100.0 * players
            earners_last = earners
            age_h = (last_ts - ts).total_seconds() / 3600.0
            weight = 0.5 ** (age_h / RECENCY_HALFLIFE_H)
            points.append((x_of(ts), math.log(earners + 0.5), weight))

        if len(points) >= 2:
            n = len(points)
            anchor_slope = players_slope + slope_adj
            x_last = points[-1][0]
            quad = weighted_quad(points) if n >= MIN_QUAD_POINTS else None
            if quad is not None:
                qa, qb, qc, resid = quad
                # Effective slope at the last read, shrunk toward the anchor.
                slope_last = qb + 2.0 * qc * x_last
                slope = ((n / (n + 2.0)) * slope_last +
                         (2.0 / (n + 2.0)) * anchor_slope)
                slope = min(max(slope, -0.3), players_slope + 1.5)
                # Integrate the decaying slope from the last read to the
                # target, anchored at the fitted level at the last read.
                log_last = qa + qb * x_last + qc * x_last * x_last
                dx = x_target - x_last
                log_earners_raw = log_last + slope * dx + qc * dx * dx
            else:
                _, slope_fit, resid = weighted_line(points)
                slope = ((n / (n + 2.0)) * slope_fit +
                         (2.0 / (n + 2.0)) * anchor_slope)
                slope = min(max(slope, -0.3), players_slope + 1.5)
                # Re-anchor at the weighted centroid with the shrunk slope.
                sw = sum(p[2] for p in points)
                mx = sum(p[2] * p[0] for p in points) / sw
                my = sum(p[2] * p[1] for p in points) / sw
                intercept = my - slope * mx
                log_earners_raw = intercept + slope * x_target
        else:
            slope = players_slope + slope_adj
            resid = 0.5
            log_earners_raw = (math.log(earners_last + 0.5) +
                               slope * (x_target - x_of(last_ts)))

        earners_raw = math.exp(log_earners_raw)
        # Tanh soft cap toward the category day-7 cap (S-curve saturation).
        earners_cap = cap_pct / 100.0 * players_target
        if earners_cap > 0:
            earners_target = earners_cap * math.tanh(earners_raw / earners_cap)
        else:
            earners_target = earners_raw
        earners_target = max(earners_target, earners_last)
        rate_data = earners_target / players_target * 100.0

        # Level prior: blend toward the category day-7 prior in logit space.
        # The weight follows the trophy's progress: earners observed so far
        # over the earners the prior implies at the target player count. A
        # trophy that has covered little of its expected journey leans on
        # the prior; one nearly there trusts its own data. A short hop stays
        # close to the data regardless.
        earners_prior_target = prior_pct / 100.0 * players_target
        progress = earners_last / max(earners_prior_target, 1.0)
        prior_weight = 1.0 / (1.0 + PRIOR_K * progress)
        prior_weight *= min(1.0, horizon_days / 3.0)
        center = ((1.0 - prior_weight) * logit(rate_data) +
                  prior_weight * logit(prior_pct))

        # Spread: horizon + fit residual + small counts + cap proximity +
        # prior reliance, combined in quadrature.
        pieces = [
            BASE_SIGMA,
            HORIZON_SIGMA * math.sqrt(horizon_days),
            min(resid, 0.8),
            0.9 / math.sqrt(earners_last + 2.0),
            0.5 * max(0.0, rate_data / cap_pct - 0.5) if cap_pct > 0 else 0.0,
            0.7 * prior_weight,
        ]
        sigma = math.sqrt(sum(p * p for p in pieces))
        sigma = min(max(sigma, MIN_SIGMA), MAX_SIGMA)
        rows.append({"id": trophy["id"],
                     "q": [round4(sigmoid_to_pct(center + z * sigma))
                           for z in Z]})
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
