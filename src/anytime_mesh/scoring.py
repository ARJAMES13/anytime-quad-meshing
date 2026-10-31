"""Paired statistics and the pre-registered composite quad score (lower is better)."""
import numpy as np

# Pre-registered before any learning experiment: weights fixed, never tuned on test.
COMPOSITE_WEIGHTS = {"irregular_frac": 1.0, "bad_frac": 1.0, "one_minus_sj_p05": 0.5,
                     "hausdorff_rel": 10.0, "feat_dist_rel": 10.0}
FAIL_SCORE = 10.0


def composite_score(m: dict) -> float:
    if not m.get("valid"):
        return FAIL_SCORE
    w = COMPOSITE_WEIGHTS
    return float(w["irregular_frac"] * m["irregular_frac"] + w["bad_frac"] * m["bad_frac"]
                 + w["one_minus_sj_p05"] * (1.0 - m["sj_p05"]) + w["hausdorff_rel"] * m["hausdorff_rel"]
                 + w["feat_dist_rel"] * m["feat_dist_rel"])


def paired_bootstrap(a, b, n: int = 10000, seed: int = 0):
    """Mean of (a-b) with a percentile 95% CI from resampling solids."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    boots = d[idx].mean(1)
    return float(d.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
