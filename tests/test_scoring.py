import numpy as np

from anytime_mesh.scoring import composite_score, paired_bootstrap


def test_paired_bootstrap_detects_shift():
    rng = np.random.default_rng(1)
    a = rng.normal(size=200)
    md, lo, hi = paired_bootstrap(a + 0.5, a, n=2000, seed=0)
    assert np.isclose(md, 0.5) and lo > 0.4 and hi < 0.6


def test_paired_bootstrap_null_contains_zero():
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=300), rng.normal(size=300)
    _, lo, hi = paired_bootstrap(a, b, n=2000, seed=0)
    assert lo < 0 < hi


def test_paired_bootstrap_deterministic():
    a, b = np.arange(10.0), np.arange(10.0)[::-1]
    assert paired_bootstrap(a, b, seed=0) == paired_bootstrap(a, b, seed=0)


def test_composite_failure_is_worst():
    good = {"valid": True, "irregular_frac": 0.05, "bad_frac": 0.01, "sj_p05": 0.7, "hausdorff_rel": 0.01, "feat_dist_rel": 0.003}
    assert composite_score({"valid": False}) > composite_score(good) > 0
