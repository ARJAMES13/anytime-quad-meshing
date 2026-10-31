import numpy as np

from anytime_mesh.qqs import at_budget


def _p(n, irr, valid=True):
    return {"valid": valid, "n_quads": n, "irregular_frac": irr, "sj_p05": 0.5, "bad_frac": 0.0,
            "hausdorff_rel": 0.01, "feat_dist_rel": 0.0, "status": "ok"}


def test_interpolates_in_log_n():
    tp = {"p1": _p(1000, 0.10), "p2": _p(4000, 0.02), "time_s": 3.0}
    q = at_budget(tp, 2000)                      # log-midpoint of 1000 and 4000
    assert np.isclose(q["irregular_frac"], 0.06) and q["matched"] and q["n_quads"] == 2000


def test_far_extrapolation_flagged_and_clamped():
    tp = {"p1": _p(5000, 0.10), "p2": _p(6000, 0.02), "time_s": 1.0}
    q = at_budget(tp, 1000)
    assert not q["matched"]
    assert q["irregular_frac"] >= 0.0            # never negative after extrapolation


def test_single_valid_pass_and_failure():
    q = at_budget({"p1": _p(2100, 0.05), "p2": {"valid": False}, "time_s": 1.0}, 2000)
    assert q["valid"] and q["matched"] and np.isclose(q["irregular_frac"], 0.05)
    assert not at_budget({"p1": {"valid": False}, "p2": {"valid": False}, "time_s": 0}, 2000)["valid"]
