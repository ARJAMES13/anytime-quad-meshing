"""Regression test: the committed labels reproduce the reported headline numbers exactly."""
import os

import numpy as np
import pytest

from _common import load_cfg, rel
from anytime_mesh import policy as P
from anytime_mesh.scoring import paired_bootstrap


@pytest.fixture(scope="module")
def trained():
    cfg = load_cfg(rel("configs/anytime.toml"))
    pc, paths = cfg["policy"], cfg["paths"]
    if not os.path.exists(rel(paths["train"])):
        pytest.skip("labels not present")
    desc = P.load_descriptors(rel(paths["descriptors"]))
    Xtr, Str, _, _ = P.table(rel(paths["train"]), desc)
    Xv, Sv, Tv, _ = P.table(rel(paths["val"]), desc)
    models = P.fit(Xtr, Str, pc.get("gbm"), pc["clip"])
    lam, _ = P.choose_lambda(models, Xv, Sv, Tv, pc["lambdas"], pc["val_tolerance"])
    return models, lam, desc, paths


def _eval(trained, split):
    models, lam, desc, paths = trained
    X, S, T, Q = P.table(rel(paths[split]), desc)
    v, t, _ = P.run_policy(models, X, S, T, lam)
    md, lo, hi = paired_bootstrap(v, Q, n=10000)
    return len(X), lam, v.mean(), md, lo, hi, int((v < Q - .005).sum()), int((v > Q + .005).sum()), np.nanmean(t)


def test_test_set_headline(trained):
    n, lam, mean, md, lo, hi, w, l_, t = _eval(trained, "test")
    assert (n, lam, w, l_) == (142, 0.005, 73, 6)
    assert [round(x, 4) for x in (mean, md, lo, hi)] == [0.2451, -0.1309, -0.1702, -0.0944] and round(t, 2) == 5.66


def test_ood_headline(trained):
    n, lam, mean, md, lo, hi, w, l_, t = _eval(trained, "ood")
    assert (n, w, l_) == (39, 13, 4)
    assert [round(x, 4) for x in (mean, md, lo, hi)] == [0.2327, -0.0441, -0.0912, -0.0079] and round(t, 2) == 5.41
