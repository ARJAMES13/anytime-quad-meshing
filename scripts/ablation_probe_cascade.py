"""Probe router: run Instant Meshes (budget-matched, ~2 s) first, then decide from geometry + IM's *observed*
a-priori metrics whether to also run QQS (keep the better a-priori score). No new meshing: uses existing labels.

Compared at matched compute on the frozen TEST set (threshold chosen on VAL):
  QQS alone | IM alone | run both + select | geometry-only gate (same model without IM metrics) |
  classical cascade (run QQS iff IM score is high) | random gate | ML probe cascade |
  two-stage routing (geometry picks the first mesher, a probe on its output decides whether to run the second).
Usage: python scripts/ablation_probe_cascade.py [--split ood]
"""
import argparse

import numpy as np

from _common import load_cfg, rel
from anytime_mesh.labels import load
from anytime_mesh.scoring import paired_bootstrap

IM, QQS = "im_matched", "qqs_default"
MKEYS = ["irregular_frac", "sj_p05", "sj_mean", "bad_frac", "hausdorff_rel", "chamfer_rel", "feat_dist_rel", "quad_frac"]


def rows(path, desc):
    d = load(path)
    X, sq, si, tq, ti, ids = [], [], [], [], [], []
    for sid, m in d.items():
        if IM not in m or QQS not in m or sid not in desc:
            continue
        im, q = m[IM], m[QQS]
        feats = [im.get(k, 0.0) if im.get("valid") else 1.0 for k in MKEYS] + [float(bool(im.get("valid"))), im["score"]]
        X.append(np.r_[desc[sid], feats]); ids.append(sid)
        sq.append(q["score"]); si.append(im["score"])
        tq.append(q.get("time_s", np.nan) / 2 if q.get("valid") else 60.0)
        ti.append(im.get("time_s", np.nan) if im.get("valid") else 10.0)
    return np.array(X), np.array(sq), np.array(si), np.array(tq), np.array(ti), ids


def fit(X, sq, si):
    from sklearn.ensemble import HistGradientBoostingRegressor
    y = np.minimum(si, 2.0) - np.minimum(sq, 2.0)          # predicted gain of also running QQS (>0 = QQS better)
    return HistGradientBoostingRegressor(max_depth=3, max_iter=300, learning_rate=0.04, min_samples_leaf=12,
                                         l2_regularization=1.0, random_state=0).fit(X, y)


def cascade(score_gate, sq, si, tq, ti, frac):
    """Run QQS on the `frac` of parts with the highest gate score; keep the better."""
    n = int(round(frac * len(sq)))
    sel = np.argsort(-score_gate)[:n]
    v, t = si.copy(), ti.copy()
    v[sel] = np.minimum(si[sel], sq[sel]); t[sel] += tq[sel]
    return v, t


if __name__ == "__main__":
    from scipy.stats import binomtest
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=rel("configs/anytime.toml"))
    ap.add_argument("--split", default="test", choices=["test", "ood"])
    a = ap.parse_args()
    paths = load_cfg(a.config)["paths"]
    TEST = rel(paths[a.split])
    z = np.load(rel(paths["descriptors"])); desc = dict(zip(z["ids"].tolist(), z["desc"]))
    Xtr, sqtr, sitr, *_ = rows(rel(paths["train"]), desc)
    Xv, sqv, siv, tqv, tiv, _ = rows(rel(paths["val"]), desc)
    Xt, sqt, sit, tqt, tit, _ = rows(TEST, desc)
    print(f"train {len(Xtr)} val {len(Xv)} test {len(Xt)}")
    model = fit(Xtr, sqtr, sitr)
    gv, gt = model.predict(Xv), model.predict(Xt)
    # the geometry-only twin (same model, IM metrics removed) for an apples-to-apples comparison
    ng = Xtr.shape[1] - len(MKEYS) - 2
    geo = fit(Xtr[:, :ng], sqtr, sitr); gt_geo = geo.predict(Xt[:, :ng])
    print(f"{'strategy':44s} {'mean':>7s} {'median':>7s}  Δmean vs QQS [95% CI]       W/L     sign-p   time")

    def report(name, v, t):
        md, lo, hi = paired_bootstrap(v, sqt, n=10000)
        w, l_ = int((v < sqt - .005).sum()), int((v > sqt + .005).sum())
        p = binomtest(w, w + l_, .5).pvalue if w + l_ else 1
        print(f"{name:44s} {v.mean():7.4f} {np.median(v):7.4f}  {md:+.4f} [{lo:+.4f},{hi:+.4f}]  {w:3d}/{l_:3d}  {p:.1e}  {np.nanmean(t):5.2f}s")

    report("QQS alone", sqt, tqt)
    report("IM alone (matched)", sit, tit)
    report("run both + select", np.minimum(sqt, sit), tqt + tit)
    rng = np.random.default_rng(0)
    for frac in (0.2, 0.3, 0.5):
        report(f"ML probe cascade (QQS on {frac:.0%})", *cascade(gt, sqt, sit, tqt, tit, frac))
        report(f"geometry-only gate (QQS on {frac:.0%})", *cascade(gt_geo, sqt, sit, tqt, tit, frac))
        report(f"classical: IM score gate (QQS on {frac:.0%})", *cascade(sit, sqt, sit, tqt, tit, frac))
        report(f"random gate (QQS on {frac:.0%})", *cascade(rng.random(len(sqt)), sqt, sit, tqt, tit, frac))

    # ---- two-stage sequential routing: geometry picks the first mesher; a probe on its output decides the 2nd ----
    def rows2(path):
        d = load(path)
        G, FI, FQ, sq, si, tq, ti = [], [], [], [], [], [], []
        for sid, m in d.items():
            if IM not in m or QQS not in m or sid not in desc:
                continue
            f = lambda r: [r.get(k, 0.0) if r.get("valid") else 1.0 for k in MKEYS] + [float(bool(r.get("valid"))), r["score"]]
            G.append(desc[sid]); FI.append(f(m[IM])); FQ.append(f(m[QQS]))
            sq.append(m[QQS]["score"]); si.append(m[IM]["score"])
            tq.append(m[QQS].get("time_s", np.nan) / 2 if m[QQS].get("valid") else 60.0)
            ti.append(m[IM].get("time_s", np.nan) if m[IM].get("valid") else 10.0)
        return [np.array(a) for a in (G, FI, FQ, sq, si, tq, ti)]
    from sklearn.ensemble import HistGradientBoostingRegressor as HGB
    mk = lambda: HGB(max_depth=3, max_iter=300, learning_rate=0.04, min_samples_leaf=12, l2_regularization=1.0, random_state=0)
    Gtr, FItr, FQtr, sqtr2, sitr2, _, _ = rows2(rel(paths["train"]))
    Gt, FIt, FQt, sqt2, sit2, tqt2, tit2 = rows2(TEST)
    c = lambda a: np.minimum(a, 2.0)
    stage1 = mk().fit(Gtr, c(sitr2) - c(sqtr2))                  # >0: QQS predicted better -> start with QQS
    probe_im = mk().fit(np.c_[Gtr, FItr], c(sitr2) - c(sqtr2))   # after IM: gain of also running QQS
    probe_q = mk().fit(np.c_[Gtr, FQtr], c(sqtr2) - c(sitr2))    # after QQS: gain of also running IM
    start_q = stage1.predict(Gt) > 0
    gain2 = np.where(start_q, probe_q.predict(np.c_[Gt, FQt]), probe_im.predict(np.c_[Gt, FIt]))
    first_s = np.where(start_q, sqt2, sit2); first_t = np.where(start_q, tqt2, tit2)
    second_s = np.where(start_q, sit2, sqt2); second_t = np.where(start_q, tit2, tqt2)
    assert np.allclose(sqt2, sqt)
    for frac in (0.0, 0.2, 0.3, 0.5):
        n = int(round(frac * len(sqt2))); sel = np.argsort(-gain2)[:n]
        v, t = first_s.copy(), first_t.copy()
        v[sel] = np.minimum(v[sel], second_s[sel]); t[sel] += second_t[sel]
        report(f"TWO-STAGE (2nd mesher on {frac:.0%})", v, t)
