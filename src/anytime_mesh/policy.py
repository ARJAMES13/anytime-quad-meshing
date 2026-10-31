"""Learned anytime meshing policy.

Step 0: always run one budget-matched Instant Meshes pass (IM1).
Step 1: from the part's geometry descriptor + IM1's observed a-priori metrics, choose one action:
  stop   -> keep IM1
  +IMk   -> best of IM1 and 2 more IM size perturbations
  +QQS   -> better of IM1 and Gmsh QQS
  +both  -> best of everything
One gradient-boosted regressor per action predicts the resulting (clipped) composite score; the policy picks
argmin(pred_score + lam * expected_time). lam is the largest value whose VAL mean score is within `val_tolerance`
of the best VAL mean.
"""
import numpy as np

from .labels import load

MKEYS = ["irregular_frac", "sj_p05", "sj_mean", "bad_frac", "hausdorff_rel", "chamfer_rel", "feat_dist_rel", "quad_frac"]
ACTIONS = ["stop", "+IMk", "+QQS", "+both"]
DEFAULT_GBM = {"max_depth": 3, "max_iter": 300, "learning_rate": 0.04, "min_samples_leaf": 12,
               "l2_regularization": 1.0, "random_state": 0}


def load_descriptors(path):
    z = np.load(path)
    return dict(zip(z["ids"].tolist(), z["desc"]))


def features(desc_vec, im1: dict) -> np.ndarray:
    """Policy input: 15-D geometry descriptor + IM1 metrics (+ validity flag and IM1 score)."""
    valid = bool(im1.get("valid"))
    return np.r_[desc_vec, [im1.get(k, 0.0) if valid else 1.0 for k in MKEYS], [float(valid), im1["score"]]]


ACTION_MEMBERS = [(0,), (0, 1), (0, 2), (0, 1, 2)]   # candidates per action: 0 = IM1, 1 = best extra IM run, 2 = QQS


def table(path, desc, ids=None, cands=None):
    """Labels -> X (features), S (score per action), T (time per action), Q (QQS-alone score).
    If `cands` is a list, the per-part candidate metric dicts [IM1, IMk-best, QQS] are appended to it."""
    d = load(path)
    X, S, T, Q = [], [], [], []
    for sid, m in d.items():
        if not all(k in m for k in ("qqs_default", "im_matched", "im_k3")) or sid not in desc:
            continue
        im1, ik, q = m["im_matched"], m["im_k3"], m["qqs_default"]
        if "k_scores" not in ik:
            continue
        s1 = im1["score"]; sk = min([s1] + list(ik["k_scores"])); sq = q["score"]
        t1 = im1.get("time_s", 2.0) if im1.get("valid") else 10.0
        tk = 0.5 * ik.get("time_s", 6.0)                      # 2 extra IM runs out of ~4 in the im_k3 timing
        tq = q.get("time_s", np.nan) / 2 if q.get("valid") else 60.0
        X.append(features(desc[sid], im1))
        S.append([s1, sk, min(s1, sq), min(sk, sq)])
        T.append([t1, t1 + tk, t1 + tq, t1 + tk + tq])
        Q.append(sq)
        if ids is not None:
            ids.append(sid)
        if cands is not None:
            cands.append([im1, ik, q])
    return np.array(X), np.array(S), np.array(T), np.array(Q)


def chosen_metrics(cands, actions):
    """Metric dict of the mesh each part ends up with (lowest score among the action's candidates)."""
    return [min((c[i] for i in ACTION_MEMBERS[a]), key=lambda r: r["score"]) for c, a in zip(cands, actions)]


def fit(X, S, gbm=None, clip=2.0):
    from sklearn.ensemble import HistGradientBoostingRegressor
    params = {**DEFAULT_GBM, **(gbm or {})}
    return [HistGradientBoostingRegressor(**params).fit(X, np.minimum(S[:, a], clip)) for a in range(S.shape[1])]


def run_policy(models, X, S, T, lam):
    P = np.stack([m.predict(X) for m in models], 1)
    a = np.argmin(P + lam * T, 1)
    r = np.arange(len(S))
    return S[r, a], T[r, a], a


def choose_lambda(models, Xv, Sv, Tv, lams, tol=0.02):
    """Largest lam whose VAL mean score is within `tol` (relative) of the best VAL mean."""
    val = [(lam, *[np.mean(x) for x in run_policy(models, Xv, Sv, Tv, lam)[:2]]) for lam in lams]
    best = min(v[1] for v in val)
    return max(v[0] for v in val if v[1] <= best * (1 + tol)), val


def decide(models, x, lam, expected_times):
    """Pick an action for a single part (inference); expected_times = time per action in seconds."""
    p = np.array([m.predict(x[None])[0] for m in models])
    return int(np.argmin(p + lam * np.asarray(expected_times))), p


def save(path, models, lam, meta=None):
    import joblib
    joblib.dump({"models": models, "lam": lam, "actions": ACTIONS, "meta": meta or {}}, path)


def load_policy(path):
    import joblib
    return joblib.load(path)
