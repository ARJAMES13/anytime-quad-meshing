"""Gmsh quasi-structured quad meshing (Reberol et al. 2021) driven by an optional external size field.

All runs are budget-matched: a first pass measures the quad count for the given (relative) size field,
then the field is rescaled once by sqrt(n1 / n_target) and re-run. Sizes are per reference-surface vertex,
relative (only their shape matters; the global scale is set by the budget).
"""
import time

import numpy as np

from .geometry import SurfaceSample, import_cad
from .meshers import QuadResult

N_TARGET = 2500
QQS_FIRST_PASS_FACTOR = 2.2   # QQS realises ~4-5x more quads than integral(dA/h^2); start near target


def vertex_areas(s: SurfaceSample) -> np.ndarray:
    a = 0.5 * np.linalg.norm(np.cross(s.V[s.F[:, 1]] - s.V[s.F[:, 0]], s.V[s.F[:, 2]] - s.V[s.F[:, 0]]), axis=1)
    A = np.zeros(len(s.V))
    for j in range(3):
        np.add.at(A, s.F[:, j], a / 3)
    return A


def budget_scale(h_rel: np.ndarray, s: SurfaceSample, n_target: int) -> np.ndarray:
    """Absolute sizes with integral dA / h^2 = n_target."""
    A = vertex_areas(s)
    return h_rel * np.sqrt(np.sum(A / h_rel ** 2) / n_target)


def _qqs_once(path, s, h_abs, method):
    import gmsh
    t0 = time.perf_counter()
    gmsh.initialize(readConfigFiles=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        import_cad(path, 1.0, heal=True)
        if np.ndim(h_abs) == 0:
            gmsh.option.setNumber("Mesh.MeshSizeMin", float(h_abs))
            gmsh.option.setNumber("Mesh.MeshSizeMax", float(h_abs))
        else:
            P = s.V[s.F]
            data = np.c_[P[:, :, 0], P[:, :, 1], P[:, :, 2], h_abs[s.F]].ravel()
            v = gmsh.view.add("size")
            gmsh.view.addListData(v, "ST", len(s.F), data.tolist())
            f = gmsh.model.mesh.field.add("PostView")
            gmsh.model.mesh.field.setNumber(f, "ViewTag", v)
            gmsh.model.mesh.field.setAsBackgroundMesh(f)
            for k in ("Mesh.MeshSizeExtendFromBoundary", "Mesh.MeshSizeFromPoints", "Mesh.MeshSizeFromCurvature"):
                gmsh.option.setNumber(k, 0)
        gmsh.option.setNumber("Mesh.Algorithm", 11)
        gmsh.option.setNumber("Mesh.QuadqsSizemapMethod", method)
        gmsh.model.mesh.generate(2)
        tags, xyz, _ = gmsh.model.mesh.getNodes()
        V = xyz.reshape(-1, 3)
        tag2i = np.zeros(int(tags.max()) + 1, np.int64)
        tag2i[tags.astype(np.int64)] = np.arange(len(tags))
        Q, T = [], []
        for et, _, en in zip(*gmsh.model.mesh.getElements(2)):
            if et == 3:
                Q.append(tag2i[en.astype(np.int64)].reshape(-1, 4))
            elif et == 2:
                T.append(tag2i[en.astype(np.int64)].reshape(-1, 3))
    finally:
        gmsh.finalize()
    Q = np.concatenate(Q) if Q else np.zeros((0, 4), np.int64)
    T = np.concatenate(T) if T else np.zeros((0, 3), np.int64)
    used = np.unique(np.r_[Q.ravel(), T.ravel()])
    remap = -np.ones(len(V), np.int64); remap[used] = np.arange(len(used))
    return QuadResult("ok", V[used], remap[Q], remap[T], time.perf_counter() - t0)


def run_qqs(path: str, s: SurfaceSample, h_rel=None, method: int = 3, n_target: int = N_TARGET,
            calibrate: bool = True) -> QuadResult:
    """h_rel None -> QQS default sizing (uniform target size, QQS's own size map, method 0)."""
    if h_rel is None:
        method = 0
        h = np.sqrt(vertex_areas(s).sum() / n_target)
    else:
        h = budget_scale(np.asarray(h_rel, float), s, n_target)
    r = _qqs_once(path, s, h, method)
    t = r.time_s
    if calibrate and len(r.Q) > 0:
        h = h * np.sqrt(len(r.Q) / n_target)
        r = _qqs_once(path, s, h, method)
        t += r.time_s
    r.time_s = t
    return r


def classical_sizes(s: SurfaceSample) -> dict:
    """Hand-designed relative size fields (the classical sizing families an engineer would try)."""
    kmax = s.attrs[:, s.attr_index["kmax"]]          # |k| in unit-diagonal units
    dfeat = s.attrs[:, s.attr_index["dfeat"]]
    return {
        "uniform": np.ones(len(s.V)),
        "curv_hi": 1.0 / (1.0 + 0.1 * kmax),
        "feat": np.clip(0.4 + 15.0 * dfeat, 0.4, 1.0),
        "curv_feat": np.minimum(1.0 / (1.0 + 0.05 * kmax), np.clip(0.4 + 15.0 * dfeat, 0.4, 1.0)),
    }


def two_pass(path: str, s: SurfaceSample, h_rel=None, method: int = 4, n_target: int = N_TARGET) -> dict:
    """Run QQS at two budgets (first guess, then rescaled by sqrt(n1/n_target)); returns raw metrics of both.
    h_rel None -> QQS default sizing (method 0, uniform target size)."""
    from .metrics import quad_metrics
    if h_rel is None:
        method_used, h = 0, np.sqrt(vertex_areas(s).sum() / n_target)
    else:
        method_used, h = method, budget_scale(np.asarray(h_rel, float), s, n_target)
    h = h * QQS_FIRST_PASS_FACTOR
    r1 = _qqs_once(path, s, h, method_used)
    m1 = quad_metrics(r1, s)
    m2, t2 = {"valid": False}, 0.0
    if m1.get("valid"):
        r2 = _qqs_once(path, s, h * np.sqrt(len(r1.Q) / n_target), method_used)
        m2, t2 = quad_metrics(r2, s), r2.time_s
    return {"p1": m1, "p2": m2, "time_s": r1.time_s + t2}


MAX_EXTRAP = np.log(1.5)


def at_budget(tp: dict, n_star: float) -> dict:
    """Metrics of a two-pass result linearly inter/extrapolated in log(n_quads) to n_star.
    Extrapolation beyond 1.5x of the nearest pass marks matched=False (still returned)."""
    ps = [p for p in (tp["p1"], tp["p2"]) if p.get("valid")]
    if not ps:
        return {"valid": False, "matched": False, "time_s": tp.get("time_s", 0.0)}
    xt = np.log(n_star)
    if len(ps) == 1 or abs(np.log(ps[0]["n_quads"]) - np.log(ps[1]["n_quads"])) < 1e-6:
        p = ps[-1]
        out = dict(p)
        out["matched"] = bool(abs(np.log(p["n_quads"]) - xt) <= MAX_EXTRAP)
    else:
        a, b = ps
        xa, xb = np.log(a["n_quads"]), np.log(b["n_quads"])
        lam = (xt - xa) / (xb - xa)
        out = {}
        for k, v in a.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and k in b:
                out[k] = (1 - lam) * v + lam * b[k]
            else:
                out[k] = v
        for k in ("sj_p05", "sj_mean", "quad_frac"):
            if k in out:
                out[k] = float(np.clip(out[k], -1, 1))
        for k in ("irregular_frac", "bad_frac", "hausdorff_rel", "chamfer_rel", "feat_dist_rel"):
            if k in out:
                out[k] = float(max(out[k], 0.0))
        bracketed = min(xa, xb) <= xt <= max(xa, xb)
        out["matched"] = bool(bracketed or min(abs(xa - xt), abs(xb - xt)) <= MAX_EXTRAP)
    out["n_quads"] = float(n_star)
    out["time_s"] = tp["time_s"]
    out["valid"] = True
    return out
