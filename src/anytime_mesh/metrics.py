"""Quad mesh quality metrics against the reference CAD surface sample (no simulation)."""
import numpy as np
from scipy.spatial import cKDTree

from .geometry import SurfaceSample


def _sample_tris(V, F, n_per=3, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    a, b = rng.random((2, len(F), n_per))
    flip = a + b > 1
    a[flip], b[flip] = 1 - a[flip], 1 - b[flip]
    P = V[F[:, 0], None] + a[..., None] * (V[F[:, 1]] - V[F[:, 0]])[:, None] + b[..., None] * (V[F[:, 2]] - V[F[:, 0]])[:, None]
    return P.reshape(-1, 3)


def _quad_corner_stats(V, Q):
    """Scaled Jacobian (min over corners, sign w.r.t. quad normal) and corner angles in degrees."""
    P = V[Q]                                                   # (q,4,3)
    nrm = np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 1])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-30
    sj, ang = [], []
    for k in range(4):
        e1 = P[:, (k + 1) % 4] - P[:, k]
        e2 = P[:, (k - 1) % 4] - P[:, k]
        l1 = np.linalg.norm(e1, axis=1); l2 = np.linalg.norm(e2, axis=1)
        c = np.cross(e1, e2)
        sj.append(np.sum(c * nrm, 1) / (l1 * l2 + 1e-30))
        ang.append(np.degrees(np.arccos(np.clip(np.sum(e1 * e2, 1) / (l1 * l2 + 1e-30), -1, 1))))
    return np.min(sj, axis=0), np.stack(ang, 1)


def quad_metrics(r, s: SurfaceSample) -> dict:
    out = {"status": r.status, "time_s": float(r.time_s)}
    if r.status != "ok" or len(r.Q) == 0:
        out["valid"] = False
        return out
    V, Q, T = r.V, r.Q, r.T
    n_faces = len(Q) + len(T) + r.other
    sj, ang = _quad_corner_stats(V, Q)
    # valence over all faces; vertices near sharp features / boundaries excluded from irregularity
    val = np.zeros(len(V), int)
    for f in (Q, T):
        if len(f):
            np.add.at(val, f.ravel(), 1)
    if len(s.feat_edges):
        t = np.linspace(0, 1, 9)[:, None, None]
        fpts = (s.V[s.feat_edges[:, 0]][None] * (1 - t) + s.V[s.feat_edges[:, 1]][None] * t).reshape(-1, 3)
        edge_len = np.median(np.linalg.norm(V[Q[:, 1]] - V[Q[:, 0]], axis=1))
        near_feat = cKDTree(fpts).query(V)[0] < 0.5 * edge_len
    else:
        fpts = np.zeros((0, 3)); near_feat = np.zeros(len(V), bool)
    interior = ~near_feat & (val > 0)
    irregular = interior & (val != 4)
    # geometric fidelity: symmetric distances between output surface and reference triangulation
    import igl
    Tq = np.r_[Q[:, [0, 1, 2]], Q[:, [0, 2, 3]], T] if len(T) else np.r_[Q[:, [0, 1, 2]], Q[:, [0, 2, 3]]]
    Pm = np.r_[V, _sample_tris(V, Tq)]
    Ps = np.r_[s.V, _sample_tris(s.V, s.F)]
    d_ms = np.sqrt(igl.point_mesh_squared_distance(Pm, s.V, s.F.astype(np.int64))[0])
    d_sm = np.sqrt(igl.point_mesh_squared_distance(Ps, V.astype(np.float64), Tq.astype(np.int64))[0])
    # feature preservation: distance from CAD sharp-edge samples to the output mesh edges
    if len(fpts):
        E = np.r_[Q[:, [0, 1]], Q[:, [1, 2]], Q[:, [2, 3]], Q[:, [3, 0]]]
        tt = np.linspace(0, 1, 41)[:, None, None]
        epts = (V[E[:, 0]][None] * (1 - tt) + V[E[:, 1]][None] * tt).reshape(-1, 3)
        feat_dist = cKDTree(epts).query(fpts)[0].mean() / s.diag
    else:
        feat_dist = 0.0
    out.update({
        "valid": True,
        "n_elems": int(n_faces), "n_quads": int(len(Q)), "n_verts": int(len(V)),
        "quad_frac": float(len(Q) / n_faces),
        "sj_min": float(sj.min()), "sj_p05": float(np.quantile(sj, 0.05)), "sj_mean": float(sj.mean()),
        "bad_frac": float(np.mean(sj < 0.3)),
        "angle_dev_mean": float(np.mean(np.abs(ang - 90.0))),
        "irregular_frac": float(irregular.sum() / max(interior.sum(), 1)),
        "hausdorff_rel": float(max(d_ms.max(), d_sm.max()) / s.diag),
        "chamfer_rel": float(0.5 * (d_ms.mean() + d_sm.mean()) / s.diag),
        "feat_dist_rel": float(feat_dist),
    })
    return out
