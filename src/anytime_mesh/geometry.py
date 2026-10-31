"""Reference surface sample of a CAD solid: fine triangulation + per-vertex B-Rep attributes.

All lengths are made relative to the bounding-box diagonal so attributes are scale invariant.
Directional quantities (curvature directions) are kept separately as 3-vectors, never in `attrs`,
so `attrs` is rotation invariant.
"""
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

SURFACE_TYPES = ["Plane", "Cylinder", "Cone", "Sphere", "Torus", "BSpline surface",
                 "Surface of Revolution", "Surface of Extrusion"]
FEATURE_DIHEDRAL_DEG = 30.0

ATTR_NAMES = ["type_" + t.lower().replace(" ", "_") for t in SURFACE_TYPES] + ["type_other"] + [
    "kmax", "kmin", "ksum", "dfeat", "on_feature", "face_area_frac"]


class ImportFail(Exception):
    pass


@dataclass
class SurfaceSample:
    V: np.ndarray            # (n,3)
    F: np.ndarray            # (m,3) int
    tri_face_id: np.ndarray  # (m,) CAD face index per triangle
    feat_edges: np.ndarray   # (k,2) vertex pairs on sharp CAD edges / open boundary
    N: np.ndarray            # (n,3) unit vertex normals
    attrs: np.ndarray        # (n,d) rotation + scale invariant
    diag: float
    kdir: np.ndarray         # (n,3) unit max-curvature direction (zero where umbilic / planar)
    face_types: List[str] = field(default_factory=list)
    attr_index: Dict[str, int] = field(default_factory=lambda: {n: i for i, n in enumerate(ATTR_NAMES)})


def _tri_normals(V, F):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    return n  # length = 2*area


def _feature_edges(V, F, curve_segments):
    """Sharp edges: CAD curve segments whose two incident triangles meet at > threshold dihedral,
    or that have a single incident triangle (open boundary)."""
    tn = _tri_normals(V, F)
    tn = tn / (np.linalg.norm(tn, axis=1, keepdims=True) + 1e-30)
    e = np.sort(np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=1)
    tri = np.tile(np.arange(len(F)), 3)
    order = np.lexsort((e[:, 1], e[:, 0]))
    e, tri = e[order], tri[order]
    key = e[:, 0].astype(np.int64) * len(V) + e[:, 1]
    inc = {}
    for k, t in zip(key, tri):
        inc.setdefault(int(k), []).append(int(t))
    cos_thr = np.cos(np.radians(FEATURE_DIHEDRAL_DEG))
    out = []
    for a, b in curve_segments:
        a, b = (a, b) if a < b else (b, a)
        ts = inc.get(int(a) * len(V) + int(b), [])
        if len(ts) == 1 or (len(ts) == 2 and np.dot(tn[ts[0]], tn[ts[1]]) < cos_thr) or len(ts) > 2:
            out.append((a, b))
    return np.array(out, dtype=np.int64).reshape(-1, 2)


def import_cad(step_path: str, scale: float = 1.0, heal: bool = True):
    """Import a CAD file into the current (clear) gmsh model, optionally healed.

    Healing is kept only if it preserves a single solid; otherwise the file is re-imported unhealed
    (gmsh's healShapes can destroy e.g. periodic torus solids). Shared by every method so all see the
    same geometry. Returns (volume_tag, healed_flag); raises ImportFail.
    """
    import gmsh
    gmsh.option.setNumber("General.NumThreads", 1)   # parallelism comes from worker processes
    for do_heal in ((True, False) if heal else (False,)):
        gmsh.clear()
        gmsh.option.setNumber("Geometry.OCCScaling", scale)
        try:
            gmsh.model.occ.importShapes(step_path)
            if do_heal:
                gmsh.model.occ.healShapes()
            gmsh.model.occ.synchronize()
        except Exception as e:
            if do_heal:
                continue
            raise ImportFail(f"import: {e}")
        vols = gmsh.model.getEntities(3)
        if len(vols) == 1 and gmsh.model.getEntities(2):
            return vols[0][1], do_heal
    raise ImportFail(f"expected 1 volume, got {len(gmsh.model.getEntities(3))} "
                     f"(surfaces {len(gmsh.model.getEntities(2))})")


def build_surface(step_path: str, rel_h: float = 0.01, heal: bool = True) -> SurfaceSample:
    import gmsh
    gmsh.initialize(readConfigFiles=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        vtag, _ = import_cad(step_path, 1.0, heal=False)          # measure
        x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(3, vtag)
        diag = float(np.linalg.norm([x1 - x0, y1 - y0, z1 - z0]))
        if not diag > 0:
            raise ImportFail("degenerate bounding box")
        vtag, _ = import_cad(step_path, 1.0 / diag, heal=heal)     # unit-diagonal, healed
        surfs = gmsh.model.getEntities(2)
        h = rel_h  # model is now unit-diagonal
        for k, v in [("Mesh.MeshSizeMin", h), ("Mesh.MeshSizeMax", h), ("Mesh.MeshSizeFromCurvature", 0),
                     ("Mesh.MeshSizeExtendFromBoundary", 0), ("Mesh.Algorithm", 5)]:
            gmsh.option.setNumber(k, v)
        try:
            gmsh.model.mesh.generate(2)
        except Exception as e:
            raise ImportFail(f"surface mesh: {e}")

        tags, xyz, _ = gmsh.model.mesh.getNodes()
        V = xyz.reshape(-1, 3)
        tag2i = np.zeros(int(tags.max()) + 1, dtype=np.int64)
        tag2i[tags.astype(np.int64)] = np.arange(len(tags))

        F_list, fid_list, face_types, face_area = [], [], [], []
        kmax = np.zeros(len(V)); kmin = np.zeros(len(V)); kdir = np.zeros((len(V), 3)); kcnt = np.zeros(len(V))
        for fi, (_, stag) in enumerate(surfs):
            face_types.append(gmsh.model.getType(2, stag))
            face_area.append(gmsh.model.occ.getMass(2, stag))
            etypes, _, enodes = gmsh.model.mesh.getElements(2, stag)
            for et, en in zip(etypes, enodes):
                if et != 2:
                    continue
                T = tag2i[en.astype(np.int64)].reshape(-1, 3)
                F_list.append(T)
                fid_list.append(np.full(len(T), fi))
                vids = np.unique(T)
                uv = gmsh.model.getParametrization(2, stag, V[vids].ravel())
                c1, c2, d1, d2 = gmsh.model.getPrincipalCurvatures(stag, uv)
                c1, c2 = np.abs(np.asarray(c1)), np.abs(np.asarray(c2))
                d1 = np.asarray(d1).reshape(-1, 3); d2 = np.asarray(d2).reshape(-1, 3)
                big = np.maximum(c1, c2); small = np.minimum(c1, c2)
                dmax = np.where((c1 >= c2)[:, None], d1, d2)
                kmax[vids] += big; kmin[vids] += small; kcnt[vids] += 1
                # align sign before accumulating so opposite directions don't cancel
                s = np.sign(np.sum(kdir[vids] * dmax, axis=1)); s[s == 0] = 1
                kdir[vids] += dmax * s[:, None] * (big - small)[:, None]
        if not F_list:
            raise ImportFail("no triangles")
        F = np.concatenate(F_list); fid = np.concatenate(fid_list)
        kcnt[kcnt == 0] = 1
        kmax /= kcnt; kmin /= kcnt

        curve_segments = []
        for _, ctag in gmsh.model.getEntities(1):
            etypes, _, enodes = gmsh.model.mesh.getElements(1, ctag)
            for et, en in zip(etypes, enodes):
                if et == 1:
                    curve_segments.append(tag2i[en.astype(np.int64)].reshape(-1, 2))
        curve_segments = np.concatenate(curve_segments) if curve_segments else np.zeros((0, 2), np.int64)
    finally:
        gmsh.finalize()

    used = np.unique(F)
    remap = -np.ones(len(V), dtype=np.int64); remap[used] = np.arange(len(used))
    V, kmax, kmin, kdir = V[used], kmax[used], kmin[used], kdir[used]
    F = remap[F]
    curve_segments = remap[curve_segments]
    curve_segments = curve_segments[(curve_segments >= 0).all(1)]

    tn = _tri_normals(V, F)
    N = np.zeros_like(V)
    for j in range(3):
        np.add.at(N, F[:, j], tn)
    N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-30
    kdir = kdir - N * np.sum(kdir * N, axis=1, keepdims=True)
    kn = np.linalg.norm(kdir, axis=1, keepdims=True)
    kdir = np.where(kn > 1e-12, kdir / np.maximum(kn, 1e-30), 0.0)

    feat = _feature_edges(V, F, curve_segments)

    # per-vertex attributes
    ntype = len(SURFACE_TYPES) + 1
    tri_type = np.array([SURFACE_TYPES.index(t) if t in SURFACE_TYPES else ntype - 1 for t in face_types])[fid]
    onehot = np.zeros((len(V), ntype)); cnt = np.zeros(len(V))
    farea = (np.array(face_area) / max(sum(face_area), 1e-30))[fid]
    afrac = np.zeros(len(V))
    for j in range(3):
        np.add.at(onehot, (F[:, j], tri_type), 1.0)
        np.add.at(afrac, F[:, j], farea)
        np.add.at(cnt, F[:, j], 1.0)
    onehot /= cnt[:, None]; afrac /= cnt
    on_feat = np.zeros(len(V)); on_feat[np.unique(feat)] = 1.0
    if len(feat):
        from scipy.spatial import cKDTree
        t = np.linspace(0, 1, 5)[:, None, None]
        pts = (V[feat[:, 0]][None] * (1 - t) + V[feat[:, 1]][None] * t).reshape(-1, 3)
        dfeat = cKDTree(pts).query(V)[0]
    else:
        dfeat = np.ones(len(V))
    attrs = np.c_[onehot, kmax, kmin, kmax + kmin, dfeat, on_feat, afrac]  # unit-diagonal units
    V = V * diag  # back to model units
    return SurfaceSample(V=V, F=F, tri_face_id=fid, feat_edges=feat, N=N, attrs=attrs.astype(np.float64),
                         diag=diag, kdir=kdir, face_types=face_types)
