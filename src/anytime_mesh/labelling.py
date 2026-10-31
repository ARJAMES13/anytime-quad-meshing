"""Mesh one part with one method and record its a-priori metrics (one JSONL row per (part, method)).

Methods
  qqs_default       Gmsh QQS with its own size map, two passes (budget calibration); defines the budget n*
  qqs_<name>        QQS + classical size field <name> from qqs.classical_sizes (uniform, curv_hi, feat, curv_feat)
  qqs_pert<f>       QQS default at its 2nd-pass size scaled by f, one pass (needs n1 of qqs_default)
  im_default        Instant Meshes at a fixed relative edge length
  im_matched        Instant Meshes calibrated to n* (needs n* of qqs_default)
  im_k3             best of 3 budget-matched Instant Meshes runs (size x1.00, x0.98, x1.02)
  gmsh_blossom      Gmsh frontal-Delaunay for quads + Blossom recombination
"""
import time
import traceback

import numpy as np

CFG = {"rel_h": 0.012, "rel_edge": 0.015}
QQS_BG_METHOD = 4   # QuadqsSizemapMethod for external size fields


def evaluate(solid_id, path, method, extra=None):
    """Return one result row; failures are recorded (status != ok), never raised."""
    from .geometry import ImportFail, build_surface
    from .meshers import run_gmsh_quads, run_instant_meshes
    from .metrics import quad_metrics
    from .qqs import N_TARGET, QQS_FIRST_PASS_FACTOR, _qqs_once, classical_sizes, two_pass, vertex_areas
    from .scoring import composite_score
    t0 = time.perf_counter()
    try:
        s = build_surface(path, rel_h=CFG["rel_h"])
    except ImportFail as e:
        return {"solid_id": solid_id, "method": method, "status": "import_fail", "valid": False, "msg": str(e)[:200]}
    t_surf = time.perf_counter() - t0
    m = method
    row = {"solid_id": solid_id, "method": m, "n_ref_verts": int(len(s.V))}
    tf = time.perf_counter()
    try:
        if m in ("im_matched", "im_k3"):
            n_star = extra["n_star"]
            rel = np.sqrt(vertex_areas(s).sum() / n_star) / s.diag
            r = run_instant_meshes(s, rel)
            if m == "im_matched":
                if len(r.Q):
                    rel *= np.sqrt(len(r.Q) / n_star)
                    t_first = r.time_s
                    r = run_instant_meshes(s, rel)
                    r.time_s += t_first
                row.update(quad_metrics(r, s))
            else:
                t_all = r.time_s
                if len(r.Q):
                    rel *= np.sqrt(len(r.Q) / n_star)
                best, ks = None, []
                for fac in (1.0, 0.98, 1.02):
                    r = run_instant_meshes(s, rel * fac); t_all += r.time_s
                    q = quad_metrics(r, s); sc = composite_score(q); ks.append(sc)
                    if best is None or sc < best[0]:
                        best = (sc, q)
                row.update(best[1]); row["k_scores"] = ks; row["time_s"] = t_all
        elif m.startswith("qqs_pert"):
            f = float(m[len("qqs_pert"):])
            h2 = np.sqrt(vertex_areas(s).sum() / N_TARGET) * QQS_FIRST_PASS_FACTOR * np.sqrt(extra["n1"] / N_TARGET)
            r = _qqs_once(path, s, h2 * f, 0)
            m1 = quad_metrics(r, s)
            row.update({"tp": {"p1": m1, "p2": {"valid": False}, "time_s": r.time_s}, "valid": bool(m1.get("valid")),
                        "status": "ok" if m1.get("valid") else "extract_fail", "pert": f})
        elif m.startswith("qqs_"):
            name = m[4:]
            h = None if name == "default" else classical_sizes(s)[name]
            tp = two_pass(path, s, h, method=QQS_BG_METHOD)
            row.update({"tp": tp, "valid": bool(tp["p2"].get("valid") or tp["p1"].get("valid")),
                        "status": "ok" if tp["p1"].get("valid") else "extract_fail"})
        elif m == "im_default":
            row.update(quad_metrics(run_instant_meshes(s, CFG["rel_edge"]), s))
        elif m == "gmsh_blossom":
            row.update(quad_metrics(run_gmsh_quads(path, s, CFG["rel_edge"], 8), s))
        else:
            raise ValueError(f"unknown method {m}")
    except MemoryError:   # out of RAM: infrastructure failure, rerun later, never a method loss
        row.update({"status": "infra_error", "valid": False, "msg": traceback.format_exc()[-300:]})
    except Exception:  # noqa: BLE001 — recorded as a failure of this method on this solid
        row.update({"status": "extract_fail", "valid": False, "msg": traceback.format_exc()[-300:]})
    row["field_time_s"] = 0.0
    row["total_time_s"] = t_surf + (time.perf_counter() - tf)
    # QQS rows are scored later at the per-part budget n* (labels.load); everything else is scored now
    row["score"] = None if "tp" in row else composite_score(row)
    return row
