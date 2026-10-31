"""Quad-mesh one CAD part with the learned anytime policy and write the best mesh as OBJ.

1. budget-matched Instant Meshes pass (IM1) at --target-quads elements;
2. the policy predicts, from the 15-D shape descriptor + IM1's a-priori metrics, which extra attempts are worth
   their time: stop | +IMk (2 size-perturbed IM runs) | +QQS (Gmsh quasi-structured quads) | +both;
3. the attempt with the lowest composite a-priori score is written.
Example: python scripts/infer.py --step part.step --model checkpoints/anytime_policy.joblib --out part_quads.obj
"""
import argparse
import json
import time

from _common import load_cfg, rel


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", required=True, help="input CAD file (STEP/BREP, one solid)")
    ap.add_argument("--model", required=True, help="policy saved by scripts/train.py")
    ap.add_argument("--out", required=True, help="output quad mesh (.obj)")
    ap.add_argument("--summary", default=None, help="also write the JSON summary to this file")
    ap.add_argument("--config", default=rel("configs/anytime.toml"))
    ap.add_argument("--target-quads", type=int, default=None, help="element budget (default: config value)")
    ap.add_argument("--lam", type=float, default=None, help="override the saved time/quality trade-off")
    ap.add_argument("--action", default=None, choices=["stop", "+IMk", "+QQS", "+both"],
                    help="force an action instead of the policy decision")
    a = ap.parse_args()

    import numpy as np
    from anytime_mesh import policy as P
    from anytime_mesh.catalog import shape_descriptor
    from anytime_mesh.geometry import build_surface
    from anytime_mesh.labelling import CFG
    from anytime_mesh.meshers import run_instant_meshes, write_quad_obj
    from anytime_mesh.metrics import quad_metrics
    from anytime_mesh.qqs import run_qqs, vertex_areas
    from anytime_mesh.scoring import composite_score

    cfg = load_cfg(a.config)["inference"]
    pol = P.load_policy(a.model)
    lam = pol["lam"] if a.lam is None else a.lam
    n_star = a.target_quads or cfg["target_quads"]
    t0 = time.perf_counter()
    s = build_surface(a.step, rel_h=CFG["rel_h"])
    desc = shape_descriptor(a.step)
    if desc is None:
        raise SystemExit("input does not import as exactly one solid")

    def attempt(name, r):
        m = quad_metrics(r, s)
        return {"name": name, "result": r, "metrics": m, "score": composite_score(m)}

    # IM1: area-based edge length, one calibration rerun to hit the budget
    relh = np.sqrt(vertex_areas(s).sum() / n_star) / s.diag
    r = run_instant_meshes(s, relh)
    if len(r.Q):
        relh *= np.sqrt(len(r.Q) / n_star)
        t_first = r.time_s
        r = run_instant_meshes(s, relh)
        r.time_s += t_first
    tried = [attempt("IM1", r)]
    im1 = dict(tried[0]["metrics"], score=tried[0]["score"])
    if a.action:
        act = P.ACTIONS.index(a.action)
    else:
        act, _ = P.decide(pol["models"], P.features(desc, im1), lam, cfg["expected_time_s"])
    if P.ACTIONS[act] in ("+IMk", "+both"):
        for fac in (0.98, 1.02):
            tried.append(attempt(f"IM x{fac}", run_instant_meshes(s, relh * fac)))
    if P.ACTIONS[act] in ("+QQS", "+both"):
        tried.append(attempt("QQS", run_qqs(a.step, s, None, n_target=n_star)))
    best = min(tried, key=lambda t: t["score"])
    write_quad_obj(a.out, best["result"])
    summary = {"action": P.ACTIONS[act], "lam": lam, "target_quads": n_star,
               "attempts": {t["name"]: round(t["score"], 4) for t in tried}, "kept": best["name"],
               "n_quads": int(len(best["result"].Q)), "wall_s": round(time.perf_counter() - t0, 2),
               "metrics": {k: round(v, 5) for k, v in best["metrics"].items() if isinstance(v, float)}}
    print(json.dumps(summary, indent=1))
    if a.summary:
        with open(a.summary, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()
