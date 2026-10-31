"""Evaluate the anytime policy on the frozen TEST set (or the HexMe OOD set) against QQS alone.

Prints paired statistics for QQS alone, the four fixed action sequences and the policy at every lambda in the
config grid; the VAL-chosen lambda is marked. Without --model the policy is trained first (deterministic, ~1 min).
Examples:
  python scripts/eval.py                       # frozen test set
  python scripts/eval.py --split ood           # HexMe out-of-distribution set
  python scripts/eval.py --model checkpoints/anytime_policy.joblib
"""
import argparse

from _common import load_cfg, rel, report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=rel("configs/anytime.toml"), help="policy config (TOML)")
    ap.add_argument("--split", default="test", choices=["test", "ood"], help="evaluation set")
    ap.add_argument("--labels", default=None, help="override the evaluation label file")
    ap.add_argument("--model", default=None, help="saved policy from scripts/train.py (default: train now)")
    a = ap.parse_args()

    import numpy as np
    from anytime_mesh import policy as P
    cfg = load_cfg(a.config)
    pc, paths = cfg["policy"], cfg["paths"]
    desc = P.load_descriptors(rel(paths["descriptors"]))
    Xtr, Str, _, _ = P.table(rel(paths["train"]), desc)
    Xv, Sv, Tv, _ = P.table(rel(paths["val"]), desc)
    test_path = a.labels or rel(paths[a.split])
    cands = []
    Xt, St, Tt, Qt = P.table(test_path, desc, cands=cands)
    print(f"train {len(Xtr)} val {len(Xv)} test {len(Xt)} ({a.labels or paths[a.split]})")
    models = P.load_policy(a.model)["models"] if a.model else P.fit(Xtr, Str, pc.get("gbm"), pc["clip"])
    lams = pc["lambdas"]
    lam_star, val = P.choose_lambda(models, Xv, Sv, Tv, lams, pc["val_tolerance"])
    print("val sweep (lam, mean score, mean time):", [(l, round(s, 4), round(t, 2)) for l, s, t in val], "-> lam*", lam_star)

    report("QQS alone", Qt, Qt, Tt[:, 2] - Tt[:, 0])
    for k, name in enumerate(P.ACTIONS):
        report(f"fixed: IM1 then {name}", St[:, k], Qt, Tt[:, k])
    for lam in lams:
        v, t, act = P.run_policy(models, Xt, St, Tt, lam)
        tag = " <- val-chosen" if lam == lam_star else ""
        report(f"ML policy lam={lam}{tag}", v, Qt, t)
        if lam == lam_star:
            print("    action mix:", {P.ACTIONS[k]: round(float(np.mean(act == k)), 2) for k in range(4)})
            act_star = act

    # per-metric breakdown of the kept meshes (parts where both QQS and the policy's mesh are valid)
    keys = ["irregular_frac", "bad_frac", "sj_p05", "hausdorff_rel", "feat_dist_rel", "chamfer_rel"]
    pol = P.chosen_metrics(cands, act_star)
    qqs = [c[2] for c in cands]
    ok = [i for i in range(len(pol)) if pol[i].get("valid") and qqs[i].get("valid")]
    print(f"component means at lam*={lam_star} over {len(ok)} parts valid for both (lower is better except sj_p05):")
    for name, rows in (("QQS alone", qqs), ("ML policy", pol)):
        print(f"  {name:12s}" + "".join(f"  {k} {np.mean([rows[i][k] for i in ok]):.5f}" for k in keys))


if __name__ == "__main__":
    main()
