"""Train the anytime policy on TRAIN labels, choose lambda on VAL, save the policy.

Example: python scripts/train.py --out checkpoints/anytime_policy.joblib
"""
import argparse
import os

from _common import load_cfg, rel


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=rel("configs/anytime.toml"), help="policy config (TOML)")
    ap.add_argument("--train", default=None, help="override train label file")
    ap.add_argument("--val", default=None, help="override val label file")
    ap.add_argument("--out", default=rel("checkpoints/anytime_policy.joblib"), help="where to save the policy")
    a = ap.parse_args()

    from anytime_mesh import policy as P
    cfg = load_cfg(a.config)
    pc, paths = cfg["policy"], cfg["paths"]
    desc = P.load_descriptors(rel(paths["descriptors"]))
    Xtr, Str, _, _ = P.table(a.train or rel(paths["train"]), desc)
    Xv, Sv, Tv, _ = P.table(a.val or rel(paths["val"]), desc)
    print(f"train {len(Xtr)} val {len(Xv)}")
    models = P.fit(Xtr, Str, pc.get("gbm"), pc["clip"])
    lam, val = P.choose_lambda(models, Xv, Sv, Tv, pc["lambdas"], pc["val_tolerance"])
    print("val sweep (lam, mean score, mean time):", [(l, round(float(s), 4), round(float(t), 2)) for l, s, t in val],
          "-> lam*", lam)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    P.save(a.out, models, lam, {"n_train": int(len(Xtr)), "n_val": int(len(Xv)),
                                "val_sweep": [(l, float(s), float(t)) for l, s, t in val]})
    print("saved", a.out, "(lam* =", lam, ")")


if __name__ == "__main__":
    main()
