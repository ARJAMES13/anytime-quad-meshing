"""Headline table on the frozen test set: single methods + select-the-best-a-priori-score hybrids.

Select-by-score is a legitimate a-priori method: the composite score needs only the mesh and the CAD
(no simulation, no ground truth), so a mesher can run k variants and keep the best-scoring one.
Usage: python scripts/eval_baselines.py [--labels results/labels/test_v1.jsonl] [--out results/logs/baselines_test.md]
"""
import argparse

import numpy as np

from _common import rel
from anytime_mesh.labels import KEYS, load
from anytime_mesh.scoring import FAIL_SCORE, paired_bootstrap

PERT = ["qqs_pert1.01", "qqs_pert1.02", "qqs_pert1.03", "qqs_pert1.04"]
CONFIGS = ["qqs_default", "qqs_uniform", "qqs_curv_hi", "qqs_feat", "qqs_curv_feat"]


def pick_best(d, members):
    rows = [d[m] for m in members if m in d]
    rows = [r for r in rows if r.get("valid")] or [{"valid": False, "score": FAIL_SCORE}]
    return min(rows, key=lambda r: r["score"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default=rel("results/labels/test_v1.jsonl"), help="label file with QQS variants")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    data = load(a.labels)
    for sid, d in data.items():
        d["perturb&select k=2"] = pick_best(d, ["qqs_default"] + PERT[:1])
        d["perturb&select k=5"] = pick_best(d, ["qqs_default"] + PERT)
        d["mesh&select k=5 (configs)"] = pick_best(d, CONFIGS)
        d["select over all 9 QQS runs"] = pick_best(d, CONFIGS + PERT)
    methods = ["qqs_default", "qqs_uniform", "qqs_curv_hi", "qqs_feat", "qqs_curv_feat", "im_default", "gmsh_blossom",
               "perturb&select k=2", "perturb&select k=5", "mesh&select k=5 (configs)", "select over all 9 QQS runs"]
    sids = sorted(data)
    ref = [data[s].get("qqs_default", {"score": FAIL_SCORE}) for s in sids]
    rs = np.array([r["score"] for r in ref])
    lines = [f"Frozen test set: {len(sids)} solids. Reference = QQS default. Composite score: lower is better.", "",
             "| method | QQS runs | success | score mean | median | Δ vs QQS [95% CI] | W/T/L | " + " | ".join(KEYS) + " |",
             "|" + "---|" * (8 + len(KEYS))]
    runs = {"perturb&select k=2": 2, "perturb&select k=5": 5, "mesh&select k=5 (configs)": 5, "select over all 9 QQS runs": 9}
    for m in methods:
        rows = [data[s].get(m, {"valid": False, "score": FAIL_SCORE}) for s in sids]
        sc = np.array([r["score"] for r in rows])
        ok = np.array([bool(r.get("valid")) for r in rows])
        md, lo, hi = paired_bootstrap(sc, rs) if m != "qqs_default" else (0, 0, 0)
        w = int(np.sum(sc < rs - 0.005)); t = int(np.sum(np.abs(sc - rs) <= 0.005)); l_ = int(np.sum(sc > rs + 0.005))
        both = ok & np.array([bool(r.get("valid")) for r in ref])
        km = [np.mean([rows[i].get(k, np.nan) for i in np.where(both)[0]]) for k in KEYS]
        nr = runs.get(m, 1 if m.startswith("qqs") else "-")
        lines.append(f"| {m} | {nr} | {ok.mean():.1%} | {sc.mean():.4f} | {np.median(sc):.4f} | {md:+.4f} [{lo:+.4f}, {hi:+.4f}] | "
                     f"{w}/{t}/{l_} | " + " | ".join(f"{v:.4f}" for v in km) + " |")
    md_txt = "\n".join(lines)
    print(md_txt)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(md_txt + "\n")


if __name__ == "__main__":
    main()
