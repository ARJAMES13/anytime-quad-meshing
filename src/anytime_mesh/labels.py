"""Label files (benchmark JSONL) -> per-part metrics at the equal element budget, plus markdown summaries.

QQS methods are evaluated at the per-solid budget n* = quad count QQS-default realises (its 2nd pass);
every other QQS method's two passes are inter/extrapolated in log(n) to n* (qqs.at_budget).
Failures (import/extract/timeout) score FAIL_SCORE and count as losses. Paired bootstrap vs a reference method.
"""
import collections
import json

import numpy as np

from .qqs import at_budget
from .scoring import FAIL_SCORE, composite_score, paired_bootstrap

KEYS = ["irregular_frac", "sj_p05", "bad_frac", "hausdorff_rel", "feat_dist_rel"]


def load(path):
    by = collections.defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        by[r["solid_id"]][r["method"]] = r
    out = {}
    for sid, ms in by.items():
        ref = ms.get("qqs_default")
        n_star = ref["tp"]["p2"].get("n_quads") if ref and "tp" in ref and ref["tp"]["p2"].get("valid") else None
        res = {}
        for m, r in ms.items():
            if "choice" in r:                     # learned configurator: result of the chosen config
                cm = "qqs_" + r["choice"]
                r = dict(ms.get(cm, {"valid": False}))
                r["total_time_s"] = r.get("total_time_s", np.nan)
            if "tp" in r:
                q = at_budget(r["tp"], n_star) if n_star else {"valid": False, "matched": False}
            else:
                q = r
            q = dict(q)
            q["score"] = composite_score(q)
            q["total_time_s"] = r.get("total_time_s", np.nan)
            res[m] = q
        out[sid] = res
    return out


def summarize(data, ref, methods=None):
    methods = methods or sorted({m for v in data.values() for m in v})
    sids = sorted(data)
    lines = [f"Solids: {len(sids)}. Reference: `{ref}`. Lower score is better (composite, pre-registered).", "",
             "| method | success | matched@n* | score mean | Δscore vs ref [95% CI] | W/T/L vs ref | " +
             " | ".join(KEYS) + " | time s |", "|" + "---|" * (8 + len(KEYS))]
    for m in methods:
        rows = [data[s].get(m, {"valid": False, "score": FAIL_SCORE}) for s in sids]
        refs = [data[s].get(ref, {"valid": False, "score": FAIL_SCORE}) for s in sids]
        sc = np.array([r["score"] for r in rows]); rs = np.array([r["score"] for r in refs])
        ok = np.array([bool(r.get("valid")) for r in rows])
        matched = np.mean([bool(r.get("matched", True)) for r in rows if r.get("valid")]) if ok.any() else 0
        md, lo, hi = paired_bootstrap(sc, rs) if m != ref else (0.0, 0.0, 0.0)
        tol = 0.005
        w, t, l_ = int(np.sum(sc < rs - tol)), int(np.sum(np.abs(sc - rs) <= tol)), int(np.sum(sc > rs + tol))
        both = ok & np.array([bool(r.get("valid")) for r in refs])
        km = [np.mean([rows[i][k] for i in np.where(both)[0]]) if both.any() else np.nan for k in KEYS]
        tm = np.nanmean([r.get("total_time_s", np.nan) for r in rows]) if ok.any() else np.nan
        lines.append(f"| {m} | {ok.mean():.1%} | {matched:.0%} | {sc.mean():.4f} | {md:+.4f} [{lo:+.4f}, {hi:+.4f}] | "
                     f"{w}/{t}/{l_} | " + " | ".join(f"{v:.4f}" for v in km) + f" | {tm:.1f} |")
    return "\n".join(lines)


def by_source(data, ref, methods):
    src = collections.defaultdict(list)
    for s in data:
        src[s.split("/")[0]].append(s)
    out = []
    for k, ss in sorted(src.items()):
        out.append(f"\n**{k}** ({len(ss)} solids)\n")
        out.append(summarize({s: data[s] for s in ss}, ref, methods))
    return "\n".join(out)

