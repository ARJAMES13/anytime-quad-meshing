"""Regenerate the README figures from the committed labels (and, for the teaser, one inference run).

  results/figures/forest_test.png   Δ mean composite score vs QQS with 95 % CI, frozen test set
  results/figures/qqs_chaos.png     QQS score under ±1-4 % size perturbations (test parts with all 5 runs)
  results/figures/teaser.png        input CAD triangulation -> output quad mesh (needs --step and --model)
Example: python scripts/make_figures.py --step part.step --model checkpoints/anytime_policy.joblib
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

from _common import ROOT, load_cfg, rel


def forest(out, cfg):
    import matplotlib.pyplot as plt
    import numpy as np
    from anytime_mesh import policy as P
    from anytime_mesh.scoring import paired_bootstrap
    pc, paths = cfg["policy"], cfg["paths"]
    desc = P.load_descriptors(rel(paths["descriptors"]))
    Xtr, Str, _, _ = P.table(rel(paths["train"]), desc)
    Xv, Sv, Tv, _ = P.table(rel(paths["val"]), desc)
    Xt, St, Tt, Qt = P.table(rel(paths["test"]), desc)
    models = P.fit(Xtr, Str, pc.get("gbm"), pc["clip"])
    lam, _ = P.choose_lambda(models, Xv, Sv, Tv, pc["lambdas"], pc["val_tolerance"])
    rows = [("Instant Meshes, 1 run", St[:, 0], Tt[:, 0]), ("IM best of 3", St[:, 1], Tt[:, 1]),
            ("run IM + QQS, keep best", St[:, 2], Tt[:, 2]), ("run everything, keep best", St[:, 3], Tt[:, 3])]
    v, t, _ = P.run_policy(models, Xt, St, Tt, lam)
    rows.insert(2, (f"learned anytime policy (λ={lam})", v, t))
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for k, (name, s, tt) in enumerate(rows[::-1]):
        md, lo, hi = paired_bootstrap(s, Qt, n=10000)
        w, l_ = int((s < Qt - .005).sum()), int((s > Qt + .005).sum())
        c = "#1b9e77" if "policy" in name else "#555555"
        ax.plot([lo, hi], [k, k], color=c, lw=2.5); ax.plot(md, k, "o", color=c, ms=7)
        ax.text(0.13, k, f"{md:+.3f}  W/L {w}/{l_}  {np.nanmean(tt):.1f} s", va="center", fontsize=8)
    ax.axvline(0, color="#2166ac", lw=1.2)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=9)
    ax.set_xlim(-0.2, 0.3)
    ax.set_xlabel(f"Δ mean composite score vs QQS (95 % CI), {len(Qt)} test parts; left is better (QQS {np.nanmean(Tt[:, 2] - Tt[:, 0]):.1f} s)",
                  fontsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def chaos(out, cfg):
    import matplotlib.pyplot as plt
    import numpy as np
    from anytime_mesh.labels import load
    d = load(rel(cfg["paths"]["test"]))
    pert = ["qqs_pert1.01", "qqs_pert1.02", "qqs_pert1.03", "qqs_pert1.04"]
    rows = []
    for m in d.values():
        if "qqs_default" in m and all(p in m for p in pert) and m["qqs_default"].get("valid"):
            s = [m["qqs_default"]["score"]] + [m[p]["score"] for p in pert]
            if max(s) < 10:
                rows.append(s)
    S = np.array(rows)
    S = S[np.argsort(S.max(1) - S.min(1))]
    x = np.arange(len(S))
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.vlines(x, S.min(1), S.max(1), color="#cccccc", lw=1.5)
    ax.plot(x, S[:, 0], "o", ms=3.5, color="#2166ac", label="QQS default size")
    ax.plot(x, S.min(1), "_", ms=7, color="#1b9e77", label="best of default + 4 size perturbations (1-4 %)")
    ax.set_yscale("log"); ax.set_xlabel("test parts, sorted by spread"); ax.set_ylabel("composite score (log)")
    ax.set_title(f"QQS is chaotic: ±1-4 % size changes swing quality (median spread {np.median(S.max(1) - S.min(1)):.3f})",
                 fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def _draw(ax, V, faces, title, color):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    polys = [V[f] for f in faces]
    ax.add_collection3d(Poly3DCollection(polys, facecolor=color, edgecolor="#222222", linewidths=0.15, alpha=1.0))
    lo, hi = V.min(0), V.max(0); c = (lo + hi) / 2; r = (hi - lo).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    ax.view_init(elev=28, azim=-55); ax.set_box_aspect(None, zoom=1.3); ax.set_axis_off(); ax.set_title(title, fontsize=10)


def teaser(out, step, model):
    import matplotlib.pyplot as plt
    import numpy as np
    from anytime_mesh.geometry import build_surface
    from anytime_mesh.meshers import _read_poly_obj
    with tempfile.TemporaryDirectory() as td:
        obj, js = os.path.join(td, "out.obj"), os.path.join(td, "summary.json")
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "infer.py"), "--step", step, "--model", model,
                        "--out", obj, "--summary", js], capture_output=True, text=True, check=True)
        V, faces = _read_poly_obj(obj)
        info = json.load(open(js, encoding="utf-8"))
    s = build_surface(step, rel_h=0.012)
    fig = plt.figure(figsize=(9, 3.6))
    _draw(fig.add_subplot(1, 2, 1, projection="3d"), s.V, list(s.F), "input CAD (reference triangulation)", "#d9d9d9")
    _draw(fig.add_subplot(1, 2, 2, projection="3d"), np.asarray(V), faces,
          f"output: {info['n_quads']} quads, kept {info['kept']} (action {info['action']})", "#a6cee3")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.92, wspace=0)
    fig.savefig(out, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=rel("configs/anytime.toml"))
    ap.add_argument("--out-dir", default=rel("results/figures"))
    ap.add_argument("--step", default=None, help="CAD part for the teaser (skipped if omitted)")
    ap.add_argument("--model", default=None, help="saved policy for the teaser")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    os.makedirs(a.out_dir, exist_ok=True)
    cfg = load_cfg(a.config)
    forest(os.path.join(a.out_dir, "forest_test.png"), cfg)
    chaos(os.path.join(a.out_dir, "qqs_chaos.png"), cfg)
    if a.step and a.model:
        teaser(os.path.join(a.out_dir, "teaser.png"), a.step, a.model)
    print("figures written to", a.out_dir)


if __name__ == "__main__":
    main()
