"""Shared helpers for the command-line scripts (repo-root paths, config loading, report lines)."""
import os
import sys
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
for _stream in (sys.stdout, sys.stderr):      # results contain non-ASCII symbols (e.g. Δ); Windows consoles default to cp1252
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def load_cfg(path):
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def rel(path):
    """Resolve a config path relative to the repository root."""
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def report(name, v, ref, t, delta="Δ", tdig=2):
    """One paired-comparison line: mean, median, Δmean vs ref with 95 % bootstrap CI, W/L (tie ±0.005), sign test."""
    import numpy as np
    from scipy.stats import binomtest
    from anytime_mesh.scoring import paired_bootstrap
    md, lo, hi = paired_bootstrap(v, ref, n=10000)
    w, l_ = int((v < ref - .005).sum()), int((v > ref + .005).sum())
    p = binomtest(w, w + l_, .5).pvalue if w + l_ else 1
    print(f"  {name:34s} mean {v.mean():.4f} median {np.median(v):.4f}  {delta} {md:+.4f} [{lo:+.4f},{hi:+.4f}]  "
          f"W/L {w}/{l_} p {p:.1e}  time {np.nanmean(t):.{tdig}f}s")
