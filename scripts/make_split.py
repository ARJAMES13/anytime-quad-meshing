"""Compute shape descriptors for all local CAD solids and write a family-grouped train/val/test split.

The published split (configs/splits/v1.json) and descriptors (results/features/descriptors_v1.npz) are frozen;
this script documents how they were made and refuses to overwrite an existing split.
Example: python scripts/make_split.py --out configs/splits/my_split.json --descriptors my_descriptors.npz
"""
import argparse
import os

from _common import rel

MIN_FACES, MAX_FACES = 4, 400


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-config", default=rel("configs/data.toml"), help="dataset roots (TOML)")
    ap.add_argument("--out", required=True, help="split JSON to write (must not exist)")
    ap.add_argument("--descriptors", required=True, help="descriptor .npz to write")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--n-test", type=int, default=150)
    ap.add_argument("--n-val", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists - splits are frozen; refusing to overwrite")

    import numpy as np
    from anytime_mesh.catalog import compute_descriptors, list_solids
    from anytime_mesh.data import load_config, source_roots
    from anytime_mesh.split import make_split, save_split
    refs = list_solids(source_roots(load_config(a.data_config)))
    print("listed", len(refs), flush=True)
    kept, desc = compute_descriptors(refs, processes=a.procs)
    nf = np.expm1(desc[:, 9]).round()
    ok = (nf >= MIN_FACES) & (nf <= MAX_FACES)
    kept = [r for r, k in zip(kept, ok) if k]
    desc = desc[ok]
    print("kept", len(kept), flush=True)
    np.savez_compressed(a.descriptors, ids=np.array([r.id for r in kept]), desc=desc,
                        sources=np.array([r.source for r in kept]))
    split = make_split(kept, desc, n_test=a.n_test, n_val=a.n_val, seed=a.seed)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    save_split(split, a.out)
    print({k: len(v) for k, v in split.items() if not k.startswith("_")})


if __name__ == "__main__":
    main()
