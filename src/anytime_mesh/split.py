"""Frozen, family-grouped train/val/test split.

Families = union of (a) name-based design families and (b) descriptor near-duplicate clusters.
A family never spans two splits. Test draws at most `max_per_family` solids per family so no
single design dominates the benchmark.
"""
import json
from typing import Dict, List

import numpy as np

from .catalog import SolidRef, family_key


class _DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def _families(refs: List[SolidRef], desc: np.ndarray, dup_tol: float) -> List[int]:
    from scipy.spatial import cKDTree
    n = len(refs)
    dsu = _DSU(n)
    first = {}
    for i, r in enumerate(refs):
        k = family_key(r)
        if k in first:
            dsu.union(first[k], i)
        else:
            first[k] = i
    if n > 1:
        z = (desc - desc.mean(0)) / (desc.std(0) + 1e-9)
        for i, j in cKDTree(z).query_pairs(dup_tol):
            dsu.union(i, j)
    return [dsu.find(i) for i in range(n)]


def make_split(refs: List[SolidRef], desc: np.ndarray, n_test: int = 150, n_val: int = 150,
               seed: int = 0, dup_tol: float = 0.15, max_per_family: int = 3,
               test_only_sources=("hexme",)) -> Dict:
    fam = _families(refs, desc, dup_tol)
    groups: Dict[int, List[int]] = {}
    for i, f in enumerate(fam):
        groups.setdefault(f, []).append(i)
    rng = np.random.default_rng(seed)
    order = sorted(groups)
    rng.shuffle(order)
    test, val, train = [], [], []
    ood = []
    for f in order:
        members = groups[f]
        if any(refs[i].source in test_only_sources for i in members):
            ood += members
            continue
        if len(test) < n_test:
            pick = sorted(members, key=lambda i: refs[i].id)
            rng.shuffle(pick)
            test += pick[:max_per_family]          # remaining members are discarded, never trained on
        elif len(val) < n_val:
            val += members[:max_per_family]
        else:
            train += members
    ids = lambda idx: sorted(refs[i].id for i in idx)
    return {
        "train": ids(train), "val": ids(val), "test": ids(test), "test_ood": ids(ood),
        "_family": {refs[i].id: f"f{fam[i]}" for i in range(len(refs))},
        "_params": {"seed": seed, "dup_tol": dup_tol, "max_per_family": max_per_family},
    }


def save_split(split: Dict, path: str) -> None:
    with open(path, "w") as fh:
        json.dump(split, fh, indent=1, sort_keys=True)
