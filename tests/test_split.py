import numpy as np
import pytest

from anytime_mesh.catalog import SolidRef, family_key, shape_descriptor
from anytime_mesh.split import make_split


def _refs(n_clusters=40, per=6, seed=0):
    rng = np.random.default_rng(seed)
    refs, desc = [], []
    for c in range(n_clusters):
        centre = rng.normal(size=8) * 10
        for k in range(per):
            refs.append(SolidRef(id=f"s{c}_{k}", path="", source="abc"))
            desc.append(centre + rng.normal(size=8) * 0.01)
    return refs, np.array(desc)


def test_clusters_disjoint():
    refs, desc = _refs()
    split = make_split(refs, desc, n_test=60, n_val=30, seed=0)
    where = {}
    for name, ids in split.items():
        if name.startswith("_"):
            continue
        for i in ids:
            fam = split["_family"][i]
            assert where.setdefault(fam, name) == name, f"family {fam} spans splits"


def test_test_size_at_least_50():
    refs, desc = _refs()
    split = make_split(refs, desc, n_test=60, n_val=30, seed=0)
    assert len(split["test"]) >= 50


def test_deterministic_seed():
    refs, desc = _refs()
    a = make_split(refs, desc, n_test=60, n_val=30, seed=0)
    b = make_split(refs, desc, n_test=60, n_val=30, seed=0)
    assert a["test"] == b["test"] and a["val"] == b["val"]


def test_name_family_overrides_descriptor():
    # same design prefix must share a family even if geometry differs
    refs = [SolidRef(id="set_b/100108_aaa_1", path="", source="set_b"),
            SolidRef(id="set_b/100108_bbb_2", path="", source="set_b")]
    assert family_key(refs[0]) == family_key(refs[1])


def test_descriptor_scale_invariant(shapes):
    d = shape_descriptor(shapes["plate"])
    assert d is not None and np.all(np.isfinite(d))
    d2 = shape_descriptor(shapes["plate"], scale=1000.0)
    assert np.allclose(d, d2, atol=1e-6)
