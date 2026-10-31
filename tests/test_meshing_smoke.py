"""Smoke tests for the CAD -> surface sample -> quad mesh -> metrics pipeline on tiny generated parts."""
import numpy as np
import pytest

from anytime_mesh.geometry import build_surface
from anytime_mesh.meshers import run_gmsh_quads
from anytime_mesh.metrics import quad_metrics
from anytime_mesh.qqs import run_qqs
from anytime_mesh.scoring import composite_score


@pytest.fixture(scope="module")
def plate(shapes):
    return shapes["plate"], build_surface(shapes["plate"], rel_h=0.03)


def test_surface_sample(plate):
    _, s = plate
    assert len(s.F) > 100 and len(s.feat_edges) > 0 and np.isfinite(s.attrs).all()


def test_blossom_quads_and_metrics(plate):
    path, s = plate
    m = quad_metrics(run_gmsh_quads(path, s, 0.05, 8), s)
    assert m["valid"] and m["n_quads"] > 50 and 0 <= composite_score(m) < 10


def test_qqs_budget(plate):
    path, s = plate
    r = run_qqs(path, s, None, n_target=400)
    assert len(r.Q) > 0 and 0.5 < len(r.Q) / 400 < 2.0


def test_instant_meshes(plate):
    pytest.importorskip("pyinstantmeshes")
    from anytime_mesh.meshers import run_instant_meshes
    path, s = plate
    m = quad_metrics(run_instant_meshes(s, 0.05), s)
    assert m["valid"] and m["quad_frac"] > 0.9
