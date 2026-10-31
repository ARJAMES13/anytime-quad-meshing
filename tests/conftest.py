import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def _write_shape(path, builder):
    import gmsh
    gmsh.initialize(readConfigFiles=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        builder(gmsh.model.occ)
        gmsh.model.occ.synchronize()
        gmsh.write(path)
    finally:
        gmsh.finalize()
    return path


@pytest.fixture(scope="session")
def shapes(tmp_path_factory):
    d = tmp_path_factory.mktemp("shapes")
    return {
        "cube": _write_shape(str(d / "cube.step"), lambda o: o.addBox(0, 0, 0, 1, 1, 1)),
        "plate": _write_shape(str(d / "plate.step"),
                              lambda o: o.cut([(3, o.addBox(0, 0, 0, 4, 3, 0.5))], [(3, o.addCylinder(2, 1.5, -1, 0, 0, 3, 0.6))])),
    }
