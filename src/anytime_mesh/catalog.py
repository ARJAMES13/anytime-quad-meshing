"""Solid catalog: enumerate CAD solids from the local datasets and compute cheap shape descriptors.

Descriptors are used only to group near-duplicate / same-family solids before splitting
(so the frozen test set never shares a family with training data).
"""
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

# file extensions per source; the root directory of each source comes from configs/data.toml
SOURCE_EXTS = {
    "abc": (".step", ".stp"),
    "set_b": (".step", ".stp"),
    "simjeb": (".step", ".stp"),
    "cad_models": (".step", ".stp"),
    "hexme": (".step", ".stp", ".brep"),
}

SURFACE_TYPES = ["Plane", "Cylinder", "Cone", "Sphere", "Torus", "BSpline surface",
                 "Surface of Revolution", "Surface of Extrusion"]


@dataclass(frozen=True)
class SolidRef:
    id: str
    path: str
    source: str


def list_solids(roots: Dict[str, str]) -> List[SolidRef]:
    """roots: {source name: local directory}. Part id = '<source>/<path relative to the root>'."""
    refs = []
    for src, root in roots.items():
        exts = SOURCE_EXTS.get(src, (".step", ".stp", ".brep"))
        if not os.path.isdir(root):
            continue
        for dirpath, _, files in os.walk(root):
            for f in sorted(files):
                if f.lower().endswith(exts):
                    p = os.path.join(dirpath, f).replace("\\", "/")
                    rel = os.path.relpath(p, root).replace("\\", "/")
                    refs.append(SolidRef(id=f"{src}/{rel}", path=p, source=src))
    return sorted(refs, key=lambda r: r.id)


def family_key(ref: SolidRef) -> str:
    """Name-based family: solids that share a design prefix are one family regardless of geometry."""
    name = os.path.basename(ref.id)
    if ref.source == "set_b":                # <design>_<hash>_<body>.stp
        return f"set_b:{name.split('_')[0]}"
    if ref.source == "abc":                  # <8-digit id>/<...>.step ; ids are per Onshape document
        parts = ref.id.split("/")
        return f"abc:{parts[1] if len(parts) > 2 else name}"
    if ref.source == "simjeb":               # all brackets share one design brief
        return "simjeb:bracket"
    if ref.source == "cad_models":           # bearing_block_00 -> bearing_block
        return "cad_models:" + re.sub(r"_?\d+$", "", os.path.splitext(name)[0])
    return f"{ref.source}:{os.path.splitext(name)[0]}"


def shape_descriptor(path: str, scale: float = 1.0) -> Optional[np.ndarray]:
    """Scale-invariant descriptor (area-weighted surface-type histogram, counts, bbox/area/volume ratios).

    Returns None when the file does not import to exactly one solid.
    `scale` exists for testing scale invariance.
    """
    import gmsh
    gmsh.initialize(readConfigFiles=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Geometry.OCCScaling", scale)
        gmsh.model.occ.importShapes(path)
        gmsh.model.occ.synchronize()
        vols = gmsh.model.getEntities(3)
        faces = gmsh.model.getEntities(2)
        if len(vols) != 1 or not faces:
            return None
        edges = gmsh.model.getEntities(1)
        area_by_type = np.zeros(len(SURFACE_TYPES) + 1)
        for _, tag in faces:
            t = gmsh.model.getType(2, tag)
            k = SURFACE_TYPES.index(t) if t in SURFACE_TYPES else len(SURFACE_TYPES)
            area_by_type[k] += gmsh.model.occ.getMass(2, tag)
        total_area = area_by_type.sum()
        vol = gmsh.model.occ.getMass(3, vols[0][1])
        x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(3, vols[0][1])
        ext = np.sort([x1 - x0, y1 - y0, z1 - z0])[::-1]
        diag = float(np.linalg.norm(ext))
        if diag <= 0 or total_area <= 0:
            return None
        return np.concatenate([
            area_by_type / total_area,
            [np.log1p(len(faces)), np.log1p(len(edges))],
            ext[1:] / ext[0],
            [total_area / diag ** 2, abs(vol) / diag ** 3],
        ])
    except Exception:
        return None
    finally:
        gmsh.finalize()


def _descriptor_worker(ref: SolidRef):
    d = shape_descriptor(ref.path)
    return ref, (None if d is None else d.tolist())


def compute_descriptors(refs: List[SolidRef], processes: int = 10):
    """Parallel descriptors; returns (kept_refs, desc_array). Solids that fail import are dropped here
    (they are never part of any split)."""
    from multiprocessing import Pool
    kept, desc = [], []
    with Pool(processes, maxtasksperchild=50) as pool:
        for ref, d in pool.imap(_descriptor_worker, refs, chunksize=4):
            if d is not None:
                kept.append(ref)
                desc.append(d)
    return kept, np.array(desc)
