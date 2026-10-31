# Data

No CAD files are committed. You only need them to **re-mesh** parts (`scripts/make_labels.py`) or to rebuild the split.
Reproducing the reported numbers needs only what is already in the repository:

| file | content | size |
|---|---|---|
| `configs/splits/v1.json` | frozen split: part ids of train / val / test / test_ood + family id of every part | 0.7 MB |
| `results/features/descriptors_v1.npz` | 15-D shape descriptor of all 5,471 parts (`anytime_mesh.catalog.shape_descriptor`) | 0.4 MB |
| `results/labels/*.jsonl` | per-(part, method) a-priori mesh metrics (QQS, Instant Meshes, Blossom, QQS variants) | 5.2 MB |

## Sources

A part id is `<source>/<path relative to the source root>`. Set each root in `configs/data.toml`.

| source | parts in split | where to get it | licence |
|---|---|---|---|
| `abc` | 4,923 | ABC dataset, chunk 0000 STEP archive `abc_0000_step_v00.7z` (~1.5 GB), <https://deep-geometry.github.io/abc-dataset/>; root = the folder with the 8-digit sub-folders | Onshape terms of use / see dataset page |
| `set_b` | 449 | a second public CAD collection; not redistributed and not needed to reproduce any reported number | – |
| `simjeb` | 14 | SimJEB, Harvard Dataverse doi:10.7910/DVN/XFUWJG; root = folder of `.stp` CAD models | CC0 1.0 |
| `hexme` | 39 | HexMe, <https://github.com/cgg-bern/hex-me-if-you-can>; root = its `original/` folder | CC-BY 4.0 |
| `cad_models` | 36 | generate: `python scripts/make_parametric_parts.py --out <root>` (40 deterministic parametric parts) | MIT (this repo) |

Notes
- 5,471 parts imported as a single solid with 4-400 faces (`scripts/make_split.py`); 5,461 are in a split (the other
  10 are extra members of test families, discarded so no family is over-represented and none leaks into training).
- One training part (`cad_models/your_part.stp`) is not redistributed; it only affects regenerating train labels.
- The regenerated parametric parts reproduce the stored descriptors exactly (checked: max difference 0.0 over 40 parts).

## Split protocol (frozen before any learning experiment)

- Families = name/design families (same ABC document, same design id, same parametric family) united with
  near-duplicate clusters of the shape descriptor (`dup_tol = 0.15` in standardised descriptor space).
- A family never spans two splits; at most 3 parts per family in test and val.
- HexMe is test-only (different source, used as out-of-distribution set).
- Sizes: train 5,120 / val 150 / test 150 / test_ood 41. Results use the parts that have every label
  (test 142, OOD 39-40).
