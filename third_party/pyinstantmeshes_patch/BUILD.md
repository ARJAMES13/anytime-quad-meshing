# Patched pyinstantmeshes

Instant Meshes (Jakob et al. 2015) through the `pyinstantmeshes` Python binding (both BSD-3-Clause), with a
small patch. Upstream: <https://github.com/greenbrettmichael/pyinstantmeshes> with submodule
<https://github.com/wjakob/instant-meshes> at `external/instant-meshes`. Base commits are in `BASE_COMMITS`
(line 1: pyinstantmeshes, line 2: instant-meshes).

What the patch changes:
1. Adds `set_orientation_file`, `set_position_file`, `set_dump_file` (optional per-vertex constraints and a dump of
   the final orientation field). This repo does not use them, but resets them to `""` before every run.
2. Replaces TBB by a serial shim (`shim/`), because static TBB 4.x crashes under MinGW on Windows; the BVH build
   is made serial. Runs are deterministic (`deterministic=True`).

Instant Meshes numbers OBJ vertices by first appearance in the face list; `anytime_mesh.meshers` renumbers
vertices that way before writing. Use `remesh_file` (the in-memory `remesh` re-triangulates quads on read).

## Build

Needs a C++ compiler, CMake, Ninja, and `pybind11` + `scikit-build-core` in the active Python environment.
Run from the repository root:

```sh
P=$PWD/third_party/pyinstantmeshes_patch
git clone https://github.com/greenbrettmichael/pyinstantmeshes.git build/pyinstantmeshes
cd build/pyinstantmeshes
git checkout "$(sed -n 1p $P/BASE_COMMITS)"
git submodule update --init external/instant-meshes
git -C external/instant-meshes checkout "$(sed -n 2p $P/BASE_COMMITS)"
git -C external/instant-meshes submodule update --init --recursive
git apply $P/pyinstantmeshes.patch
git -C external/instant-meshes apply $P/instant-meshes.patch
cp -r $P/shim ./shim
pip install --no-build-isolation .
python -c "import pyinstantmeshes as p; print(p.set_orientation_file)"
```

Developed and tested on Windows 11 with the w64devkit GCC toolchain (`CC=gcc CXX=g++ CMAKE_GENERATOR=Ninja`;
add `-C cmake.define.pybind11_DIR=$(python -m pybind11 --cmakedir)` if CMake does not find pybind11).
