"""Generate the 40 parametric mechanical parts of the 'cad_models' source (deterministic, Gmsh OpenCASCADE).

Five families x 8 sizes: L-brackets, bolted flanges, pocketed plates, ribbed U-channels, bearing blocks.
Example: python scripts/make_parametric_parts.py --out /path/to/cad_models
"""
import argparse
import math
import os


def l_bracket(occ, length, width, thickness, hole_radius):
    b1 = occ.addBox(0, 0, 0, length, width, thickness)
    b2 = occ.addBox(0, 0, thickness, thickness, width, length)
    fused, _ = occ.fuse([(3, b1)], [(3, b2)])
    h1 = occ.addCylinder(length * 0.65, width * 0.5, -thickness * 0.5, 0, 0, thickness * 2.0, hole_radius)
    cut1, _ = occ.cut(fused, [(3, h1)])
    h2 = occ.addCylinder(-thickness * 0.5, width * 0.5, length * 0.65, thickness * 2.0, 0, 0, hole_radius)
    occ.cut(cut1, [(3, h2)])


def flange(occ, outer_radius, inner_radius, height, num_holes, hole_radius):
    c_main = occ.addCylinder(0, 0, 0, 0, 0, height, outer_radius)
    c_inner = occ.addCylinder(0, 0, -height * 0.5, 0, 0, height * 2.0, inner_radius)
    solid, _ = occ.cut([(3, c_main)], [(3, c_inner)])
    bolt_r = (outer_radius + inner_radius) / 2.0
    for i in range(num_holes):
        a = 2.0 * math.pi * i / num_holes
        h = occ.addCylinder(bolt_r * math.cos(a), bolt_r * math.sin(a), -height * 0.5, 0, 0, height * 2.0, hole_radius)
        solid, _ = occ.cut(solid, [(3, h)])


def plate_with_pockets(occ, length, width, height, pocket_depth, num_holes):
    b = occ.addBox(0, 0, 0, length, width, height)
    pocket = occ.addBox(length * 0.25, width * 0.25, height - pocket_depth, length * 0.5, width * 0.5, pocket_depth * 1.5)
    solid, _ = occ.cut([(3, b)], [(3, pocket)])
    corners = [(length * 0.12, width * 0.12), (length * 0.88, width * 0.12),
               (length * 0.12, width * 0.88), (length * 0.88, width * 0.88)]
    hr = min(length, width) * 0.04
    for cx, cy in corners[:num_holes]:
        ch = occ.addCylinder(cx, cy, -height * 0.5, 0, 0, height * 2.0, hr)
        solid, _ = occ.cut(solid, [(3, ch)])


def ribbed_channel(occ, length, width, height, wall, num_ribs):
    b_outer = occ.addBox(0, 0, 0, length, width, height)
    b_inner = occ.addBox(-length * 0.1, wall, wall, length * 1.2, width - 2 * wall, height * 1.5)
    channel, _ = occ.cut([(3, b_outer)], [(3, b_inner)])
    if num_ribs > 0:
        step_x = length / (num_ribs + 1)
        ribs = [(3, occ.addBox(i * step_x - wall / 2.0, wall, wall, wall, width - 2 * wall, height * 0.7))
                for i in range(1, num_ribs + 1)]
        occ.fuse(channel, ribs)


def bearing_block(occ, base_l, base_w, base_h, bore_r):
    b_base = occ.addBox(0, 0, 0, base_l, base_w, base_h)
    arch_w, arch_h = base_w * 0.7, base_h * 2.5
    b_arch = occ.addBox(base_l * 0.25, (base_w - arch_w) / 2.0, base_h, base_l * 0.5, arch_w, arch_h)
    fused, _ = occ.fuse([(3, b_base)], [(3, b_arch)])
    c_bore = occ.addCylinder(-base_l * 0.1, base_w / 2.0, base_h + arch_h * 0.5, base_l * 1.2, 0, 0, bore_r)
    solid, _ = occ.cut(fused, [(3, c_bore)])
    mh_r = min(base_l, base_w) * 0.05
    mh1 = occ.addCylinder(base_l * 0.12, base_w * 0.5, -base_h * 0.5, 0, 0, base_h * 2.0, mh_r)
    mh2 = occ.addCylinder(base_l * 0.88, base_w * 0.5, -base_h * 0.5, 0, 0, base_h * 2.0, mh_r)
    occ.cut(solid, [(3, mh1), (3, mh2)])


def parts():
    for i in range(8):
        yield f"bracket_{i:02d}", l_bracket, (40.0 + i * 5.0, 25.0 + i * 3.0, 6.0 + (i % 3) * 1.5, 3.5 + (i % 3) * 0.5)
    for i in range(8):
        yield f"flange_{i:02d}", flange, (30.0 + i * 4.0, 10.0 + i * 1.5, 10.0 + (i % 4) * 2.0, 4 + (i % 4) * 2,
                                          3.0 + (i % 2) * 0.5)
    for i in range(8):
        yield f"plate_pocket_{i:02d}", plate_with_pockets, (60.0 + i * 5.0, 45.0 + i * 4.0, 12.0 + (i % 3) * 2.0,
                                                            5.0 + (i % 3) * 1.5, 2 + (i % 3))
    for i in range(8):
        yield f"ribbed_channel_{i:02d}", ribbed_channel, (70.0 + i * 6.0, 35.0 + i * 3.0, 20.0 + (i % 3) * 3.0,
                                                          3.0 + (i % 2) * 1.0, 1 + (i % 4))
    for i in range(8):
        yield f"bearing_block_{i:02d}", bearing_block, (50.0 + i * 4.0, 30.0 + i * 3.0, 12.0 + (i % 3) * 1.5,
                                                        8.0 + (i % 4) * 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output directory (becomes the 'cad_models' root)")
    a = ap.parse_args()
    import gmsh
    os.makedirs(a.out, exist_ok=True)
    gmsh.initialize(readConfigFiles=False)
    try:
        gmsh.option.setNumber("General.Verbosity", 0)
        for name, fn, args in parts():
            gmsh.model.add(name)
            fn(gmsh.model.occ, *args)
            gmsh.model.occ.synchronize()
            gmsh.write(os.path.join(a.out, name + ".stp"))
            gmsh.model.remove()
    finally:
        gmsh.finalize()
    print("wrote 40 parts to", a.out)


if __name__ == "__main__":
    main()
