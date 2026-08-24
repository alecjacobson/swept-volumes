#!/usr/bin/env python3
"""Tiny end-to-end example: compute a swept volume and render an animation.

Sweeps a bar along a cyclic orbit, computes the swept volume, and renders it
with polyscope -- semi-transparent swept volume, soft ground shadows, and the
input shape animated cyclically inside. Saves an animated GIF.

    python python/example_polyscope.py --gif assets/swept_volume_demo.gif

Works headless (polyscope's EGL backend); no display required.
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from numba import njit

import swept_volumes as sv
from swept_volumes import NativeTransform, native_transform

TWO_PI = 2.0 * math.pi
ORBIT_R = 0.55   # orbit radius
BOB_Z = 0.18     # vertical bob amplitude
TWIST = 2.0      # full twists of the shape per orbit (integer -> cyclic)


@njit(cache=True)
def _mm3(a, b):
    c = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            s = 0.0
            for k in range(3):
                s += a[i, k] * b[k, j]
            c[i, j] = s
    return c


# --- the cyclic trajectory: orbit on a circle, spin, twist, and bob ----------
@native_transform
def orbit(t, A, Adot):
    th = TWO_PI * t                 # orbital angle
    ph = TWIST * TWO_PI * t         # twist angle about the shape's radial axis
    wth = TWO_PI                    # d(th)/dt
    wph = TWIST * TWO_PI            # d(ph)/dt
    cz, sz = math.cos(th), math.sin(th)
    cx, sx = math.cos(ph), math.sin(ph)

    # Rz(th): orientation follows the orbit
    Rz = np.zeros((3, 3))
    Rz[0, 0] = cz; Rz[0, 1] = -sz; Rz[1, 0] = sz; Rz[1, 1] = cz; Rz[2, 2] = 1.0
    dRz = np.zeros((3, 3))          # d Rz / d th
    dRz[0, 0] = -sz; dRz[0, 1] = -cz; dRz[1, 0] = cz; dRz[1, 1] = -sz

    # Rx(ph): twist about the local x (radial) axis
    Rx = np.zeros((3, 3))
    Rx[0, 0] = 1.0; Rx[1, 1] = cx; Rx[1, 2] = -sx; Rx[2, 1] = sx; Rx[2, 2] = cx
    dRx = np.zeros((3, 3))          # d Rx / d ph
    dRx[1, 1] = -sx; dRx[1, 2] = -cx; dRx[2, 1] = cx; dRx[2, 2] = -sx

    R = _mm3(Rz, Rx)
    dR = _mm3(dRz, Rx)
    dR2 = _mm3(Rz, dRx)

    c2, s2 = math.cos(2 * th), math.sin(2 * th)
    for i in range(3):
        for j in range(3):
            A[i, j] = R[i, j]
            Adot[i, j] = wth * dR[i, j] + wph * dR2[i, j]
    # translation: circular orbit + vertical bob
    A[0, 3] = ORBIT_R * cz
    A[1, 3] = ORBIT_R * sz
    A[2, 3] = BOB_Z * s2
    A[3, 3] = 1.0
    Adot[0, 3] = -ORBIT_R * sz * wth
    Adot[1, 3] = ORBIT_R * cz * wth
    Adot[2, 3] = BOB_Z * c2 * 2.0 * wth


def make_cross():
    """A 3D cross (three orthogonal bars, half-length 0.30, half-width 0.085) as a
    single **watertight** mesh — the boolean union of the bars, not overlapping
    boxes. Overlapping boxes are a non-manifold soup whose interior winding number
    exceeds 1 and whose interior faces corrupt the distance field; that makes the
    swept volume wrong (the brush pokes out, and the surface shows stamping-like
    artifacts). A clean watertight brush avoids all of that.
    """
    V = np.array([
        [-0.3, -0.085, -0.085], [-0.085, -0.3, -0.085], [-0.085, -0.085, -0.3],
        [-0.085, -0.085, -0.085], [-0.3, -0.085, 0.085], [-0.085, -0.3, 0.085],
        [-0.085, -0.085, 0.085], [-0.085, -0.085, 0.3], [-0.3, 0.085, -0.085],
        [-0.085, 0.085, -0.3], [-0.085, 0.085, -0.085], [-0.085, 0.3, -0.085],
        [-0.3, 0.085, 0.085], [-0.085, 0.085, 0.085], [-0.085, 0.085, 0.3],
        [-0.085, 0.3, 0.085], [0.085, -0.3, -0.085], [0.085, -0.085, -0.3],
        [0.085, -0.085, -0.085], [0.3, -0.085, -0.085], [0.085, -0.3, 0.085],
        [0.085, -0.085, 0.085], [0.085, -0.085, 0.3], [0.3, -0.085, 0.085],
        [0.085, 0.085, -0.3], [0.085, 0.085, -0.085], [0.085, 0.3, -0.085],
        [0.3, 0.085, -0.085], [0.085, 0.085, 0.085], [0.085, 0.085, 0.3],
        [0.085, 0.3, 0.085], [0.3, 0.085, 0.085],
    ], float)
    F = np.array([
        [0, 3, 4], [0, 8, 3], [0, 4, 12], [4, 6, 12], [4, 3, 6], [0, 12, 8],
        [8, 10, 3], [8, 13, 10], [8, 12, 13], [12, 6, 13], [19, 21, 18],
        [19, 18, 25], [19, 27, 23], [19, 23, 21], [23, 31, 21], [19, 25, 27],
        [27, 25, 31], [31, 28, 21], [31, 25, 28], [23, 27, 31], [1, 5, 3],
        [1, 16, 5], [1, 3, 18], [5, 6, 3], [5, 21, 6], [11, 10, 13], [11, 26, 10],
        [11, 13, 15], [11, 15, 30], [15, 13, 30], [1, 18, 16], [16, 18, 21],
        [5, 16, 20], [5, 20, 21], [16, 21, 20], [26, 25, 10], [11, 30, 26],
        [26, 30, 25], [30, 13, 28], [30, 28, 25], [2, 3, 9], [2, 18, 3],
        [2, 9, 17], [7, 13, 6], [7, 6, 22], [9, 3, 10], [9, 10, 24], [7, 14, 13],
        [7, 29, 14], [14, 28, 13], [2, 17, 18], [17, 25, 18], [22, 6, 21],
        [7, 22, 29], [22, 21, 29], [9, 24, 17], [24, 10, 25], [17, 24, 25],
        [14, 29, 28], [29, 21, 28],
    ], np.int32)
    return V, F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gif", default="assets/swept_volume_demo.gif")
    ap.add_argument("--eps", type=float, default=0.015)
    ap.add_argument("--num-seeds", type=int, default=400)
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--contouring", choices=["mc", "dc"], default="mc",
                    help="marching cubes (default) or dual contouring")
    args = ap.parse_args()

    from swept_volumes import ContouringMethod
    contouring = (ContouringMethod.DualContouring if args.contouring == "dc"
                  else ContouringMethod.MarchingCubes)

    V, F = make_cross()
    nt = NativeTransform(orbit)

    # --- compute the swept volume ---
    import time
    t0 = time.time()
    U, G, _, _ = sv.swept_volume(
        V, F, nt, eps=args.eps, num_seeds=args.num_seeds,
        dir_name="/tmp/swept_demo", contouring=contouring,
    )
    print(f"swept volume computed in {time.time() - t0:.1f}s")
    print(f"swept volume: {U.shape[0]} verts, {G.shape[0]} faces")

    # --- render with polyscope ---
    import imageio.v2 as imageio
    import polyscope as ps
    from PIL import Image

    ps.set_allow_headless_backends(True)
    ps.set_program_name("swept-volumes demo")
    ps.init()
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height_factor(0.02, is_relative=True)
    ps.set_shadow_darkness(0.55)
    ps.set_shadow_blur_iters(4)
    ps.set_transparency_mode("pretty")
    ps.set_SSAA_factor(3)                      # supersample for crisp edges
    ps.set_background_color((1.0, 1.0, 1.0))
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")

    swept = ps.register_surface_mesh("swept volume", U, G, smooth_shade=True)
    swept.set_color((0.30, 0.55, 0.95))
    swept.set_transparency(0.55)
    swept.set_material("wax")

    shape = ps.register_surface_mesh("shape", V.copy(), F, smooth_shade=False)
    shape.set_color((0.95, 0.55, 0.15))

    ps.look_at((1.6, -1.9, 1.35), (0.0, 0.0, 0.0))

    frames = []
    for i in range(args.frames):
        t = i / args.frames
        A, _ = nt(t)
        R, x = A[:3, :3], A[:3, 3]
        shape.update_vertex_positions((V @ R.T) + x)
        buf = ps.screenshot_to_buffer(transparent_bg=False)  # HxWx4 uint8
        img = Image.fromarray(buf[:, :, :3])
        if img.width > args.width:
            h = round(img.height * args.width / img.width)
            img = img.resize((args.width, h), Image.LANCZOS)
        frames.append(np.asarray(img))

    os.makedirs(os.path.dirname(os.path.abspath(args.gif)), exist_ok=True)
    imageio.mimsave(args.gif, frames, duration=1 / 24, loop=0)
    print(f"wrote {args.gif} ({len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]})")


if __name__ == "__main__":
    main()
