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


def _box(hx, hy, hz):
    V = np.array(
        [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
         [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float
    ) * [hx, hy, hz]
    F = np.array(
        [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
         [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
         [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], np.int32
    )
    return V, F


def make_cross():
    """A 3D cross: three orthogonal bars (the algorithm needs no manifoldness)."""
    L, W = 0.30, 0.085
    parts = [_box(L, W, W), _box(W, L, W), _box(W, W, L)]
    Vs, Fs, off = [], [], 0
    for V, F in parts:
        Vs.append(V)
        Fs.append(F + off)
        off += V.shape[0]
    return np.vstack(Vs), np.vstack(Fs).astype(np.int32)


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
    swept.set_transparency(0.35)
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
