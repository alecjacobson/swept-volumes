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

import swept_volumes as sv
from swept_volumes import NativeTransform, native_transform

TWO_PI = 2.0 * math.pi
ORBIT_R = 0.55   # orbit radius
BOB_Z = 0.18     # vertical bob amplitude


# --- the cyclic trajectory: orbit on a circle + spin + vertical bob ----------
@native_transform
def orbit(t, A, Adot):
    th = TWO_PI * t
    c, s = math.cos(th), math.sin(th)
    c2, s2 = math.cos(2 * th), math.sin(2 * th)
    w = TWO_PI  # d(th)/dt
    # pose: rotation Rz(th), translation on a circle + a vertical bob
    A[0, 0] = c;  A[0, 1] = -s
    A[1, 0] = s;  A[1, 1] = c
    A[2, 2] = 1.0
    A[3, 3] = 1.0
    A[0, 3] = ORBIT_R * c
    A[1, 3] = ORBIT_R * s
    A[2, 3] = BOB_Z * s2
    # time-derivative
    Adot[0, 0] = -s * w; Adot[0, 1] = -c * w
    Adot[1, 0] = c * w;  Adot[1, 1] = -s * w
    Adot[0, 3] = -ORBIT_R * s * w
    Adot[1, 3] = ORBIT_R * c * w
    Adot[2, 3] = BOB_Z * c2 * 2.0 * w


def make_bar(hx=0.28, hy=0.08, hz=0.08):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gif", default="assets/swept_volume_demo.gif")
    ap.add_argument("--eps", type=float, default=0.04)
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--width", type=int, default=720)
    args = ap.parse_args()

    V, F = make_bar()
    nt = NativeTransform(orbit)

    # --- compute the swept volume ---
    U, G, _, _ = sv.swept_volume(
        V, F, nt, eps=args.eps, num_seeds=200, dir_name="/tmp/swept_demo"
    )
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
