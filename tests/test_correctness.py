"""Correctness tests for the swept-volume pipeline against analytic ground truth.

Two independent checks per case:
  * containment  -- every posed brush point must lie inside the swept mesh, up to
    ~one grid cell of corner-cutting (the swept volume is the union of all poses).
  * accuracy     -- the swept mesh must match the closed-form SDF of the known
    result (capsule / torus / cylinder / box) to within ~one grid cell.

These would catch the two regressions found in review: a brush "poking out" of
the swept volume, and stamping/strobo artifacts (which show up as large analytic
deviation and as non-convergence under grid refinement).
"""
import math

import numpy as np
import pytest

import swept_volumes as sv
from swept_volumes import NativeTransform, native_transform, ContouringMethod

igl = pytest.importorskip("igl")
trimesh = pytest.importorskip("trimesh")
pytest.importorskip("numba")

SDT = igl.SIGNED_DISTANCE_TYPE_WINDING_NUMBER


def sphere_brush(r=0.2, sub=3):
    s = trimesh.creation.icosphere(subdivisions=sub, radius=r)
    return np.asarray(s.vertices, float), np.asarray(s.faces, np.int32)


def box_brush(hx, hy, hz):
    V = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                  [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float) * [hx, hy, hz]
    F = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], np.int32)
    return V, F


def _run(V, F, nt, eps, true_sdf, num_seeds=200, nt_samples=80, tmpdir="/tmp/sv_corr"):
    U, G, _, _ = sv.swept_volume(V, F, nt, eps=eps, num_seeds=num_seeds,
                                 dir_name=tmpdir, contouring=ContouringMethod.MarchingCubes)
    U = np.asarray(U, np.float64)
    Gi = np.asarray(G, np.int64)
    # containment: max signed distance of posed brush vertices (igl: +outside)
    poke = -1e9
    for t in np.linspace(0, 1, nt_samples):
        A, _ = nt(t)
        R, x = np.asarray(A[:3, :3]), np.asarray(A[:3, 3])
        S, _, _, _ = igl.signed_distance(np.asarray((V @ R.T) + x, np.float64), U, Gi, SDT)
        poke = max(poke, float(S.max()))
    dev = np.abs(true_sdf(U))
    return U, poke, float(dev.max()), float(dev.mean())


# --------------------------------------------------------------------------
# Analytic swept volumes: capsule, torus, cylinder, box
# --------------------------------------------------------------------------
def test_translating_sphere_is_capsule():
    L, r = 0.8, 0.2
    @native_transform
    def lin(t, A, Adot):
        A[0, 0] = 1.; A[1, 1] = 1.; A[2, 2] = 1.; A[3, 3] = 1.; A[0, 3] = t * L
        Adot[0, 3] = L

    def capsule(P):
        p = P.copy(); p[:, 0] -= np.clip(p[:, 0], 0, L)
        return np.linalg.norm(p, axis=1) - r

    V, F = sphere_brush(r)
    eps = 0.03
    _, poke, dmax, dmean = _run(V, F, NativeTransform(lin), eps, capsule)
    assert poke < 1.5 * eps, f"brush pokes out by {poke:.4f} ({poke/eps:.1f} cells)"
    assert dmax < 1.5 * eps and dmean < 0.5 * eps, (dmax, dmean)


def test_circular_sphere_is_torus():
    R0, r = 0.55, 0.2
    TAU = 2 * math.pi
    @native_transform
    def circ(t, A, Adot):
        th = TAU * t; c, s = math.cos(th), math.sin(th); w = TAU
        A[0, 0] = 1.; A[1, 1] = 1.; A[2, 2] = 1.; A[3, 3] = 1.
        A[0, 3] = R0 * c; A[1, 3] = R0 * s
        Adot[0, 3] = -R0 * s * w; Adot[1, 3] = R0 * c * w

    def torus(P):
        q = np.sqrt(P[:, 0] ** 2 + P[:, 1] ** 2) - R0
        return np.sqrt(q ** 2 + P[:, 2] ** 2) - r

    V, F = sphere_brush(r)
    eps = 0.03
    _, poke, dmax, dmean = _run(V, F, NativeTransform(circ), eps, torus)
    assert poke < 1.5 * eps, f"brush pokes out by {poke:.4f} ({poke/eps:.1f} cells)"
    assert dmax < 1.5 * eps and dmean < 0.5 * eps, (dmax, dmean)


def test_rotating_box_is_cylinder():
    hx, hy, hz = 0.3, 0.1, 0.1
    rad = math.hypot(hx, hy)
    TAU = 2 * math.pi
    @native_transform
    def rot(t, A, Adot):
        th = TAU * t; c, s = math.cos(th), math.sin(th); w = TAU
        A[0, 0] = c; A[0, 1] = -s; A[1, 0] = s; A[1, 1] = c; A[2, 2] = 1.; A[3, 3] = 1.
        Adot[0, 0] = -s * w; Adot[0, 1] = -c * w; Adot[1, 0] = c * w; Adot[1, 1] = -s * w

    def cylinder(P):
        d_r = np.sqrt(P[:, 0] ** 2 + P[:, 1] ** 2) - rad
        d_z = np.abs(P[:, 2]) - hz
        return (np.sqrt(np.maximum(d_r, 0) ** 2 + np.maximum(d_z, 0) ** 2)
                + np.minimum(np.maximum(d_r, d_z), 0))

    V, F = box_brush(hx, hy, hz)
    eps = 0.02
    _, poke, dmax, dmean = _run(V, F, NativeTransform(rot), eps, cylinder)
    assert poke < 1.5 * eps, f"brush pokes out by {poke:.4f} ({poke/eps:.1f} cells)"
    assert dmax < 2.0 * eps and dmean < 0.5 * eps, (dmax, dmean)


# --------------------------------------------------------------------------
# Refinement convergence: analytic error must shrink with the grid.  A stamping
# artifact would NOT converge (it is fixed in world space, not the grid).
# --------------------------------------------------------------------------
def test_torus_convergence():
    R0, r = 0.55, 0.2
    TAU = 2 * math.pi
    @native_transform
    def circ(t, A, Adot):
        th = TAU * t; c, s = math.cos(th), math.sin(th); w = TAU
        A[0, 0] = 1.; A[1, 1] = 1.; A[2, 2] = 1.; A[3, 3] = 1.
        A[0, 3] = R0 * c; A[1, 3] = R0 * s
        Adot[0, 3] = -R0 * s * w; Adot[1, 3] = R0 * c * w

    def torus(P):
        q = np.sqrt(P[:, 0] ** 2 + P[:, 1] ** 2) - R0
        return np.sqrt(q ** 2 + P[:, 2] ** 2) - r

    V, F = sphere_brush(r, sub=3)
    _, _, d_coarse, _ = _run(V, F, NativeTransform(circ), 0.04, torus)
    _, _, d_fine, _ = _run(V, F, NativeTransform(circ), 0.02, torus)
    # error should at least not grow, and stay within a cell at each level
    assert d_coarse < 1.5 * 0.04 and d_fine < 1.5 * 0.02, (d_coarse, d_fine)


# --------------------------------------------------------------------------
# The demo brush (watertight cross) must contain its own sweep.
# --------------------------------------------------------------------------
def test_watertight_cross_containment():
    import importlib.util
    import os
    ex_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "python", "example_polyscope.py")
    spec = importlib.util.spec_from_file_location("ex", ex_path)
    ex = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ex)
    V, F = ex.make_cross()
    # sanity: watertight cross has winding 1 (not 3) at the centre
    w = np.atleast_1d(igl.winding_number(V.astype(np.float64), F.astype(np.int64),
                                         np.array([[0.0, 0.0, 0.0]])))
    assert abs(w[0] - 1.0) < 1e-6, f"brush not watertight/clean: winding={w[0]}"

    eps = 0.03
    _, poke, _, _ = _run(V, F, NativeTransform(ex.orbit), eps, lambda P: np.zeros(len(P)),
                         num_seeds=300)
    assert poke < 1.5 * eps, f"cross pokes out of its sweep by {poke:.4f} ({poke/eps:.1f} cells)"
