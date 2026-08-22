import math

import numpy as np
import pytest

import swept_volumes as sv
from swept_volumes import NativeTransform, native_transform, ContouringMethod

numba = pytest.importorskip("numba")


# --------------------------------------------------------------------------
# Shared analytic trajectory: rotation about z by t + translation t along x.
# Expressed as a native (numba) transform, a pure-Python callable, and closed
# form -- all three must agree.
# --------------------------------------------------------------------------
@native_transform
def spin(t, A, Adot):
    c, s = math.cos(t), math.sin(t)
    A[0, 0] = c;  A[0, 1] = -s; A[0, 3] = t
    A[1, 0] = s;  A[1, 1] = c
    A[2, 2] = 1.0
    A[3, 3] = 1.0
    Adot[0, 0] = -s; Adot[0, 1] = -c; Adot[0, 3] = 1.0
    Adot[1, 0] = c;  Adot[1, 1] = -s


def spin_py(t):
    c, s = math.cos(t), math.sin(t)
    A = np.array([[c, -s, 0, t], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], float)
    Adot = np.array(
        [[-s, -c, 0, 1], [c, -s, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], float
    )
    return A, Adot


def unit_cube():
    V = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float
    )
    V = (V - 0.5) * 0.3
    F = np.array(
        [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
         [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
         [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], np.int32
    )
    return V, F


# --------------------------------------------------------------------------
# ABI correctness
# --------------------------------------------------------------------------
@pytest.mark.parametrize("t", [0.0, 0.37, 1.0])
def test_native_transform_matches_closed_form(t):
    nt = NativeTransform(spin)
    A, Adot = nt(t)
    expA, expAdot = spin_py(t)
    assert np.allclose(A, expA), (A, expA)
    assert np.allclose(Adot, expAdot), (Adot, expAdot)


def test_native_transform_from_raw_address():
    # The C++ baseline transform, addressed by integer, is usable directly.
    nt = NativeTransform(sv.cpp_baseline_address)
    A, _ = nt(0.5)
    c, s = math.cos(0.5), math.sin(0.5)
    assert np.allclose(A, [[c, -s, 0, 0.5], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


# --------------------------------------------------------------------------
# Output equivalence: the transform delivery mechanism must not change results.
# --------------------------------------------------------------------------
def test_native_matches_pyfunc(tmp_path):
    V, F = unit_cube()
    kw = dict(eps=0.08, num_seeds=40)
    Un, Gn, _, _ = sv.swept_volume(
        V, F, NativeTransform(spin), dir_name=str(tmp_path / "nat"), **kw
    )
    Up, Gp, _, _ = sv.swept_volume_pyfunc(
        V, F, spin_py, dir_name=str(tmp_path / "py"), **kw
    )
    assert Un.shape == Up.shape
    assert Gn.shape == Gp.shape
    # Same continuation, same contouring -> identical mesh (up to fp noise).
    assert np.allclose(np.sort(Un, axis=0), np.sort(Up, axis=0), atol=1e-9)
    assert Gn.shape[0] > 0


# --------------------------------------------------------------------------
# Contouring choice: both back-ends produce a plausible mesh on the same cells.
# --------------------------------------------------------------------------
def test_marching_cubes_and_dual_contouring(tmp_path):
    V, F = unit_cube()
    kw = dict(eps=0.08, num_seeds=40)
    Umc, Gmc, _, _ = sv.swept_volume(
        V, F, NativeTransform(spin), dir_name=str(tmp_path / "mc"),
        contouring=ContouringMethod.MarchingCubes, **kw
    )
    Udc, Gdc, _, _ = sv.swept_volume(
        V, F, NativeTransform(spin), dir_name=str(tmp_path / "dc"),
        contouring=ContouringMethod.DualContouring, **kw
    )
    assert Gmc.shape[0] > 0 and Gdc.shape[0] > 0
    # Both surfaces should occupy essentially the same bounding box.
    bb_mc = Umc.max(0) - Umc.min(0)
    bb_dc = Udc.max(0) - Udc.min(0)
    assert np.allclose(bb_mc, bb_dc, atol=3 * 0.08)
    # Every dual-contouring vertex is near the marching-cubes surface.
    from scipy.spatial import cKDTree  # noqa
    d, _ = cKDTree(Umc).query(Udc)
    assert d.max() < 3 * 0.08


# --------------------------------------------------------------------------
# Keyframe API still works.
# --------------------------------------------------------------------------
def test_keyframe_api(tmp_path):
    V, F = unit_cube()
    K0 = np.eye(4)
    K1 = np.eye(4); K1[0, 3] = 0.5           # translate along x
    K2 = np.eye(4); K2[0, 3] = 0.5; K2[1, 3] = 0.5
    U, G, _, _ = sv.swept_volume_keyframes(
        V, F, [K0, K1, K2], eps=0.08, num_seeds=40, dir_name=str(tmp_path / "kf")
    )
    assert U.shape[0] > 0 and G.shape[0] > 0


# --------------------------------------------------------------------------
# Runtime parity: numba cfunc is ~C++ speed; a Python callable is far slower.
# --------------------------------------------------------------------------
def test_runtime_parity():
    n = 5_000_000
    nt = NativeTransform(spin)
    t_native = sv.bench_native(nt, n)
    t_cpp = sv.bench_cpp_baseline(n)
    # The numba cfunc is a single indirect native call, like the C++ baseline.
    assert t_native < 4.0 * t_cpp, (t_native, t_cpp)


def test_pyfunc_is_much_slower():
    n = 100_000
    t_py = sv.bench_pyfunc(spin_py, n) / n
    t_native = sv.bench_native(NativeTransform(spin), n) / n
    assert t_py > 10.0 * t_native, (t_py, t_native)


def _read_obj(path):
    Vs, Fs = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                Vs.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                Fs.append([int(x.split("/")[0]) - 1 for x in line.split()[1:4]])
    return np.array(Vs, float), np.array(Fs, np.int32)


@pytest.mark.slow
def test_bunny_end_to_end(tmp_path):
    import os

    bunny = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "bunny.obj"
    )
    if not os.path.exists(bunny):
        pytest.skip("data/bunny.obj not present")
    V, F = _read_obj(bunny)
    U, G, _, _ = sv.swept_volume(
        V, F, NativeTransform(spin), eps=0.06, num_seeds=100,
        dir_name=str(tmp_path / "bunny")
    )
    assert U.shape[0] > 1000 and G.shape[0] > 1000
    # native transform and pure-Python callable agree on the real mesh too.
    Up, Gp, _, _ = sv.swept_volume_pyfunc(
        V, F, spin_py, eps=0.06, num_seeds=100, dir_name=str(tmp_path / "bunny_py")
    )
    assert U.shape == Up.shape and G.shape == Gp.shape
