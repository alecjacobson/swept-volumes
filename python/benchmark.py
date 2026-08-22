#!/usr/bin/env python3
"""Per-call overhead of the transform delivery mechanisms.

Compares, for the same trajectory:
  * a numba.cfunc native transform (recommended)
  * the C++ baseline transform (indirect call reference)
  * a plain Python callable (slow path)

Run:  python python/benchmark.py
"""
import math
import sys

import numpy as np

import swept_volumes as sv
from swept_volumes import NativeTransform, native_transform


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
    Adot = np.array([[-s, -c, 0, 1], [c, -s, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], float)
    return A, Adot


def main():
    nt = NativeTransform(spin)
    nt(0.0)  # warm up numba compilation

    n = 20_000_000
    t_native = sv.bench_native(nt, n)
    t_cpp = sv.bench_cpp_baseline(n)
    n_py = 200_000
    t_py = sv.bench_pyfunc(spin_py, n_py)

    print(f"{'mechanism':<24}{'ns/call':>12}{'vs C++':>12}")
    print("-" * 48)
    print(f"{'C++ baseline':<24}{t_cpp / n * 1e9:>12.2f}{1.0:>12.2f}")
    print(f"{'numba cfunc':<24}{t_native / n * 1e9:>12.2f}{t_native / t_cpp:>12.2f}")
    print(f"{'python callable':<24}{t_py / n_py * 1e9:>12.2f}{(t_py / n_py) / (t_cpp / n):>12.2f}")


if __name__ == "__main__":
    sys.exit(main())
