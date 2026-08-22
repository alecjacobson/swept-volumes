"""Swept Volumes via Spacetime Numerical Continuation.

Python bindings (nanobind) for the C++ implementation.

The trajectory ``transform(t)`` is evaluated a very large number of times inside
the C++ continuation. To keep that hot loop free of any Python/GIL/NumPy
overhead, write the transform with the native ABI and compile it with
``numba.cfunc`` using the :func:`native_transform` decorator::

    import math
    from numba import farray
    from swept_volumes import native_transform, NativeTransform, swept_volume

    @native_transform
    def spin(t, A, Adot):
        # A, Adot are Fortran-order (column-major) 4x4 views (Eigen storage).
        c, s = math.cos(t), math.sin(t)
        A[0, 0] = c;  A[0, 1] = -s; A[0, 3] = t
        A[1, 0] = s;  A[1, 1] = c
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        Adot[0, 0] = -s; Adot[0, 1] = -c; Adot[0, 3] = 1.0
        Adot[1, 0] = c;  Adot[1, 1] = -s

    U, G, _, _ = swept_volume(V, F, NativeTransform(spin), eps=0.02)

The address of the compiled ``cfunc`` is extracted once at the C++ boundary;
every inner evaluation is then an ordinary indirect native call.
"""

from ._swept_volumes import (  # noqa: F401
    ContouringMethod,
    NativeTransform,
    swept_volume,
    swept_volume_keyframes,
    swept_volume_pyfunc,
    bench_native,
    bench_cpp_baseline,
    bench_pyfunc,
    cpp_baseline_address,
)

__all__ = [
    "ContouringMethod",
    "NativeTransform",
    "swept_volume",
    "swept_volume_keyframes",
    "swept_volume_pyfunc",
    "native_transform",
    "TRANSFORM_SIG",
    "bench_native",
    "bench_cpp_baseline",
    "bench_pyfunc",
    "cpp_baseline_address",
]


def _transform_sig():
    """The numba signature for a native transform: void(double, double*, double*)."""
    from numba import types

    return types.void(
        types.float64,
        types.CPointer(types.float64),
        types.CPointer(types.float64),
    )


try:  # Only available if numba is installed.
    TRANSFORM_SIG = _transform_sig()
except Exception:  # pragma: no cover - numba optional at import time
    TRANSFORM_SIG = None


def native_transform(fn=None, *, cache=True):
    """Compile ``fn(t, A, Adot)`` into a native transform via ``numba.cfunc``.

    ``A`` and ``Adot`` are exposed inside ``fn`` as Fortran-order (column-major)
    4x4 views over the destination Eigen matrices, obtained with
    ``numba.farray(ptr, (4, 4))``. This decorator wires that up for you and
    returns an object usable directly as ``NativeTransform(result)``.

    Usage::

        @native_transform
        def xf(t, A, Adot):
            ...

    The returned value is the numba ``CFunc`` (it exposes ``.address``); pass it
    to :class:`NativeTransform`.
    """
    from numba import cfunc, farray, njit

    def _decorate(f):
        # Compile the user's (t, A, Adot) body so the cfunc trampoline can call
        # it in nopython mode, then expose it via the pointer ABI.
        inner = f if hasattr(f, "inspect_llvm") else njit(cache=cache)(f)

        def _wrapped(t, a_ptr, adot_ptr):
            A = farray(a_ptr, (4, 4))
            Adot = farray(adot_ptr, (4, 4))
            # Zero the destination first (the C++ side passes uninitialized
            # storage) so the user only needs to set the non-zero entries.
            for _i in range(4):
                for _j in range(4):
                    A[_i, _j] = 0.0
                    Adot[_i, _j] = 0.0
            inner(t, A, Adot)

        _wrapped.__name__ = getattr(f, "__name__", "native_transform")
        return cfunc(_transform_sig(), nopython=True, cache=cache)(_wrapped)

    if fn is not None:
        return _decorate(fn)
    return _decorate
