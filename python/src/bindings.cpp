// nanobind bindings for swept-volumes.
//
// The performance-critical piece is the trajectory transform(t), which the
// spacetime continuation evaluates a very large number of times.  We expose it
// via NativeTransform: the user writes the transform in Python, compiles it to
// native machine code with numba.cfunc using the ABI
//
//     void transform(double t, double* A, double* Adot)
//
// where A and Adot each point at 16 doubles (column-major 4x4, matching Eigen
// storage / numba farray Fortran order).  We extract the function pointer once
// (via cfunc.address) and the hot loop is then an ordinary indirect native
// call -- no CPython, no GIL, no NumPy object per evaluation.
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/pair.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <vector>

#include "swept_volume.h"

namespace nb = nanobind;
using namespace nb::literals;

// ---------------------------------------------------------------------------
// NativeTransform: a native function pointer produced by numba.cfunc.
// ---------------------------------------------------------------------------
class NativeTransform {
public:
    using Func = void (*)(double, double *, double *);

    // Construct from a numba cfunc object (uses its .address) or a raw integer
    // address.  We keep `owner_` alive because the numba CFunc object owns the
    // generated machine code.
    explicit NativeTransform(nb::object obj) : owner_(std::move(obj)) {
        uintptr_t address = 0;
        if (nb::hasattr(owner_, "address")) {
            address = nb::cast<uintptr_t>(owner_.attr("address"));
        } else {
            // Allow passing a raw integer address directly.
            try {
                address = nb::cast<uintptr_t>(owner_);
            } catch (...) {
                throw nb::type_error(
                    "NativeTransform expects a numba.cfunc (with an .address "
                    "attribute) or an integer function address");
            }
        }
        if (address == 0)
            throw nb::value_error("NativeTransform: null callback address");
        func_ = reinterpret_cast<Func>(address);
    }

    // Evaluate: fill A (pose) and Adot (its time derivative) at time t.
    void eval(double t, Eigen::Ref<Eigen::Matrix4d> A,
              Eigen::Ref<Eigen::Matrix4d> Adot) const {
        func_(t, A.data(), Adot.data());
    }

    // Python-facing __call__ returning (A, Adot).
    std::pair<Eigen::Matrix4d, Eigen::Matrix4d> call(double t) const {
        Eigen::Matrix4d A, Adot;
        func_(t, A.data(), Adot.data());
        return {A, Adot};
    }

    Func raw() const { return func_; }

    // A SweptTransform view.  Captures only the raw pointer (not the nb::object)
    // so it is safe to invoke with the GIL released.  The caller must keep the
    // owning NativeTransform alive for the duration of the call.
    SweptTransform as_transform() const {
        Func f = func_;
        return [f](double t, Eigen::Matrix4d &A, Eigen::Matrix4d &Adot) {
            f(t, A.data(), Adot.data());
        };
    }

private:
    Func func_ = nullptr;
    nb::object owner_;
};

// ---------------------------------------------------------------------------
// A representative C++ baseline transform (rotation about z + translation in x)
// used both as a benchmark reference and to demonstrate the ABI.
// ---------------------------------------------------------------------------
static void cpp_baseline_transform(double t, double *A, double *Adot) {
    const double c = std::cos(t), s = std::sin(t);
    for (int i = 0; i < 16; ++i) { A[i] = 0.0; Adot[i] = 0.0; }
    // column-major: A[col*4 + row]
    A[0] = c;  A[1] = s;               // col 0
    A[4] = -s; A[5] = c;               // col 1
    A[10] = 1.0;                       // col 2
    A[12] = t; A[15] = 1.0;            // col 3 (translation + homogeneous)
    Adot[0] = -s; Adot[1] = c;
    Adot[4] = -c; Adot[5] = -s;
    Adot[12] = 1.0;
}

// Sink to prevent the optimizer from eliminating benchmark loops.
static volatile double g_bench_sink = 0.0;

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------
NB_MODULE(_swept_volumes, m) {
    m.doc() = "Swept Volumes via Spacetime Numerical Continuation (nanobind)";

    nb::enum_<ContouringMethod>(m, "ContouringMethod")
        .value("MarchingCubes", ContouringMethod::MarchingCubes)
        .value("DualContouring", ContouringMethod::DualContouring);

    nb::class_<NativeTransform>(m, "NativeTransform",
        "Wraps a numba.cfunc(void(double, double*, double*)) trajectory.")
        .def(nb::init<nb::object>(), "cfunc"_a,
             "Construct from a numba cfunc (or raw integer address).")
        .def("__call__", &NativeTransform::call, "t"_a,
             "Evaluate the transform, returning (A, Adot) as 4x4 arrays.");

    // Functor-driven swept volume (the user-defined native transform).
    m.def(
        "swept_volume",
        [](const Eigen::MatrixXd &V, const Eigen::MatrixXi &F,
           const NativeTransform &transform, double eps, int num_seeds,
           const std::string &dir_name, ContouringMethod contouring,
           bool diagnostics) {
            Eigen::MatrixXd U;
            Eigen::MatrixXi G;
            std::vector<Eigen::MatrixXd> sV;
            std::vector<Eigen::MatrixXi> sF;
            SweptTransform tf = transform.as_transform();
            {
                nb::gil_scoped_release release;
                swept_volume(V, F, tf, eps, num_seeds, dir_name, contouring, U,
                             G, sV, sF, diagnostics);
            }
            return std::make_tuple(U, G, sV, sF);
        },
        "V"_a, "F"_a, "transform"_a, "eps"_a = 0.02, "num_seeds"_a = 100,
        "dir_name"_a = "swept_output",
        "contouring"_a = ContouringMethod::MarchingCubes,
        "diagnostics"_a = false,
        "Compute the swept volume of mesh (V,F) under a native transform(t).\n"
        "Returns (U, G, strobo_V_list, strobo_F_list). With diagnostics=True the\n"
        "strobo/stamping baseline is computed (into the strobo lists) and debug\n"
        "meshes are written under dir_name; off by default (they don't affect U,G).");

    // Keyframe-driven swept volume (backwards compatible trajectory).
    m.def(
        "swept_volume_keyframes",
        [](const Eigen::MatrixXd &V, const Eigen::MatrixXi &F,
           const std::vector<Eigen::Matrix4d> &keyframes, double eps,
           int num_seeds, const std::string &dir_name, bool diagnostics) {
            Eigen::MatrixXd U;
            Eigen::MatrixXi G;
            std::vector<Eigen::MatrixXd> sV;
            std::vector<Eigen::MatrixXi> sF;
            {
                nb::gil_scoped_release release;
                swept_volume(V, F, keyframes, eps, num_seeds, dir_name, U, G, sV,
                             sF, diagnostics);
            }
            return std::make_tuple(U, G, sV, sF);
        },
        "V"_a, "F"_a, "keyframes"_a, "eps"_a = 0.02, "num_seeds"_a = 100,
        "dir_name"_a = "swept_output", "diagnostics"_a = false,
        "Compute the swept volume from a list of 4x4 keyframe poses "
        "(Catmull-Rom translation + slerp rotation between keyframes).");

    // Slow reference path: a plain Python callable returning (A, Adot).  Every
    // evaluation re-enters Python -- provided only for correctness comparison.
    m.def(
        "swept_volume_pyfunc",
        [](const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, nb::callable fn,
           double eps, int num_seeds, const std::string &dir_name,
           ContouringMethod contouring) {
            Eigen::MatrixXd U;
            Eigen::MatrixXi G;
            std::vector<Eigen::MatrixXd> sV;
            std::vector<Eigen::MatrixXi> sF;
            SweptTransform tf = [&fn](double t, Eigen::Matrix4d &A,
                                      Eigen::Matrix4d &Adot) {
                nb::gil_scoped_acquire acquire;
                auto res = fn(t);
                auto tup = nb::cast<std::pair<Eigen::Matrix4d, Eigen::Matrix4d>>(res);
                A = tup.first;
                Adot = tup.second;
            };
            {
                nb::gil_scoped_release release;
                swept_volume(V, F, tf, eps, num_seeds, dir_name, contouring, U,
                             G, sV, sF);
            }
            return std::make_tuple(U, G, sV, sF);
        },
        "V"_a, "F"_a, "fn"_a, "eps"_a = 0.02, "num_seeds"_a = 100,
        "dir_name"_a = "swept_output",
        "contouring"_a = ContouringMethod::MarchingCubes,
        "Slow reference: transform is a Python callable t -> (A, Adot).");

    // ----- Microbenchmarks: isolate per-call transform overhead -----
    m.def(
        "bench_native",
        [](const NativeTransform &transform, int64_t n) {
            NativeTransform::Func f = transform.raw();
            nb::gil_scoped_release release;
            Eigen::Matrix4d A, Adot;
            auto t0 = std::chrono::steady_clock::now();
            double acc = 0.0;
            for (int64_t i = 0; i < n; ++i) {
                const double t = double(i) / double(n);
                f(t, A.data(), Adot.data());
                acc += A(0, 0);
            }
            auto t1 = std::chrono::steady_clock::now();
            g_bench_sink = acc;
            return std::chrono::duration<double>(t1 - t0).count();
        },
        "transform"_a, "n"_a,
        "Call a NativeTransform n times in a tight native loop; return seconds.");

    m.def(
        "bench_cpp_baseline",
        [](int64_t n) {
            void (*f)(double, double *, double *) = &cpp_baseline_transform;
            nb::gil_scoped_release release;
            Eigen::Matrix4d A, Adot;
            auto t0 = std::chrono::steady_clock::now();
            double acc = 0.0;
            for (int64_t i = 0; i < n; ++i) {
                const double t = double(i) / double(n);
                f(t, A.data(), Adot.data());
                acc += A(0, 0);
            }
            auto t1 = std::chrono::steady_clock::now();
            g_bench_sink = acc;
            return std::chrono::duration<double>(t1 - t0).count();
        },
        "n"_a,
        "Call the C++ baseline transform n times (indirect call); return seconds.");

    m.def(
        "bench_pyfunc",
        [](nb::callable fn, int64_t n) {
            // GIL held: every call re-enters Python.
            auto t0 = std::chrono::steady_clock::now();
            double acc = 0.0;
            for (int64_t i = 0; i < n; ++i) {
                const double t = double(i) / double(n);
                auto res = fn(t);
                auto tup = nb::cast<std::pair<Eigen::Matrix4d, Eigen::Matrix4d>>(res);
                acc += tup.first(0, 0);
            }
            auto t1 = std::chrono::steady_clock::now();
            g_bench_sink = acc;
            return std::chrono::duration<double>(t1 - t0).count();
        },
        "fn"_a, "n"_a,
        "Call a Python callable t -> (A, Adot) n times; return seconds.");

    m.attr("cpp_baseline_address") =
        reinterpret_cast<uintptr_t>(&cpp_baseline_transform);
}
