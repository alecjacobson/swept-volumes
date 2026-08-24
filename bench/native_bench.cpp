// Native C++ driver mirroring the Python binding path, for output/runtime
// comparison.  Uses the SAME analytic transform as tests/native_transform in
// Python (rotation about z + translation in x), so meshes should match.
//
// Usage:
//   swept-volumes-bench <mesh.obj|cube> <eps> <num_seeds> <out_dir> <mc|dc>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <igl/read_triangle_mesh.h>
#include "swept_volume.h"

// Analytic trajectory shared with the Python `spin` test transform.
static void fill_transform(double t, Eigen::Matrix4d &A, Eigen::Matrix4d &Adot) {
    const double c = std::cos(t), s = std::sin(t);
    A.setIdentity();
    A(0, 0) = c;  A(0, 1) = -s; A(0, 3) = t;
    A(1, 0) = s;  A(1, 1) = c;
    Adot.setZero();
    Adot(0, 0) = -s; Adot(0, 1) = -c; Adot(0, 3) = 1.0;
    Adot(1, 0) = c;  Adot(1, 1) = -s;
}

static void unit_cube(Eigen::MatrixXd &V, Eigen::MatrixXi &F) {
    V.resize(8, 3);
    V << 0, 0, 0,  1, 0, 0,  1, 1, 0,  0, 1, 0,
         0, 0, 1,  1, 0, 1,  1, 1, 1,  0, 1, 1;
    V.array() -= 0.5;  // center
    V *= 0.3;          // shrink so the sweep is modest
    F.resize(12, 3);
    F << 0, 2, 1,  0, 3, 2,  4, 5, 6,  4, 6, 7,
         0, 1, 5,  0, 5, 4,  1, 2, 6,  1, 6, 5,
         2, 3, 7,  2, 7, 6,  3, 0, 4,  3, 4, 7;
}

int main(int argc, char **argv) {
    std::string mesh = argc > 1 ? argv[1] : "cube";
    double eps = argc > 2 ? std::atof(argv[2]) : 0.05;
    int num_seeds = argc > 3 ? std::atoi(argv[3]) : 100;
    std::string out_dir = argc > 4 ? argv[4] : "native_output";
    std::string method = argc > 5 ? argv[5] : "mc";

    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    if (mesh == "cube") {
        unit_cube(V, F);
    } else if (!igl::read_triangle_mesh(mesh, V, F)) {
        std::cerr << "Failed to read mesh: " << mesh << std::endl;
        return 1;
    }

    ContouringMethod contouring = (method == "dc")
                                      ? ContouringMethod::DualContouring
                                      : ContouringMethod::MarchingCubes;

    SweptTransform tf = &fill_transform;

    // Per-call microbenchmark (indirect native call).
    {
        const int64_t n = 20'000'000;
        Eigen::Matrix4d A, Adot;
        auto t0 = std::chrono::steady_clock::now();
        double acc = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            tf(double(i) / double(n), A, Adot);
            acc += A(0, 0);
        }
        auto t1 = std::chrono::steady_clock::now();
        const double sec = std::chrono::duration<double>(t1 - t0).count();
        std::cerr << "[bench] transform: " << (sec / n * 1e9)
                  << " ns/call (acc=" << acc << ")" << std::endl;
    }

    Eigen::MatrixXd U;
    Eigen::MatrixXi G;
    std::vector<Eigen::MatrixXd> sV;
    std::vector<Eigen::MatrixXi> sF;

    auto t0 = std::chrono::steady_clock::now();
    swept_volume(V, F, tf, eps, num_seeds, out_dir, contouring, U, G, sV, sF);
    auto t1 = std::chrono::steady_clock::now();
    std::cerr << "[bench] swept_volume: "
              << std::chrono::duration<double>(t1 - t0).count() << " s"
              << std::endl;
    std::cout << "V " << U.rows() << " F " << G.rows() << std::endl;
    return 0;
}
