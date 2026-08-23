// Analytic swept-volume sanity check.
//
// A cube (box half-extents h) translated along x by t*L, t in [0,1].  Its swept
// volume is EXACTLY a larger box (half-extents (L/2+h_x, h_y, h_z), centered at
// (L/2,0,0)) whose SDF and gradient are known in closed form.
//
// We contour, with the same sparse pipeline (sparse_voxel_grid -> cells for MC,
// -> unique edges for DC):
//   (A) the TRUE box SDF (exact field + gradient), and
//   (B) the SWEEP SDF  f(P) = min_t box_sdf(P - c(t)) with the EXACT envelope
//       gradient (box gradient at the argmin shift; argmin t* = clamp(Px/L)).
// If DC on (A) has crisp box edges but DC on (B) is wrinkled/rounded, the
// sweep-SDF field/gradient is the culprit; if both are crisp, our swept-volume
// wrinkle comes from the continuation's approximate values/argmin-times.
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <map>
#include <utility>

#include <igl/dual_contouring.h>
#include <igl/marching_cubes.h>
#include <igl/sparse_voxel_grid.h>
#include <igl/unique_simplices.h>
#include <igl/writeOBJ.h>

using Eigen::RowVector3d;

static double box_sdf(const RowVector3d &p, const RowVector3d &h) {
    const RowVector3d d = p.cwiseAbs() - h;
    const double outside = d.cwiseMax(0.0).norm();
    const double inside = std::min(d.maxCoeff(), 0.0);
    return outside + inside;
}

static RowVector3d box_grad(const RowVector3d &p, const RowVector3d &h) {
    const RowVector3d d = p.cwiseAbs() - h;
    const RowVector3d sgn(p(0) < 0 ? -1 : 1, p(1) < 0 ? -1 : 1, p(2) < 0 ? -1 : 1);
    if (d.maxCoeff() > 0.0) {  // outside: gradient of ||max(d,0)||
        RowVector3d q = d.cwiseMax(0.0).cwiseProduct(sgn);
        const double nn = q.norm();
        return nn > 0 ? (q / nn).eval() : RowVector3d(0, 0, 1);
    }
    int a; d.maxCoeff(&a);      // inside: along the least-interior axis
    RowVector3d g(0, 0, 0); g(a) = sgn(a);
    return g;
}

static void mesh_stats(const char *name, const Eigen::MatrixXd &V,
                       const Eigen::MatrixXi &F) {
    std::map<std::pair<int, int>, int> ec;
    for (int f = 0; f < F.rows(); ++f)
        for (int k = 0; k < 3; ++k) {
            int a = F(f, k), b = F(f, (k + 1) % 3);
            if (a > b) std::swap(a, b);
            ec[{a, b}]++;
        }
    int boundary = 0, nonman = 0;
    for (auto &kv : ec) { if (kv.second == 1) boundary++; else if (kv.second > 2) nonman++; }
    printf("  [%s] V=%ld F=%ld boundary=%d nonmanifold=%d\n", name,
           (long)V.rows(), (long)F.rows(), boundary, nonman);
}

// Max distance of mesh vertices from the true box surface (how crisp/correct).
static void deviation(const char *name, const Eigen::MatrixXd &V,
                      const RowVector3d &center, const RowVector3d &H) {
    double mx = 0, sum = 0;
    for (int i = 0; i < V.rows(); ++i) {
        const double d = std::abs(box_sdf(RowVector3d(V.row(i)) - center, H));
        mx = std::max(mx, d); sum += d;
    }
    printf("  [%s] |true SDF| at verts: max=%.4g mean=%.4g\n", name, mx,
           sum / std::max<long>(1, V.rows()));
}

static void contour(const char *tag,
                    const std::function<double(const RowVector3d &)> &f,
                    const std::function<RowVector3d(const RowVector3d &)> &fg,
                    const RowVector3d &p0, double h, const RowVector3d &center,
                    const RowVector3d &H) {
    Eigen::MatrixXd GV; Eigen::VectorXd Gf;
    Eigen::Matrix<int, Eigen::Dynamic, 8> GI;
    igl::sparse_voxel_grid(p0, f, h, 16.0 * std::pow(h, -2.0), Gf, GV, GI);

    Eigen::MatrixXd mcV; Eigen::MatrixXi mcF;
    igl::marching_cubes(Gf, GV, GI, 0.0, mcV, mcF);

    Eigen::Matrix<int, Eigen::Dynamic, 2> GI2;
    {
        Eigen::Matrix<int, Eigen::Dynamic, 2> all(GI.rows() * 12, 2);
        all << GI.col(0), GI.col(1), GI.col(1), GI.col(2), GI.col(2), GI.col(3),
            GI.col(3), GI.col(0), GI.col(4), GI.col(5), GI.col(5), GI.col(6),
            GI.col(6), GI.col(7), GI.col(7), GI.col(4), GI.col(0), GI.col(4),
            GI.col(1), GI.col(5), GI.col(2), GI.col(6), GI.col(3), GI.col(7);
        Eigen::VectorXi _1, _2;
        igl::unique_simplices(all, GI2, _1, _2);
    }
    const RowVector3d step(h, h, h);
    Eigen::MatrixXd dcV; Eigen::MatrixXi dcF;
    igl::dual_contouring(f, fg, step, Gf, GV, GI2, false, true, true, dcV, dcF);

    printf("[%s] %ld cells\n", tag, (long)GI.rows());
    mesh_stats("MC", mcV, mcF); deviation("MC", mcV, center, H);
    mesh_stats("DC", dcV, dcF); deviation("DC", dcV, center, H);
    igl::writeOBJ(std::string("/tmp/an_") + tag + "_mc.obj", mcV, mcF);
    igl::writeOBJ(std::string("/tmp/an_") + tag + "_dc.obj", dcV, dcF);
}

int main() {
    const RowVector3d h(0.2, 0.2, 0.2);   // brush half-extents
    const double L = 0.8;                  // translation length along x
    const RowVector3d center(L / 2, 0, 0);
    const RowVector3d H(L / 2 + h(0), h(1), h(2));  // swept box half-extents

    // (A) true box
    const std::function<double(const RowVector3d &)> f_true =
        [&](const RowVector3d &p) { return box_sdf(p - center, H); };
    const std::function<RowVector3d(const RowVector3d &)> g_true =
        [&](const RowVector3d &p) { return box_grad(p - center, H); };

    // (B) sweep SDF: min_t box_sdf(P - (tL,0,0)); argmin t* = clamp(Px/L,0,1)
    const auto shift = [&](const RowVector3d &p) {
        const double t = std::min(1.0, std::max(0.0, p(0) / L));
        return RowVector3d(p(0) - t * L, p(1), p(2));
    };
    const std::function<double(const RowVector3d &)> f_sweep =
        [&](const RowVector3d &p) { return box_sdf(shift(p), h); };
    const std::function<RowVector3d(const RowVector3d &)> g_sweep =
        [&](const RowVector3d &p) { return box_grad(shift(p), h); };  // envelope gradient

    // sanity: the two fields agree near the boundary
    {
        double mx = 0;
        for (int i = 0; i < 20000; ++i) {
            RowVector3d p(-0.6 + 1.6 * (i % 37) / 37.0,
                          -0.4 + 0.8 * ((i / 37) % 29) / 29.0,
                          -0.4 + 0.8 * ((i / 37 / 29) % 23) / 23.0);
            if (std::abs(f_true(p)) < 0.05) mx = std::max(mx, std::abs(f_true(p) - f_sweep(p)));
        }
        printf("max |f_true - f_sweep| near boundary: %.3e\n", mx);
    }

    const RowVector3d seed(L / 2, 0, H(2));  // a point on the top face of the box
    contour("true", f_true, g_true, seed, 0.03, center, H);
    contour("sweep", f_sweep, g_sweep, seed, 0.03, center, H);
    return 0;
}
