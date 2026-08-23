// Static-geometry experiment: can the SAME sparse cells feed both sparse
// marching cubes and sparse dual contouring?
//
// Mirrors libigl's 805_MeshImplicitFunction: sparse_voxel_grid -> (Gf,GV,GI);
// marching_cubes consumes the cells GI (#cells x 8); dual_contouring consumes
// the UNIQUE EDGES GI2 (#edges x 2) derived from those same cells.  Uses an
// exact analytic gradient (sphere) so gradient quality is not a confound.
//
// Prints watertightness diagnostics (every manifold edge shared by exactly 2
// faces, Euler characteristic) for both meshes.
#include <Eigen/Core>
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

// #boundary_edges (edges used by != 2 faces), and Euler characteristic V-E+F.
static void mesh_stats(const char *name, const Eigen::MatrixXd &V,
                       const Eigen::MatrixXi &F) {
    // count undirected edges and their face-incidence
    std::map<std::pair<int, int>, int> edge_count;
    for (int f = 0; f < F.rows(); ++f) {
        for (int k = 0; k < 3; ++k) {
            int a = F(f, k), b = F(f, (k + 1) % 3);
            if (a > b) std::swap(a, b);
            edge_count[{a, b}]++;
        }
    }
    int boundary = 0, nonmanifold = 0;
    for (auto &kv : edge_count) {
        if (kv.second == 1) boundary++;
        else if (kv.second > 2) nonmanifold++;
    }
    const int E = (int)edge_count.size();
    printf("  [%s] V=%ld F=%ld E=%d  boundary_edges=%d  nonmanifold_edges=%d  "
           "V-E+F=%ld\n",
           name, (long)V.rows(), (long)F.rows(), E, boundary, nonmanifold,
           (long)(V.rows() - E + F.rows()));
}

int main() {
    // A sphere: exact SDF and exact gradient.
    const Eigen::RowVector3d c(0.0, 0.0, 0.0);
    const double r = 0.7;
    const std::function<double(const Eigen::RowVector3d &)> f =
        [&](const Eigen::RowVector3d &x) { return (x - c).norm() - r; };
    const std::function<Eigen::RowVector3d(const Eigen::RowVector3d &)> f_grad =
        [&](const Eigen::RowVector3d &x) {
            return (x - c).normalized().eval();
        };

    const double h = 0.05;                 // cell size
    const Eigen::RowVector3d p0(0.0, 0.0, r);  // on the zero level set

    // Sparse voxel grid: cells GI (#cells x 8), values Gf, vertices GV.
    Eigen::MatrixXd GV;
    Eigen::VectorXd Gf;
    Eigen::Matrix<int, Eigen::Dynamic, 8> GI;
    igl::sparse_voxel_grid(p0, f, h, 16.0 * std::pow(h, -2.0), Gf, GV, GI);
    printf("sparse_voxel_grid: %ld cells, %ld verts\n", (long)GI.rows(),
           (long)GV.rows());

    // ---- Marching cubes on the cells (as libigl intends) ----
    Eigen::MatrixXd mcV;
    Eigen::MatrixXi mcF;
    igl::marching_cubes(Gf, GV, GI, 0.0, mcV, mcF);
    mesh_stats("MC ", mcV, mcF);

    // ---- Dual contouring on the SAME cells, expressed as UNIQUE EDGES ----
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
    printf("unique edges for DC: %ld\n", (long)GI2.rows());

    const Eigen::RowVector3d step(h, h, h);
    Eigen::MatrixXd dcV;
    Eigen::MatrixXi dcF;
    igl::dual_contouring(f, f_grad, step, Gf, GV, GI2,
                         /*constrained=*/false, /*triangles=*/true,
                         /*root_finding=*/true, dcV, dcF);
    mesh_stats("DC ", dcV, dcF);

    igl::writeOBJ("/tmp/exp_mc.obj", mcV, mcF);
    igl::writeOBJ("/tmp/exp_dc.obj", dcV, dcF);
    printf("wrote /tmp/exp_mc.obj and /tmp/exp_dc.obj\n");
    return 0;
}
