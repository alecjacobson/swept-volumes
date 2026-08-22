#ifndef SWEPT_VOLUME_FUN
#define SWEPT_VOLUME_FUN
#include <Eigen/Core>
#include <functional>
#include <string>
#include <vector>

// Contouring back-end used to extract the final mesh from the sparse cell set
// (both operate on the sparse grid produced by the continuation).
enum class ContouringMethod {
    MarchingCubes,   // igl::copyleft::marching_cubes on (CS, CV, CI)
    DualContouring   // igl::dual_contouring (sparse overload) on the same cells
};

// User-defined continuous trajectory.  Given a time t in [0,1] it must fill:
//   A    : the 4x4 affine pose  [ R | x ; 0 0 0 1 ]  at time t
//   Adot : its time-derivative  [ R'| x'; 0 0 0 0 ]  at time t
// The derivative is required by the spacetime continuation (it needs the
// velocity of the trajectory, not just the pose).  Storage is column-major to
// match Eigen / numba's farray view.
using SweptTransform =
    std::function<void(double t, Eigen::Matrix4d & A, Eigen::Matrix4d & Adot)>;

// Build a SweptTransform from a list of keyframe poses (Catmull-Rom position
// spline + quaternion slerp with linearly interpolated scale), reproducing the
// original keyframe trajectory analytically (pose and derivative).
SweptTransform make_keyframe_transform(const std::vector<Eigen::Matrix4d> & Transformations);

// ---------------------------------------------------------------------------
// Functor-driven API (a user-defined continuous transform)
// ---------------------------------------------------------------------------
void swept_volume(const Eigen::MatrixXd & V, const Eigen::MatrixXi & F, const Eigen::MatrixXd & UV, const Eigen::MatrixXi & UVF, const SweptTransform & transform, const double eps, const int num_seeds, const std::string dir_name, const ContouringMethod contouring, Eigen::MatrixXd & U, Eigen::MatrixXi & G, std::vector<Eigen::MatrixXd> & strobo_V_list, std::vector<Eigen::MatrixXi> & strobo_F_list);

void swept_volume(const Eigen::MatrixXd & V, const Eigen::MatrixXi & F, const SweptTransform & transform, const double eps, const int num_seeds, const std::string dir_name, const ContouringMethod contouring, Eigen::MatrixXd & U, Eigen::MatrixXi & G, std::vector<Eigen::MatrixXd> & strobo_V_list, std::vector<Eigen::MatrixXi> & strobo_F_list);

// ---------------------------------------------------------------------------
// Keyframe API (backwards compatible; MarchingCubes by default)
// ---------------------------------------------------------------------------
void swept_volume(const Eigen::MatrixXd & V, const Eigen::MatrixXi & F, const Eigen::MatrixXd & UV, const Eigen::MatrixXi & UVF, const std::vector<Eigen::Matrix4d> Transformations, const double eps, const int num_seeds, const std::string dir_name, Eigen::MatrixXd & U, Eigen::MatrixXi & G, std::vector<Eigen::MatrixXd> & strobo_V_list, std::vector<Eigen::MatrixXi> & strobo_F_list);

void swept_volume(const Eigen::MatrixXd & V, const Eigen::MatrixXi & F, const std::vector<Eigen::Matrix4d> Transformations, const double eps, const int num_seeds, const std::string dir_name, Eigen::MatrixXd & U, Eigen::MatrixXi & G, std::vector<Eigen::MatrixXd> & strobo_V_list, std::vector<Eigen::MatrixXi> & strobo_F_list);

#endif
