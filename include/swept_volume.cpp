#include <vector>
#include <limits>
#include <cstdint>
#include <unordered_map>
#include <array>
#include <Eigen/Geometry>
#include <igl/unique_simplices.h>
#include "swept_volume.h"
#include <igl/doublearea.h>
#include <igl/per_face_normals.h>
#include <igl/parallel_for.h>
#include <igl/readOBJ.h>
#include <igl/read_triangle_mesh.h>
#include <igl/writeOBJ.h>
#include <igl/random_points_on_mesh.h>
#include <igl/writePLY.h>
#include <igl/signed_distance.h>
#include <igl/dual_contouring.h>
#include <igl/sparse_voxel_grid.h>
#include <igl/upsample.h>
#include <igl/get_seconds.h>
#include <igl/facet_adjacency_matrix.h>
#include <igl/barycentric_coordinates.h>
#include <igl/grid.h>
#include <igl/copyleft/marching_cubes.h>
#include <igl/connected_components.h>
#include <igl/polygon_corners.h>
#include <igl/per_face_normals.h>
#include <igl/slice.h>
#include <igl/per_corner_normals.h>
#include <igl/swept_volume_signed_distance.h>
#include "gradient_descent_test.h"
#include "write_transformation.h"
#include "inigo_example.h"
#include "fd_interpolate.h"
#include <random>
#include "sparse_continuation.h"
#include "random_points_on_mesh.h"
#include <igl/fast_winding_number.h>
#include <igl/winding_number.h>
#include <igl/writeDMAT.h>


Eigen::Quaternion <double> logq(const Eigen::Quaternion <double> &q) {
    double exp_w = q.norm();
    double w = log(exp_w);
    double a = acos(q.w() / exp_w);
    
    if (a == 0.0) {
        return Eigen::Quaternion<double>(w, 0.0,0.0,0.0);
    }
    
    Eigen::Quaternion <double> res;
    res.w() = w;
    res.vec() = q.vec() / exp_w / (sin(a) / a);
    
    return res;
}

SweptTransform make_keyframe_transform(const std::vector<Eigen::Matrix4d> & Transformations){
    Eigen::VectorXd time_keyframes;
    time_keyframes.setLinSpaced(Transformations.size(),0.0,1.0);
    std::vector<Eigen::RowVector3d> tangents;
    double tau = time_keyframes(1);
    Eigen::RowVector3d running_tan;
    for (int mm = 0; mm < Transformations.size(); mm++) {
        if (mm == 0) {
            running_tan << (Transformations[1](0,3) - Transformations[0](0,3))/(1.0), (Transformations[1](1,3) - Transformations[0](1,3))/(1.0), (Transformations[1](2,3) - Transformations[0](2,3))/(1.0);
            tangents.push_back(running_tan);
        }else if (mm == Transformations.size()-1) {
            running_tan << (Transformations[Transformations.size()-1](0,3) - Transformations[Transformations.size()-2](0,3))/(1.0), (Transformations[Transformations.size()-1](1,3) - Transformations[Transformations.size()-2](1,3))/(1.0), (Transformations[Transformations.size()-1](2,3) - Transformations[Transformations.size()-2](2,3))/(1.0);
            tangents.push_back(running_tan);
        }else{
            running_tan << (Transformations[mm+1](0,3) - Transformations[mm-1](0,3))/(2.0), (Transformations[mm+1](1,3) - Transformations[mm-1](1,3))/(2.0), (Transformations[mm+1](2,3) - Transformations[mm-1](2,3))/(2.0);
            tangents.push_back(running_tan);
        }
    }
    
    // Position function (Catmull-Rom spline)
    std::function<bool(const double,
                       Eigen::RowVector3d &,
                       Eigen::RowVector3d &,
                       Eigen::Matrix3d &,
                       Eigen::Matrix3d &)> ip = [=](const double t,
                                                                      Eigen::RowVector3d & xt,
                                                                      Eigen::RowVector3d & vt,
                                                                      Eigen::Matrix3d & Rt,
                                                                      Eigen::Matrix3d & VRt)->bool{
                           Eigen::RowVector3d x0, x1, x2, x3, v0, v1, v2;
                           int a, b;
                           b = floor(t*(Transformations.size() - 1.0));
                           if (t==1.0) {
                               b = b-1;
                           }
                           double tt = (t-time_keyframes(b))/(tau);
                           x0 << Transformations[b](0,3),  Transformations[b](1,3),  Transformations[b](2,3);
                           x1 << Transformations[b+1](0,3),  Transformations[b+1](1,3),  Transformations[b+1](2,3);
                           xt = (2.0*pow(tt,3.0) - 3.0*pow(tt,2.0) + 1.0)*x0 + (pow(tt,3.0) - 2.0*pow(tt,2.0) + tt)*tangents[b] + (-2.0*pow(tt,3.0) + 3.0*pow(tt,2.0))*x1 + (pow(tt,3.0) - pow(tt,2.0))*tangents[b+1];
                           vt = (6.0*pow(tt,2.0) - 6.0*tt)*x0 + (3.0*pow(tt,2.0) - 4.0*tt + 1.0)*tangents[b] + (-6.0*pow(tt,2.0) + 6.0*tt)*x1 + (3.0*pow(tt,2.0) - 2.0*tt)*tangents[b+1];
                           vt = vt/tau;
                           
                           Eigen::Matrix3d R0, R1;
                           R0 = Transformations[b].topLeftCorner(3,3);
                           R1 = Transformations[b+1].topLeftCorner(3,3);
                           
                           // scale
                           Eigen::Matrix3d S0, S1, St,VSt;
                           S0.setZero();
                           S1.setZero();
                           S0(0,0) = R0.col(0).norm();
                           S0(1,1) = R0.col(1).norm();
                           S0(2,2) = R0.col(2).norm();
                           S1(0,0) = R1.col(0).norm();
                           S1(1,1) = R1.col(1).norm();
                           S1(2,2) = R1.col(2).norm();
                           St = S1 + (1.0-tt)*(S0-S1);
                           // rotation
                           
                           R0 = R0*S0.inverse();
                           R1 = R1*S1.inverse();
                           Eigen::Quaterniond q0(R0);
                           
                           Eigen::Quaterniond q1(R1);
                           q1.normalize();
                           q0.normalize();
                           Eigen::Quaterniond qt = q0.slerp(tt,q1);
                           Rt = qt.toRotationMatrix();
                           if (q0.dot(q1) < 0) {
                               q1.coeffs() = -q1.coeffs();
                           }
                           Eigen::Quaterniond qs = q0.conjugate() * q1;
                           Eigen::Quaterniond qvt = qt * logq(qs);
                           
                           
                           double qr, qi, qj, qk;
                           Eigen::Matrix3d Rr, Ri, Rj, Rk;
                           qr = qt.w();
                           qi = qt.x();
                           qj = qt.y();
                           qk = qt.z();
                           Rr << 0, -2*qk, 2*qj,
                           2*qk, 0, -2*qi,
                           -2*qj, 2*qi, 0;
                           Rk << -4*qk, -2*qr, 2*qi,
                           2*qr, -4*qk, 2*qj,
                           2*qi, 2*qj, 0;
                           Rj << -4*qj, 2*qi, 2*qr,
                           2*qi, 0, 2*qk,
                           -2*qr, 2*qk, -4*qj;
                           Ri << 0, 2*qj, 2*qk,
                           2*qj, -4*qi, -2*qr,
                           2*qk, 2*qr, -4*qi;
                           VRt = Rr*qvt.w() + Ri*qvt.x() + Rj*qvt.y() + Rk*qvt.z();
                           VRt = VRt / tau;
                           VSt = (S1-S0)/tau;
                           
                           // Scaling
                           Rt = Rt * St;
                           VRt = VRt * St + Rt * VSt;
                           
                           
                           // cool trajectory hack (uncomment to hack trajectory)
                           //                           xt << cos(5.*t), sin(5.*t), t;
                           //                           vt << -5.*sin(5.*t), 5.*cos(5.*t), 1.0;
                           //                           Rt << cos(5.*t), sin(5.*t), 0.0,
                           //                                 -sin(5.*t), cos(5.*t), 0.0,
                           //                                 0.0, 0.0, 1.0;
                           //                           VRt << -5.*sin(5.*t), 5.*cos(5.*t), 0.0,
                           //                                  -5.*cos(5.*t), -5.*sin(5.*t), 0.0,
                           //                                  0.0, 0.0, 0.0;
                           return true;
                       };
    return [ip](double t, Eigen::Matrix4d & A, Eigen::Matrix4d & Adot){
        Eigen::RowVector3d xt, vt;
        Eigen::Matrix3d Rt, VRt;
        ip(t, xt, vt, Rt, VRt);
        A.setIdentity();
        A.topLeftCorner<3,3>() = Rt;
        A.block<3,1>(0,3) = xt.transpose();
        Adot.setZero();
        Adot.topLeftCorner<3,3>() = VRt;
        Adot.block<3,1>(0,3) = vt.transpose();
    };
}

// Dual-contour the swept volume from exactly the continuation output that
// marching cubes uses -- the sparse cells (CS values, CV verts, CI cells).  No
// re-evaluation of the field (no stamping, no dense grid): DC is a pure function
// of (CS, CV, CI).
//
// libigl's sparse dual_contouring consumes the cells expressed as UNIQUE EDGES
// (#edges x 2), not the #cells x 8 matrix; it places one vertex per cell and one
// face per sign-change edge.  We give it:
//   * values : CS directly (crossings by interpolation, same as marching cubes).
//   * normals: the normalized central finite-difference gradient of CS on the
//     grid, lightly smoothed over grid neighbours.  (CS is accurate to ~the
//     continuation tolerance; the per-vertex argmin times are much coarser, so a
//     P-C-from-argmin normal is noticeably worse -- measured ~16 deg vs ~10 deg.)
static void dual_contour_continuation(
    const double eps,
    const Eigen::VectorXd & CS, const Eigen::MatrixXd & CV,
    const Eigen::MatrixXi & CI,
    Eigen::MatrixXd & U, Eigen::MatrixXi & G)
{
    const int n = (int)CV.rows();

    // Grid-vertex lookup (the continuation grid is regular with spacing eps).
    const Eigen::RowVector3d origin = CV.colwise().minCoeff();
    const auto keyijk = [](int64_t i, int64_t j, int64_t k) -> int64_t {
        return ((i + (1 << 20)) * 2097169 + (j + (1 << 20))) * 2097169 + (k + (1 << 20));
    };
    std::vector<std::array<long,3>> gc(n);
    std::unordered_map<int64_t, int> vmap;
    vmap.reserve(n * 2);
    for (int i = 0; i < n; ++i) {
        const Eigen::RowVector3d g = (CV.row(i) - origin) / eps;
        gc[i] = {std::lround(g(0)), std::lround(g(1)), std::lround(g(2))};
        vmap[keyijk(gc[i][0], gc[i][1], gc[i][2])] = i;
    }
    const auto vidx = [&](long i, long j, long k) -> int {
        auto it = vmap.find(keyijk(i, j, k));
        return it == vmap.end() ? -1 : it->second;
    };
    const auto nearest = [&](const Eigen::RowVector3d & P) -> int {
        const Eigen::RowVector3d g = (P - origin) / eps;
        const int v = vidx(std::lround(g(0)), std::lround(g(1)), std::lround(g(2)));
        if (v >= 0) return v;
        int best = 0; double bd = std::numeric_limits<double>::infinity();
        for (int i = 0; i < n; ++i) {
            const double d = (CV.row(i) - P).squaredNorm();
            if (d < bd) { bd = d; best = i; }
        }
        return best;
    };

    // Per-vertex normals = normalized central finite-difference gradient of the
    // continuation's own (accurate) values CS on the grid; one-sided at band
    // edges.  This needs no argmin time and no re-evaluation of the field.
    Eigen::MatrixXd Nv(n, 3);
    for (int i = 0; i < n; ++i) {
        Eigen::RowVector3d g(0, 0, 0);
        for (int d = 0; d < 3; ++d) {
            std::array<long,3> a = gc[i], b = gc[i];
            a[d] -= 1; b[d] += 1;
            const int va = vidx(a[0], a[1], a[2]), vb = vidx(b[0], b[1], b[2]);
            if (va >= 0 && vb >= 0)      g(d) = (CS(vb) - CS(va)) / (2 * eps);
            else if (vb >= 0)            g(d) = (CS(vb) - CS(i)) / eps;
            else if (va >= 0)            g(d) = (CS(i) - CS(va)) / eps;
        }
        const double gn = g.norm();
        Nv.row(i) = gn > 1e-12 ? Eigen::RowVector3d(g / gn) : Eigen::RowVector3d(0, 0, 1);
    }

    // Smooth the normal field over grid neighbours (a few Jacobi passes): the
    // continuation's per-vertex values carry high-frequency noise that FD turns
    // into normal jitter, which DC's QEF then amplifies into a wrinkled surface.
    for (int pass = 0; pass < 3; ++pass) {
        Eigen::MatrixXd Ns = Nv;
        for (int i = 0; i < n; ++i) {
            Eigen::RowVector3d acc = Nv.row(i);
            for (int d = 0; d < 3; ++d)
                for (int sgn = -1; sgn <= 1; sgn += 2) {
                    std::array<long,3> a = gc[i]; a[d] += sgn;
                    const int v = vidx(a[0], a[1], a[2]);
                    if (v >= 0) acc += Nv.row(v);
                }
            const double nn = acc.norm();
            if (nn > 1e-12) Ns.row(i) = acc / nn;
        }
        Nv.swap(Ns);
    }

    std::function<double(const Eigen::RowVector3d &)> f =
        [&](const Eigen::RowVector3d & P){ return CS(nearest(P)); };
    std::function<Eigen::RowVector3d(const Eigen::RowVector3d &)> f_grad =
        [&](const Eigen::RowVector3d & P){ return Eigen::RowVector3d(Nv.row(nearest(P))); };

    // Same cells as marching cubes, re-expressed as the unique set of cell edges.
    Eigen::Matrix<int, Eigen::Dynamic, 2> GI2;
    {
        Eigen::Matrix<int, Eigen::Dynamic, 2> all(CI.rows() * 12, 2);
        all << CI.col(0), CI.col(1),  CI.col(1), CI.col(2),  CI.col(2), CI.col(3),
               CI.col(3), CI.col(0),  CI.col(4), CI.col(5),  CI.col(5), CI.col(6),
               CI.col(6), CI.col(7),  CI.col(7), CI.col(4),  CI.col(0), CI.col(4),
               CI.col(1), CI.col(5),  CI.col(2), CI.col(6),  CI.col(3), CI.col(7);
        Eigen::VectorXi _1, _2;
        igl::unique_simplices(all, GI2, _1, _2);
    }

    const Eigen::RowVector3d step(eps, eps, eps);
    igl::dual_contouring(f, f_grad, step, CS, CV, GI2,
                         /*constrained=*/false, /*triangles=*/true,
                         /*root_finding=*/false, U, G);
}

static void run(const Eigen::MatrixXd & V, const Eigen::MatrixXi & F, const Eigen::MatrixXd & UV, const Eigen::MatrixXi & UVF, const SweptTransform & transform, const double eps, const int num_seeds, const std::string dir_name, const ContouringMethod contouring, const std::vector<Eigen::Matrix4d>* TransformationsPtr, Eigen::MatrixXd & U, Eigen::MatrixXi & G, std::vector<Eigen::MatrixXd> & strobo_V_list, std::vector<Eigen::MatrixXi> & strobo_F_list){
    double iso = 0.001;
    auto sgn = [](double val) -> double {
        return (double) ((double(0) < val) - (val < double(0)));
    };
    (void) sgn;
    auto interpolate_position = [&](const double t, Eigen::RowVector3d & xt, Eigen::RowVector3d & vt, Eigen::Matrix3d & Rt, Eigen::Matrix3d & VRt)->bool{
        Eigen::Matrix4d A, Adot;
        transform(t, A, Adot);
        Rt = A.topLeftCorner<3,3>();
        xt = A.block<3,1>(0,3).transpose();
        VRt = Adot.topLeftCorner<3,3>();
        vt = Adot.block<3,1>(0,3).transpose();
        return true;
    };

    igl::AABB<Eigen::MatrixXd,3> tree;
    tree.init(V,F);
    igl::FastWindingNumberBVH fwn_bvh;
    int order = 2;
    igl::fast_winding_number(V,F,order,fwn_bvh);

    
    //    /// PROFILING
    const auto & tictoc = []()
    {
        static double t_start = igl::get_seconds();
        double diff = igl::get_seconds()-t_start;
        t_start += diff;
        return diff;
    };
   
    
    
    std::vector<Eigen::RowVector3d> debug_vec;
    
    
    int distance_queries = 0;
    int grad_descent_queries = 0;
    std::function<double(const Eigen::RowVector3d &, double &, std::vector<std::vector<double>> &, std::vector<std::vector<double>> &, std::vector<std::vector<double>> &)> scalarFunc = [&](const Eigen::RowVector3d & P, double & time_seed, std::vector<std::vector<double>> & intervals, std::vector<std::vector<double>> & values, std::vector<std::vector<double>> & minima)->double{
        grad_descent_queries++;
        
        
        Eigen::RowVector3d running_closest_point = V.row(0);
        double running_sign = 1.0;
        
        std::function<double(const double)> f = [&](const double t)->double{
            int i;
            double s,sqrd,sqrd2,s2;
            Eigen::Matrix3d VRt,Rt;
            Eigen::RowVector3d xt,vt,pos,c,c2;
            interpolate_position(t,xt,vt,Rt,VRt);

            pos = ((Rt.inverse())*((P - xt).transpose())).transpose();
            // fast winding number
            Eigen::VectorXd w;
            igl::fast_winding_number(fwn_bvh,2.0,pos,w);
            s = 1.-2.*w(0);
            //running_sign = s;
            //double ub = (pos-running_closest_point) * (pos-running_closest_point).transpose();
            sqrd = tree.squared_distance(V,F,pos,i,c);
            distance_queries = distance_queries + 1;
            //return sgn(s)*sqrt(sqrd) - 0.0;
            //return s*(c-pos).lpNorm<1>();
            return s*sqrt(sqrd) - iso;
            //return inigo_example(pos,0.0);
        };
        
        // Gradient of f
        std::function<double(const double)> gf = [&](const double t)->double{
            int i;
            double s,sqrd,sqrd2,s2;
            Eigen::Matrix3d VRt,Rt;
            Eigen::RowVector3d xt,vt,pos,c,c2,point_velocity;
            //            xt = position(t);
            //            vt = velocity(t);
            //            Rt = rotation(t);
            //            VRt = rotational_velocity(t);
            interpolate_position(t,xt,vt,Rt,VRt);
            //pos = ((Rt.transpose())*((P - xt).transpose())).transpose();
            pos = ((Rt.inverse())*((P - xt).transpose())).transpose();
            // slow winding number
            //signed_distance_winding_number(tree,V,F,hier,pos,s,sqrd,i,c);
            // fast winding number
            Eigen::VectorXd w;
            igl::fast_winding_number(fwn_bvh,2.0,pos,w);
            s = 1.-2.*w(0);
            running_sign = s;
            //double ub = (pos-running_closest_point) * (pos-running_closest_point).transpose();
            sqrd = tree.squared_distance(V,F,pos,i,c);
//            if(running_sign>0){
//                sqrd = tree.squared_distance(V,F,pos,0.0,ub,i,c);
//            }else{
//                sqrd = tree.squared_distance(V,F,pos,ub,10.0,i,c);
//            }
            running_closest_point = c;
            
            Eigen::RowVector3d cp = c-pos;
            cp.normalize();
            point_velocity = (-Rt.inverse()*VRt*Rt.inverse()*(P.transpose() - xt.transpose()) - Rt.inverse()*vt.transpose()).transpose();
            return (-s)*cp.dot(point_velocity);
        };
        
        // Run gradient descent
        double distance, seed;
        if (intervals.size()==0) {
            std::vector<double> temp_interval;
            temp_interval.resize(0);
            intervals.push_back(temp_interval);
            values.push_back(temp_interval);
            minima.push_back(temp_interval);
        }
        gradient_descent_test(f,gf,time_seed,distance,seed, intervals[0], values[0], minima[0]);
        time_seed = seed; // updates seed so that we can add it to the queue in the next voxel
        return distance;
    };
    
    
    // // std::cout << "CHANGE THE THING!" << std::endl;
    
    
    srand(100);
    // Initialization
    Eigen::MatrixXd X,N,B;
    Eigen::VectorXi I;
    igl::per_face_normals(V,F,Eigen::Vector3d(0.0,0.0,-1.0).normalized(),N);
    igl::random_points_on_mesh(num_seeds,V,F,B,I,X);
    
    std::vector<double> init_times;
    init_times.push_back(0.0);
    std::vector<Eigen::RowVector3d> init_points;
    double minx, miny, minz;
    minx = 1000;
    miny = minx;
    minz = minx;
    Eigen::RowVector3d candidate;
    int counter = 0;
    init_times.push_back(0.0);
    for (int i = 0; i < X.rows(); i++) {
        Eigen::RowVector3d P = X.row(i);
        Eigen::Matrix3d VRt,Rt;
        Eigen::RowVector3d xt,vt,pos,c,c2,point_velocity,normal;
        for (double t = 0.0; t<=1.0; t = t+0.2) {
            
            interpolate_position(t,xt,vt,Rt,VRt);
            
            pos = (Rt*P.transpose()).transpose() + xt;
            
            point_velocity = (VRt*P.transpose()).transpose() + vt;
            
            point_velocity.normalize();
            normal = (Rt*N.row(I(i)).transpose()).transpose();
            normal.normalize();
            if ( (fabs(normal.dot(point_velocity))<0.05) || ( normal.dot(point_velocity)<0.0 && t==0.0) || ( normal.dot(point_velocity)>0.0 && t==1.0)) {
                candidate = pos + iso*normal;
                init_points.push_back(candidate);
                init_times.push_back(t);
                minx = std::min(minx,candidate(0));
                miny = std::min(miny,candidate(1));
                minz = std::min(minz,candidate(2));
            }
        }
        
    }
    
    std::vector<Eigen::RowVector3i> init_voxels;
    //init_points.push_back(Eigen::RowVector3d(0.0,0.0,1.0));
    //init_times.push_back(0.0);
    Eigen::RowVector3d p0;
    p0(0) = minx;
    p0(1) = miny;
    p0(2) = minz;
    init_voxels.resize(0);
    init_voxels.push_back(Eigen::RowVector3i(0,0,0));
    Eigen::RowVector3d this_point;
    int ix,iy,iz;
    for (int s = 0; s<init_points.size(); s++) {
        this_point = init_points[s];
        ix = std::floor((this_point[0] - minx)/eps);
        iy = std::floor((this_point[1] - miny)/eps);
        iz = std::floor((this_point[2] - minz)/eps);
        init_voxels.push_back(Eigen::RowVector3i(ix,iy,iz));
    }
    
    time_t tstart, tend;
    Eigen::MatrixXi CI, CI_10;
    Eigen::MatrixXd CV, CV_10;
    Eigen::VectorXd CS, CS_10;
    Eigen::VectorXd CV_argmins, CV_argmins_10;
    std::cout << "Starting continuation with " << init_voxels.size() << " seeds." << std::endl;
    tictoc();
    sparse_continuation(p0,init_voxels,init_times,scalarFunc,eps,1000000,CS,CV,CI,CV_argmins);
    std::cout << "Continuation took "<< tictoc() <<" second(s)."<< std::endl;
     std::cout << "Number of (unique) vertices: " << CV.rows() << std::endl;
     std::cout << "Gradient descent queries: " << grad_descent_queries << std::endl;
     std::cout << "Gradient descent queries per vertex: " << ((double) grad_descent_queries)/( (double) CV.rows()) << std::endl;
     std::cout << "Distance queries: " << distance_queries << std::endl;
     std::cout << "Distance queries per vertex: " << ((double) distance_queries)/( (double) CV.rows()) << std::endl;
    Eigen::MatrixXd input_data,output_data;
    input_data.resize(CV_argmins.size(),5);
    //input_data.col(0) = CV_argmins;
    // Get normals and UVs
    Eigen::MatrixXd normals;
    igl::per_face_normals(V,F,normals);
    for (int i = 0; i < CV.rows(); i++) {
        int k;
        double s,sqrd,sqrd2,s2;
        Eigen::Matrix3d VRt,Rt;
        Eigen::RowVector3d xt,vt,pos,c,c2,P;
        P = CV.row(i).transpose();
        interpolate_position(CV_argmins(i),xt,vt,Rt,VRt);
        pos = ((Rt.inverse())*((P - xt).transpose())).transpose();
        sqrd = tree.squared_distance(V,F,pos,k,c);
        if (UV.rows()==0) {
            input_data.row(i) << CV_argmins(i), 0.0, normals(k,0), normals(k,1), normals(k,2);
        }else{
            Eigen::MatrixXd B;
            igl::barycentric_coordinates(c,V.row(F(k,0)), V.row(F(k,1)), V.row(F(k,2)), B);
            Eigen::Vector2d uv = (B(0)*UV.row(UVF(k,0)) + B(1)*UV.row(UVF(k,1)) + B(2)*UV.row(UVF(k,2)))/1.0;
            input_data.row(i) << uv(0), uv(1), normals(k,0), normals(k,1), normals(k,2);
        }
    }
    
    
    std::string make_dir = "mkdir ";
    std::system((make_dir + dir_name).c_str());
    Eigen::MatrixXd Umc;
    Eigen::MatrixXi Gmc;
    if (contouring == ContouringMethod::DualContouring) {
        dual_contour_continuation(eps, CS, CV, CI, U, G);
    } else {
        igl::copyleft::marching_cubes(CS,CV,CI,0.0,U,G); // our mesh
    }


    igl::writeOBJ(dir_name + "/input.obj",V,F);
    if (TransformationsPtr) {
        write_transformation(dir_name + "/transformations.dmat",*TransformationsPtr);
    }
    igl::writeOBJ(dir_name + "/ours.obj",U,G);
    
    
    // Strobo
    const auto & transform_affine = [&](const double t)->Eigen::Affine3d
    {
        Eigen::Affine3d T = Eigen::Affine3d::Identity();
        Eigen::Matrix3d VRt,Rt;
        Eigen::RowVector3d xt,vt;
        interpolate_position(t,xt,vt,Rt,VRt);
        Eigen::Matrix3d rot = Rt;
        Eigen::Vector3d pos = xt.transpose();
        T.rotate(rot);
        T.translate(rot.transpose()*pos);
        return T;
    };
    
    // COMPARE TO STROBO
    int grid_size_x = std::floor((CV.col(0).maxCoeff() - CV.col(0).minCoeff())/eps);
    int grid_size_y = std::floor((CV.col(1).maxCoeff() - CV.col(1).minCoeff())/eps);
    int grid_size_z = std::floor((CV.col(2).maxCoeff() - CV.col(2).minCoeff())/eps);
    Eigen::Vector3i res;
    res.resize(3);
    res << grid_size_x + 1, grid_size_y + 1, grid_size_z + 1;
    // Generate grid in GV, res
    Eigen::MatrixXd GV;
    GV.resize(0,0);
    igl::grid(res,GV);
    // make GV start at p0
    
    Eigen::RowVector3d factor;
    factor << (CV.col(0).maxCoeff() - CV.col(0).minCoeff()), (CV.col(1).maxCoeff() - CV.col(1).minCoeff()), (CV.col(2).maxCoeff() - CV.col(2).minCoeff());
    GV.col(0) *= factor(0);
    GV.col(1) *= factor(1);
    GV.col(2) *= factor(2);
    Eigen::RowVector3d offset;
    offset << CV.col(0).minCoeff(), CV.col(1).minCoeff(), CV.col(2).minCoeff();
    //    GV.rowwise() += p0;
    GV.rowwise() += offset;
    Eigen::VectorXi divs(2);
    divs << 10, 100;
    Eigen::MatrixXd U_10, U_10_dc;
    Eigen::MatrixXi G_10, G_10_dc;
    for (int dd = 0; dd<divs.size(); dd = dd + 1) {
        int div = divs(dd);
        tictoc();
        // Call distances
        Eigen::VectorXd S;
        igl::swept_volume_signed_distance(V,F,transform_affine,div,GV,res,eps,iso,S);
        igl::copyleft::marching_cubes(S,GV,res(0),res(1),res(2),U_10,G_10);
        strobo_V_list.push_back(U_10);
        strobo_F_list.push_back(G_10);
        igl::writeOBJ(dir_name + "/strobo_" + std::to_string(div) + "mc.obj",U_10,G_10);
}

}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// Functor-driven (with UVs)
void swept_volume(const Eigen::MatrixXd & V, const Eigen::MatrixXi & F, const Eigen::MatrixXd & UV, const Eigen::MatrixXi & UVF, const SweptTransform & transform, const double eps, const int num_seeds, const std::string dir_name, const ContouringMethod contouring, Eigen::MatrixXd & U, Eigen::MatrixXi & G, std::vector<Eigen::MatrixXd> & strobo_V_list, std::vector<Eigen::MatrixXi> & strobo_F_list){
    run(V,F,UV,UVF,transform,eps,num_seeds,dir_name,contouring,nullptr,U,G,strobo_V_list,strobo_F_list);
}

// Functor-driven (no UVs)
void swept_volume(const Eigen::MatrixXd & V, const Eigen::MatrixXi & F, const SweptTransform & transform, const double eps, const int num_seeds, const std::string dir_name, const ContouringMethod contouring, Eigen::MatrixXd & U, Eigen::MatrixXi & G, std::vector<Eigen::MatrixXd> & strobo_V_list, std::vector<Eigen::MatrixXi> & strobo_F_list){
    Eigen::MatrixXd UV(0,0);
    Eigen::MatrixXi UVF(0,0);
    run(V,F,UV,UVF,transform,eps,num_seeds,dir_name,contouring,nullptr,U,G,strobo_V_list,strobo_F_list);
}

// Keyframe (with UVs) -- MarchingCubes, backwards compatible
void swept_volume(const Eigen::MatrixXd & V, const Eigen::MatrixXi & F, const Eigen::MatrixXd & UV, const Eigen::MatrixXi & UVF, const std::vector<Eigen::Matrix4d> Transformations, const double eps, const int num_seeds, const std::string dir_name, Eigen::MatrixXd & U, Eigen::MatrixXi & G, std::vector<Eigen::MatrixXd> & strobo_V_list, std::vector<Eigen::MatrixXi> & strobo_F_list){
    SweptTransform tf = make_keyframe_transform(Transformations);
    run(V,F,UV,UVF,tf,eps,num_seeds,dir_name,ContouringMethod::MarchingCubes,&Transformations,U,G,strobo_V_list,strobo_F_list);
}

// Keyframe (no UVs) -- MarchingCubes, backwards compatible
void swept_volume(const Eigen::MatrixXd & V, const Eigen::MatrixXi & F, const std::vector<Eigen::Matrix4d> Transformations, const double eps, const int num_seeds, const std::string dir_name, Eigen::MatrixXd & U, Eigen::MatrixXi & G, std::vector<Eigen::MatrixXd> & strobo_V_list, std::vector<Eigen::MatrixXi> & strobo_F_list){
    Eigen::MatrixXd UV;
    UV.resize(0,0);
    Eigen::MatrixXi UVF;
    UVF.resize(0,0);
    swept_volume(V,F,UV,UVF,Transformations,eps,num_seeds,dir_name,U,G,strobo_V_list,strobo_F_list);
}



