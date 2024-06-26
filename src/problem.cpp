#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <Eigen/Sparse>
#include <glog/logging.h>
#include "myslam/problem.h"
#include "myslam/tic_toc.h"
#include "myslam/g2o_types.h"

#ifdef USE_OPENMP

#include <omp.h>

#endif

using namespace std;

// define the format you want, you only need one instance of this...
const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

void writeToCSVfile(std::string name, Eigen::MatrixXd matrix) {
    std::ofstream f(name.c_str());
    f << matrix.format(CSVFormat);
}

namespace myslam {

void Problem::LogoutVectorSize() {
    // LOG(INFO) <<
    //           "1 problem::LogoutVectorSize verticies_:" << verticies_.size() <<
    //           " edges:" << edges_.size();
}

Problem::Problem(ProblemType problemType) :
    problemType_(problemType) {
    LogoutVectorSize();
    verticies_marg_.clear();
}

Problem::~Problem() {}

bool Problem::AddVertex(std::shared_ptr<Vertex> vertex) {
    if (verticies_.find(vertex->Id()) != verticies_.end()) {
        // LOG(WARNING) << "Vertex " << vertex->Id() << " has been added before";
        return false;
    } else {
        verticies_.insert(pair<unsigned long, shared_ptr<Vertex>>(vertex->Id(), vertex));
    }

    if (problemType_ == ProblemType::SLAM_PROBLEM) {
        if (IsPoseVertex(vertex)) {
            ResizePoseHessiansWhenAddingPose(vertex);
        }
    }
    return true;
}

// 设置H中优化变量得顺序
void Problem::AddOrderingSLAM(std::shared_ptr<myslam::Vertex> v) {
    if (IsPoseVertex(v)) {  // Pose顶点
        v->SetOrderingId(ordering_poses_);
        idx_pose_vertices_.insert(pair<ulong, std::shared_ptr<Vertex>>(v->Id(), v));
        ordering_poses_ += v->LocalDimension();     // 顶点实际优化的维度，比如姿态就是6+6+6+6+...
    } else if (IsLandmarkVertex(v)) {
        v->SetOrderingId(ordering_landmarks_);
        ordering_landmarks_ += v->LocalDimension();
        idx_landmark_vertices_.insert(pair<ulong, std::shared_ptr<Vertex>>(v->Id(), v));
    }
}

void Problem::ResizePoseHessiansWhenAddingPose(shared_ptr<Vertex> v) {

    int size = H_prior_.rows() + v->LocalDimension();
    H_prior_.conservativeResize(size, size);
    b_prior_.conservativeResize(size);

    b_prior_.tail(v->LocalDimension()).setZero();
    H_prior_.rightCols(v->LocalDimension()).setZero();
    H_prior_.bottomRows(v->LocalDimension()).setZero();

}

bool Problem::IsPoseVertex(std::shared_ptr<myslam::Vertex> v) {
    string type = v->TypeInfo();
    return type == string("VertexPose");
}

bool Problem::IsLandmarkVertex(std::shared_ptr<myslam::Vertex> v) {
    string type = v->TypeInfo();
    return type == string("VertexPointXYZ");
}

bool Problem::AddEdge(shared_ptr<Edge> edge) {
    if (edges_.find(edge->Id()) == edges_.end()) {
        edges_.insert(pair<ulong, std::shared_ptr<Edge>>(edge->Id(), edge));
    } else {
        // LOG(WARNING) << "Edge " << edge->Id() << " has been added before!";
        return false;
    }

    for (auto &vertex: edge->Verticies()) {
        vertexToEdge_.insert(pair<ulong, shared_ptr<Edge>>(vertex->Id(), edge));
    }
    return true;
}

vector<shared_ptr<Edge>> Problem::GetConnectedEdges(std::shared_ptr<Vertex> vertex) {
    vector<shared_ptr<Edge>> edges;
    auto range = vertexToEdge_.equal_range(vertex->Id());
    for (auto iter = range.first; iter != range.second; ++iter) {

        // 并且这个edge还需要存在，而不是已经被remove了
        if (edges_.find(iter->second->Id()) == edges_.end())
            continue;

        edges.emplace_back(iter->second);
    }
    return edges;
}

bool Problem::RemoveVertex(std::shared_ptr<Vertex> vertex) {
    //check if the vertex is in map_verticies_
    if (verticies_.find(vertex->Id()) == verticies_.end()) {
        // LOG(WARNING) << "The vertex " << vertex->Id() << " is not in the problem!" << endl;
        return false;
    }

    // 这里要 remove 该顶点对应的 edge.
    vector<shared_ptr<Edge>> remove_edges = GetConnectedEdges(vertex);
    for (size_t i = 0; i < remove_edges.size(); i++) {
        RemoveEdge(remove_edges[i]);
    }

    if (IsPoseVertex(vertex))
        idx_pose_vertices_.erase(vertex->Id());
    else
        idx_landmark_vertices_.erase(vertex->Id());

    vertex->SetOrderingId(-1);      // used to debug
    verticies_.erase(vertex->Id());
    vertexToEdge_.erase(vertex->Id());

    return true;
}

bool Problem::RemoveEdge(std::shared_ptr<Edge> edge) {
    //check if the edge is in map_edges_
    if (edges_.find(edge->Id()) == edges_.end()) {
        // LOG(WARNING) << "The edge " << edge->Id() << " is not in the problem!" << endl;
        return false;
    }

    edges_.erase(edge->Id());
    return true;
}

bool Problem::Solve(int iterations) {


    if (edges_.size() == 0 || verticies_.size() == 0) {
        std::cerr << "\nCannot solve problem without edges or verticies" << std::endl;
        return false;
    }

    TicToc t_solve;
    // 统计优化变量的维数，为构建 H 矩阵做准备
    SetOrdering();
    // 遍历edge, 构建 H 矩阵
    MakeHessian();
    // LM 初始化
    ComputeLambdaInitLM();
    // LM 算法迭代求解
    bool stop = false;
    int iter = 0;
    while (!stop && (iter < iterations)) {
        std::cout << "iter: " << iter << " , chi= " << currentChi_ << " , Lambda= " << currentLambda_ << std::endl;
        bool oneStepSuccess = false;
        int false_cnt = 0;
        while (!oneStepSuccess)  // 不断尝试 Lambda, 直到成功迭代一步
        {
            // setLambda
//            AddLambdatoHessianLM();
            // 第四步，解线性方程
            auto start = std::chrono::high_resolution_clock::now();
            SolveLinearSystem();
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "SolveLinearSystem Time taken: " << duration.count() << " ms" << std::endl;

            //
//            RemoveLambdaHessianLM();

            // 优化退出条件1： delta_x_ 很小则退出
            if (delta_x_.squaredNorm() <= 1e-6 || false_cnt > 10) {
                stop = true;
                break;
            }

            // 更新状态量
            UpdateStates();
            // 判断当前步是否可行以及 LM 的 lambda 怎么更新
            oneStepSuccess = IsGoodStepInLM();
            // 后续处理，
            if (oneStepSuccess) {
                // 在新线性化点 构建 hessian
                MakeHessian();
                // TODO:: 这个判断条件可以丢掉，条件 b_max <= 1e-12 很难达到，这里的阈值条件不应该用绝对值，而是相对值
//                double b_max = 0.0;
//                for (int i = 0; i < b_.size(); ++i) {
//                    b_max = max(fabs(b_(i)), b_max);
//                }
//                // 优化退出条件2： 如果残差 b_max 已经很小了，那就退出
//                stop = (b_max <= 1e-12);
                false_cnt = 0;
            } else {
                false_cnt ++;
                RollbackStates();   // 误差没下降，回滚
            }
        }
        iter++;

        // 优化退出条件3： currentChi_ 跟第一次的chi2相比，下降了 1e6 倍则退出
        if (sqrt(currentChi_) <= stopThresholdLM_)
            stop = true;
    }
    std::cout << "problem solve cost: " << t_solve.toc() << " ms" << std::endl;
    std::cout << "   makeHessian cost: " << t_hessian_cost_ << " ms" << std::endl;
    return true;
}


// 统计优化变量的维数，为构建 H 矩阵做准备
void Problem::SetOrdering() {

    // 每次重新计数
    ordering_poses_ = 0;
    ordering_generic_ = 0;      // 是所有优化变量的总维度，也是`H`矩阵的维度！
    ordering_landmarks_ = 0;
    int debug = 0;


    // `LocalDimension()`是优化变量要优化的维度，比如位姿，如果用四元数表示，输入参数维度是7，但实际优化维度是6！
    // Note:: verticies_ 是 map 类型的, 顺序是按照 id 号排序的
    for (auto vertex: verticies_) {
        ordering_generic_ += vertex.second->LocalDimension();  // 所有的优化变量总维数

        if (IsPoseVertex(vertex.second)) {
            debug += vertex.second->LocalDimension();
        }

        if (problemType_ == ProblemType::SLAM_PROBLEM)    // 如果是 slam 问题，还要分别统计 pose 和 landmark 的维数，后面会对他们进行排序
        {
            AddOrderingSLAM(vertex.second);
        }

        if (IsPoseVertex(vertex.second)) {
            //  cout<< 位姿顶点id  << order: <<  H矩阵中顶点对应的位置
            std::cout << vertex.second->Id() << " order: " << vertex.second->OrderingId() << std::endl;
        }
    }

    // 路标点个数
    std::cout << "\n ordered_landmark_vertices_ size : " << idx_landmark_vertices_.size() << std::endl;
    if (problemType_ == ProblemType::SLAM_PROBLEM) {
        // 这里要把 landmark 的 ordering 加上 pose 的数量，就保持了 landmark 在后,而 pose 在前
        ulong all_pose_dimension = ordering_poses_;
        for (auto landmarkVertex : idx_landmark_vertices_) {
            landmarkVertex.second->SetOrderingId(
                landmarkVertex.second->OrderingId() + all_pose_dimension
            );
        }
    }

//    CHECK_EQ(CheckOrdering(), true);
}

bool Problem::CheckOrdering() {
    if (problemType_ == ProblemType::SLAM_PROBLEM) {
        int current_ordering = 0;
        for (auto v: idx_pose_vertices_) {
            assert(v.second->OrderingId() == current_ordering);
            current_ordering += v.second->LocalDimension();
        }

        for (auto v: idx_landmark_vertices_) {
            assert(v.second->OrderingId() == current_ordering);
            current_ordering += v.second->LocalDimension();
        }
    }
    return true;
}

void Problem::MakeHessian() {
    TicToc t_h;
    // 直接构造大的 H 矩阵
    ulong size = ordering_generic_;             // 6n
    MatXX H(MatXX::Zero(size, size));           // 6n*6n
    VecX b(VecX::Zero(size));                   // 6n*1

    for (auto &edge: edges_) {  // 对于每一个误差边来讲

        edge.second->ComputeResidual();
        edge.second->ComputeJacobians();

        auto jacobians = edge.second->Jacobians();      // std::vector<MatXX> 分别对路标点雅可比2*3、对位姿雅可比2*6
        auto verticies = edge.second->Verticies();      // 一条误差边肯定对应两个顶点

        // 因为对一个优化变量的雅可比就是一个矩阵了，所以对所有优化变量的雅可比用vector来统计
        assert(jacobians.size() == verticies.size());   // std::vector<MatXX>大小应该是与顶点数相同的
        // cout << verticies.size() << endl;
        
        #pragma omp parallel for
        for (size_t i = 0; i < verticies.size(); ++i) {
            auto v_i = verticies[i];            // edge对应第一个顶点是路标点，第二个是位姿
            if (v_i->IsFixed()) continue;    // Hessian 里不需要添加它的信息，也就是它的雅克比为 0

            auto jacobian_i = jacobians[i];
            ulong index_i = v_i->OrderingId();
            ulong dim_i = v_i->LocalDimension();
            // int m = jacobian_i.rows();  // jacobian_i 的行数    2
            // int n = jacobian_i.cols();  // jacobian_i 的列数    3
            // int p = edge.second->Information().rows();  // Information 矩阵的行数   2
            // int q = edge.second->Information().cols();  // Information 矩阵的列数   2

            MatXX JtW = jacobian_i.transpose() * edge.second->Information();    // 3*2
            for (size_t j = i; j < verticies.size(); ++j) {
                auto v_j = verticies[j];

                if (v_j->IsFixed()) continue;

                auto jacobian_j = jacobians[j];     // 2*3 or 2*6
                // cout << jacobian_j.rows() << endl;

                ulong index_j = v_j->OrderingId();
                ulong dim_j = v_j->LocalDimension();

                assert(v_j->OrderingId() != -1);
                MatXX hessian = JtW * jacobian_j;       // 3*3 or 3*6 or 6*6
                // 所有的信息矩阵叠加起来
                H.block(index_i,index_j, dim_i, dim_j).noalias() += hessian;

                // 对称矩阵，我们从j=i开始遍历，实际上只能遍历矩阵的一半，同时利用对称矩阵的性质，构建对称部分！
                if (j != i) {   // 对称的下三角
                    H.block(index_j,index_i, dim_j, dim_i).noalias() += hessian.transpose();
                }
            }
            b.segment(index_i, dim_i).noalias() -= JtW * edge.second->Residual();
        }

    }
    Hessian_ = H;
    b_ = b;
    t_hessian_cost_ += t_h.toc();


//    Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeThinU | Eigen::ComputeThinV);
//    std::cout << svd.singularValues() <<std::endl;

    if (err_prior_.rows() > 0) {
        b_prior_ -= H_prior_ * delta_x_.head(ordering_poses_);   // update the error_prior
    }
    Hessian_.topLeftCorner(ordering_poses_, ordering_poses_) += H_prior_;
    b_.head(ordering_poses_) += b_prior_;

    delta_x_ = VecX::Zero(size);  // initial delta_x = 0_n;

}

/*
 * Solve Hx = b, we can use PCG iterative method or use sparse Cholesky
 */
void Problem::SolveLinearSystem() {
    /********************************************/
    // typedef g2o::BlockSolver_6_3 BlockSolverType;
    // typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType>
    //     LinearSolverType;
    // auto solver = new g2o::OptimizationAlgorithmLevenberg(
    //     g2o::make_unique<BlockSolverType>(
    //         g2o::make_unique<LinearSolverType>()));

    
    // MatXX H = Hessian_;
    // for (ulong i = 0; i < Hessian_.cols(); ++i) {
    //     H(i, i) += currentLambda_;
    // }
    // delta_x_ = H.ldlt().solve(b_);   
        
    /********************共轭梯度法*******************************/
        MatXX H = Hessian_;
        for (ulong i = 0; i < Hessian_.cols(); ++i) {
            H(i, i) += currentLambda_;
        }

        // 将稠密矩阵转换为稀疏矩阵
        Eigen::SparseMatrix<double> H_sparse = H.sparseView();

        // 使用共轭梯度法求解线性方程
        Eigen::ConjugateGradient<Eigen::SparseMatrix<double> > cg;
        cg.compute(H_sparse);
        delta_x_ = cg.solve(b_);


    /*******************ldlt()*****************************/
    // MatXX H = Hessian_;
    // for (ulong i = 0; i < Hessian_.cols(); ++i) {
    //     H(i, i) += currentLambda_;
    // }
    // delta_x_ = H.ldlt().solve(b_);

    /**************************************************************/
    // if (problemType_ == ProblemType::GENERIC_PROBLEM) {
    //     // 非 SLAM 问题直接求解
    //     // 使用Cholesky分解进行求解
    //     MatXX H = Hessian_;
    //     for (ulong i = 0; i < Hessian_.cols(); ++i) {
    //         H(i, i) += currentLambda_;
    //     }
    //     delta_x_ = H.ldlt().solve(b_);
    // } else {
    //     // SLAM 问题采用舒尔补的计算方式
    //     // step1: schur marginalization --> Hpp, bpp
    //     int reserve_size = ordering_poses_;
    //     int marg_size = ordering_landmarks_;

    //     MatXX Hmm = Hessian_.block(reserve_size,reserve_size, marg_size, marg_size);
    //     MatXX Hpm = Hessian_.block(0,reserve_size, reserve_size, marg_size);
    //     MatXX Hmp = Hessian_.block(reserve_size,0, marg_size, reserve_size);
        
    //     VecX bpp = b_.segment(0,reserve_size);
    //     VecX bmm = b_.segment(reserve_size,marg_size);

    //     // 使用Cholesky分解对Hmm进行求解
    //     MatXX Hmm_inv = Hmm.llt().solve(MatXX::Identity(marg_size, marg_size));

    //     MatXX tempH = Hpm * Hmm_inv;
    //     H_pp_schur_ = Hessian_.block(0,0,reserve_size,reserve_size) - tempH * Hmp;
    //     b_pp_schur_ = bpp - tempH * bmm;

    //     // step2: solve Hpp * delta_x = bpp
    //     VecX delta_x_pp = H_pp_schur_.ldlt().solve(b_pp_schur_);
    //     delta_x_.head(reserve_size) = delta_x_pp;

    //     // step3: solve landmark
    //     VecX delta_x_ll = Hmm_inv * (bmm - Hmp * delta_x_pp);
    //     delta_x_.tail(marg_size) = delta_x_ll;
    // }
}



// void Problem::SolveLinearSystem() {

//     if (problemType_ == ProblemType::GENERIC_PROBLEM) {

//         // 非 SLAM 问题直接求解
//         // PCG solver
//         MatXX H = Hessian_;
//         for (ulong i = 0; i < Hessian_.cols(); ++i) {
//             H(i, i) += currentLambda_;
//         }
// //        delta_x_ = PCGSolver(H, b_, H.rows() * 2);
//         delta_x_ = Hessian_.inverse() * b_;

//     } else {

//         // SLAM 问题采用舒尔补的计算方式
//         // step1: schur marginalization --> Hpp, bpp
//         int reserve_size = ordering_poses_;
//         int marg_size = ordering_landmarks_;

//         MatXX Hmm = Hessian_.block(reserve_size,reserve_size, marg_size, marg_size);
//         MatXX Hpm = Hessian_.block(0,reserve_size, reserve_size, marg_size);
//         MatXX Hmp = Hessian_.block(reserve_size,0, marg_size, reserve_size);
        
//         // 注意这不是行列索引，对于列向量，segment这里指的是起始值
//         VecX bpp = b_.segment(0,reserve_size);
//         VecX bmm = b_.segment(reserve_size,marg_size);

//         // Hmm 是对角线矩阵，它的求逆可以直接为对角线块分别求逆，如果是逆深度，对角线块为1维的，则直接为对角线的倒数，这里可以加速
//         MatXX Hmm_inv(MatXX::Zero(marg_size, marg_size));
//         for (auto landmarkVertex : idx_landmark_vertices_) {
//             int idx = landmarkVertex.second->OrderingId() - reserve_size;
//             int size = landmarkVertex.second->LocalDimension();     // 3*3
//             Hmm_inv.block(idx, idx, size, size) = Hmm.block(idx, idx, size, size).inverse();
//         }

//         MatXX tempH = Hpm * Hmm_inv;
//         H_pp_schur_ = Hessian_.block(0,0,reserve_size,reserve_size) - tempH * Hmp;
//         b_pp_schur_ = bpp - tempH * bmm;


//         // step2: solve Hpp * delta_x = bpp
//         VecX delta_x_pp(VecX::Zero(reserve_size));
//         // PCG Solver
//         for (ulong i = 0; i < ordering_poses_; ++i) {
//             H_pp_schur_(i, i) += currentLambda_;
//         }

//         int n = H_pp_schur_.rows() * 2;                       // 迭代次数
//         delta_x_pp = PCGSolver(H_pp_schur_, b_pp_schur_, n);  
//         delta_x_.head(reserve_size) = delta_x_pp;
//         //        std::cout << delta_x_pp.transpose() << std::endl;

//         // step3: solve landmark
//         VecX delta_x_ll(marg_size);
//         delta_x_ll = Hmm_inv * (bmm - Hmp * delta_x_pp);
//         delta_x_.tail(marg_size) = delta_x_ll;

//     }

// }

void Problem::UpdateStates() {
    for (auto vertex: verticies_) {
        ulong idx = vertex.second->OrderingId();
        ulong dim = vertex.second->LocalDimension();
        VecX delta = delta_x_.segment(idx, dim);
        vertex.second->Plus(delta);
    }
    if (err_prior_.rows() > 0) {
        b_prior_ -= H_prior_ * delta_x_.head(ordering_poses_);   // update the error_prior
        err_prior_ = Jt_prior_inv_ * b_prior_.head(ordering_poses_ - 6);
    }

}

void Problem::RollbackStates() {
    for (auto vertex: verticies_) {
        ulong idx = vertex.second->OrderingId();
        ulong dim = vertex.second->LocalDimension();
        VecX delta = delta_x_.segment(idx, dim);
        vertex.second->Plus(-delta);
    }
    if (err_prior_.rows() > 0) {
        b_prior_ += H_prior_ * delta_x_.head(ordering_poses_);   // update the error_prior
        err_prior_ = Jt_prior_inv_ * b_prior_.head(ordering_poses_ - 6);
    }
}

/// LM
void Problem::ComputeLambdaInitLM() {
    ni_ = 2.;
    currentLambda_ = -1.;
    currentChi_ = 0.0;
    // TODO:: robust cost chi2
    for (auto edge: edges_) {
        currentChi_ += edge.second->Chi2();
    }
    if (err_prior_.rows() > 0)      // marg prior residual
        currentChi_ += err_prior_.norm();

    stopThresholdLM_ = 1e-6 * currentChi_;          // 迭代条件为 误差下降 1e-6 倍

    double maxDiagonal = 0;
    ulong size = Hessian_.cols();
    assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");

    for (ulong i = 0; i < size; ++i) {
        maxDiagonal = std::max(fabs(Hessian_(i, i)), maxDiagonal);
    }
    double tau = 1e-5;
    currentLambda_ = tau * maxDiagonal;
}

void Problem::AddLambdatoHessianLM() {
    ulong size = Hessian_.cols();
    assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
    for (ulong i = 0; i < size; ++i) {
        Hessian_(i, i) += currentLambda_;
    }
}

void Problem::RemoveLambdaHessianLM() {
    ulong size = Hessian_.cols();
    assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
    // TODO:: 这里不应该减去一个，数值的反复加减容易造成数值精度出问题？而应该保存叠加lambda前的值，在这里直接赋值
    for (ulong i = 0; i < size; ++i) {
        Hessian_(i, i) -= currentLambda_;
    }
}

bool Problem::IsGoodStepInLM() {
    double scale = 0;
    scale = delta_x_.transpose() * (currentLambda_ * delta_x_ + b_);
    scale += 1e-3;    // make sure it's non-zero :)

    // recompute residuals after update state
    // TODO:: get robustChi2() instead of Chi2()
    double tempChi = 0.0;
    for (auto edge: edges_) {
        edge.second->ComputeResidual();
        tempChi += edge.second->Chi2();
    }
    if (err_prior_.size() > 0)
        tempChi += err_prior_.norm();

    double rho = (currentChi_ - tempChi) / scale;
    if (rho > 0 && isfinite(tempChi))   // last step was good, 误差在下降
    {
        double alpha = 1. - pow((2 * rho - 1), 3);
        alpha = std::min(alpha, 2. / 3.);
        double scaleFactor = (std::max)(1. / 3., alpha);
        currentLambda_ *= scaleFactor;
        ni_ = 2;
        currentChi_ = tempChi;
        return true;
    } else {
        currentLambda_ *= ni_;
        ni_ *= 2;
        return false;
    }
}

/** @brief conjugate gradient with perconditioning
 *
 *  the jacobi PCG method
 *
 */
VecX Problem::PCGSolver(const MatXX &A, const VecX &b, int maxIter = -1) {
    assert(A.rows() == A.cols() && "PCG solver ERROR: A is not a square matrix");
    int rows = b.rows();
    int n = maxIter < 0 ? rows : maxIter;
    VecX x(VecX::Zero(rows));
    MatXX M_inv = A.diagonal().asDiagonal().inverse();
    VecX r0(b);  // initial r = b - A*0 = b
    VecX z0 = M_inv * r0;
    VecX p(z0);
    VecX w = A * p;
    double r0z0 = r0.dot(z0);
    double alpha = r0z0 / p.dot(w);
    VecX r1 = r0 - alpha * w;
    int i = 0;
    double threshold = 1e-6 * r0.norm();
    while (r1.norm() > threshold && i < n) {
        i++;
        VecX z1 = M_inv * r1;
        double r1z1 = r1.dot(z1);
        double belta = r1z1 / r0z0;
        z0 = z1;
        r0z0 = r1z1;
        r0 = r1;
        p = belta * p + z1;
        w = A * p;
        alpha = r1z1 / p.dot(w);
        x += alpha * p;
        r1 -= alpha * w;
    }
    return x;
}


}





