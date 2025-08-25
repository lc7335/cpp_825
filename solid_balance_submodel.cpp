#include "solid_balance_submodel.hpp"
#include <Eigen/SparseLU>
#include <Eigen/SparseQR>
#include <Eigen/IterativeLinearSolvers>
#include <limits>
#include <stdexcept>

SolidMass::SolidMass(int Ncell, const std::vector<double>& Vcell, double rho_p, double dt)
    : Ncell_(Ncell), Vcell_(Vcell), rho_p_(rho_p), dt_(dt) {
    if (Ncell <= 0 || dt <= 0 || rho_p <= 0) {
        throw std::invalid_argument("Invalid parameters: Ncell, dt, and rho_p must be positive");
    }
    if (static_cast<int>(Vcell.size()) != Ncell) {
        throw std::invalid_argument("Vcell size must match Ncell");
    }
}

void SolidMass::solidMass(
    const std::vector<double>& dp,
    const MatrixXd& M_km,
    const MatrixXd& M_km0,
    const MatrixXd& G_upk,
    const MatrixXd& G_downk,
    const MatrixXd& Gfa,
    const MatrixXd& Gre,
    const MatrixXd& Gd,
    const VectorXd& G_outk,
    const MatrixXd& Rc,
    const MatrixXd& Xfac_km,
    const MatrixXd& Xc_km0,
    const MatrixXd& Rca,
    const MatrixXd& Xfaca_km,
    const MatrixXd& Xca_km0,
    const VectorXd& Gcyc_downk,
    const VectorXd& Gcyc_flyk,
    const VectorXd& M_cyck,
    const VectorXd& M_cyck0,
    const VectorXd& W_rek,
    const VectorXd& M_lsk,
    const VectorXd& M_lsk0,
    MatrixXd& Xc_km,
    MatrixXd& Xca_km,
    VectorXd& Gs,
    MatrixXd& Gsk) const {

    int n_dp = static_cast<int>(dp.size());

    // 初始化输出矩阵
    Gsk = MatrixXd::Zero(Ncell_, n_dp);
    Gs = VectorXd::Zero(Ncell_);

    // 计算沉降流率 Gsk
    for (int i = 0; i < Ncell_; ++i) {
        for (int j = 0; j < n_dp; ++j) {
            if (i == 0) {
                // 第一个小室
                Gsk(i, j) = G_downk(i, j) + Gfa(i, j) + Gre(i, j) - G_outk(j)
                    - Gd(i, j) - (M_km(i, j) - M_km0(i, j)) / dt_;
            }
            else if (i < (Ncell_ - 1)) {
                // 中间小室
                Gsk(i, j) = Gsk(i - 1, j) + G_downk(i, j) + Gfa(i, j) + Gre(i, j)
                    - G_upk(i, j) - Gd(i, j) - (M_km(i, j) - M_km0(i, j)) / dt_;
            }
            else {
                // 底部小室沉降量为0
                Gsk(i, j) = 0.0;
            }
        }
    }

    // 计算每个小室的总沉降量
    for (int i = 0; i < Ncell_; ++i) {
        Gs(i) = Gsk.row(i).sum();
    }

    // 计算缩核导致的碳颗粒档转移 G_tck
    MatrixXd G_tck = MatrixXd::Zero(Ncell_, n_dp);
    for (int j = 1; j < n_dp; ++j) {
        for (int i = 0; i < Ncell_; ++i) {
            G_tck(i, j) = (1.0 / 3.0) * dp[j] / (dp[j] - dp[j - 1]) * Rc(i, j);
        }
    }

    // 构建并求解碳质量分数的线性系统
    auto [A_c, b_c] = buildLinearSystemXc(dp, M_km, M_km0, G_upk, G_downk, Gfa, Gre, Gd,
        G_outk, Rc, Xfac_km, Xc_km0, Gsk, Gcyc_downk,
        Gcyc_flyk, M_cyck, M_cyck0, W_rek, M_lsk, M_lsk0, G_tck);

    VectorXd Xc_km_flat = safeSolve(A_c, b_c);

    // 将解向量转换为矩阵形式
    Xc_km = MatrixXd::Zero(Ncell_ + 2, n_dp);
    for (int i = 0; i < Ncell_ + 2; ++i) {
        for (int j = 0; j < n_dp; ++j) {
            Xc_km(i, j) = Xc_km_flat(i * n_dp + j);
        }
    }

    // 行向上滚动两行，返料装置是1，旋风分离器是2
    Xc_km = rollMatrix(Xc_km, 2);

    // 构建并求解氧化钙质量分数的线性系统
    auto [A_ca, b_ca] = buildLinearSystemXca(dp, M_km, M_km0, G_upk, G_downk, Gfa, Gre, Gd,
        G_outk, Rca, Xfaca_km, Xca_km0, Gsk, Gcyc_downk,
        Gcyc_flyk, M_cyck, M_cyck0, W_rek, M_lsk, M_lsk0);

    VectorXd Xca_km_flat = safeSolve(A_ca, b_ca);

    // 将解向量转换为矩阵形式
    Xca_km = MatrixXd::Zero(Ncell_ + 2, n_dp);
    for (int i = 0; i < Ncell_ + 2; ++i) {
        for (int j = 0; j < n_dp; ++j) {
            Xca_km(i, j) = Xca_km_flat(i * n_dp + j);
        }
    }

    // 行向上滚动两行
    Xca_km = rollMatrix(Xca_km, 2);
}

std::pair<SolidMass::SparseMatrix, SolidMass::VectorXd> SolidMass::buildLinearSystemXc(
    const std::vector<double>& dp,
    const MatrixXd& M_km,
    const MatrixXd& M_km0,
    const MatrixXd& G_upk,
    const MatrixXd& G_downk,
    const MatrixXd& Gfa,
    const MatrixXd& Gre,
    const MatrixXd& Gd,
    const VectorXd& G_outk,
    const MatrixXd& Rc,
    const MatrixXd& Xfac_km,
    const MatrixXd& Xc_km0,
    const MatrixXd& Gsk,
    const VectorXd& Gcyc_downk,
    const VectorXd& Gcyc_flyk,
    const VectorXd& M_cyck,
    const VectorXd& M_cyck0,
    const VectorXd& W_rek,
    const VectorXd& M_lsk,
    const VectorXd& M_lsk0,
    const MatrixXd& G_tck) const {

    int N_j = static_cast<int>(dp.size());
    int N_i = Ncell_ + 2;  // 小室总数
    int N_total = N_i * N_j;  // 总变量数

    std::vector<Triplet> triplets;
    VectorXd b = VectorXd::Zero(N_total);

    int row = 0;

    // 主燃室小室方程
    for (int i = 0; i < N_i; ++i) {
        for (int j = 0; j < N_j; ++j) {
            int idx = i * N_j + j;

            if (i == 0) {
                // 第一个小室
                triplets.emplace_back(row, (i + 1) * N_j + j, G_downk(i, j));
                triplets.emplace_back(row, idx, -G_outk(j) - Gsk(i, j) - Gd(i, j) - M_km(i, j) / dt_);
                triplets.emplace_back(row, Ncell_ * N_j + j, Gre(i, j));

                if (j == N_j - 1) {
                    b(row) = -(Gfa(i, j) * Xfac_km(i, j) - G_tck(i, j) - Rc(i, j) +
                        M_km0(i, j) * Xc_km0(i, j) / dt_);
                }
                else {
                    b(row) = -(Gfa(i, j) * Xfac_km(i, j) + G_tck(i, j + 1) - G_tck(i, j) - Rc(i, j) +
                        M_km0(i, j) * Xc_km0(i, j) / dt_);
                }

            }
            else if (i == Ncell_ - 1) {
                // 最后一个小室
                triplets.emplace_back(row, (i - 1) * N_j + j, Gsk(i - 1, j));
                triplets.emplace_back(row, idx, -G_upk(i, j) - Gd(i, j) - M_km(i, j) / dt_);
                triplets.emplace_back(row, Ncell_ * N_j + j, Gre(i, j));

                if (j == N_j - 1) {
                    b(row) = -(Gfa(i, j) * Xfac_km(i, j) - G_tck(i, j) - Rc(i, j) +
                        M_km0(i, j) * Xc_km0(i, j) / dt_);
                }
                else {
                    b(row) = -(Gfa(i, j) * Xfac_km(i, j) + G_tck(i, j + 1) - G_tck(i, j) - Rc(i, j) +
                        M_km0(i, j) * Xc_km0(i, j) / dt_);
                }

            }
            else if (i == Ncell_) {
                // 返料装置
                triplets.emplace_back(row, (Ncell_ + 1) * N_j + j, Gcyc_downk(j));
                triplets.emplace_back(row, idx, -(W_rek(j) + M_lsk(j) / dt_));
                b(row) = -M_lsk0(j) * Xc_km0(Ncell_, j) / dt_;

            }
            else if (i == Ncell_ + 1) {
                // 旋风分离器
                triplets.emplace_back(row, 0 * N_j + j, G_outk(j));
                triplets.emplace_back(row, idx, -(Gcyc_flyk(j) + Gcyc_downk(j) + M_cyck(j) / dt_));
                b(row) = -M_cyck0(j) * Xc_km0(Ncell_ + 1, j) / dt_;

            }
            else {
                // 中间小室
                triplets.emplace_back(row, (i - 1) * N_j + j, Gsk(i - 1, j));
                triplets.emplace_back(row, (i + 1) * N_j + j, G_downk(i, j));
                triplets.emplace_back(row, idx, -G_upk(i, j) - Gsk(i, j) - Gd(i, j) - M_km(i, j) / dt_);
                triplets.emplace_back(row, Ncell_ * N_j + j, Gre(i, j));

                if (j == N_j - 1) {
                    b(row) = -(Gfa(i, j) * Xfac_km(i, j) - G_tck(i, j) - Rc(i, j) +
                        M_km0(i, j) * Xc_km0(i, j) / dt_);
                }
                else {
                    b(row) = -(Gfa(i, j) * Xfac_km(i, j) + G_tck(i, j + 1) - G_tck(i, j) - Rc(i, j) +
                        M_km0(i, j) * Xc_km0(i, j) / dt_);
                }
            }

            row++;
        }
    }

    SparseMatrix A(N_total, N_total);
    A.setFromTriplets(triplets.begin(), triplets.end());

    return std::make_pair(A, b);
}

std::pair<SolidMass::SparseMatrix, SolidMass::VectorXd> SolidMass::buildLinearSystemXca(
    const std::vector<double>& dp,
    const MatrixXd& M_km,
    const MatrixXd& M_km0,
    const MatrixXd& G_upk,
    const MatrixXd& G_downk,
    const MatrixXd& Gfa,
    const MatrixXd& Gre,
    const MatrixXd& Gd,
    const VectorXd& G_outk,
    const MatrixXd& Rca,
    const MatrixXd& Xfaca_km,
    const MatrixXd& Xca_km0,
    const MatrixXd& Gsk,
    const VectorXd& Gcyc_downk,
    const VectorXd& Gcyc_flyk,
    const VectorXd& M_cyck,
    const VectorXd& M_cyck0,
    const VectorXd& W_rek,
    const VectorXd& M_lsk,
    const VectorXd& M_lsk0) const {

    int N_j = static_cast<int>(dp.size());
    int N_i = Ncell_ + 2;  // 小室总数
    int N_total = N_i * N_j;  // 总变量数

    std::vector<Triplet> triplets;
    VectorXd b = VectorXd::Zero(N_total);

    int row = 0;

    // 主燃室小室方程
    for (int i = 0; i < N_i; ++i) {
        for (int j = 0; j < N_j; ++j) {
            int idx = i * N_j + j;

            if (i == 0) {
                // 第一个小室
                triplets.emplace_back(row, (i + 1) * N_j + j, G_downk(i, j));
                triplets.emplace_back(row, idx, -G_outk(j) - Gsk(i, j) - Gd(i, j) - M_km(i, j) / dt_);
                triplets.emplace_back(row, Ncell_ * N_j + j, Gre(i, j));
                b(row) = -(Gfa(i, j) * Xfaca_km(i, j) - Rca(i, j) + M_km0(i, j) * Xca_km0(i, j) / dt_);

            }
            else if (i == Ncell_ - 1) {
                // 最后一个小室
                triplets.emplace_back(row, (i - 1) * N_j + j, Gsk(i - 1, j));
                triplets.emplace_back(row, idx, -G_upk(i, j) - Gd(i, j) - M_km(i, j) / dt_);
                triplets.emplace_back(row, Ncell_ * N_j + j, Gre(i, j));
                b(row) = -(Gfa(i, j) * Xfaca_km(i, j) - Rca(i, j) + M_km0(i, j) * Xca_km0(i, j) / dt_);

            }
            else if (i == Ncell_) {
                // 返料装置
                triplets.emplace_back(row, (Ncell_ + 1) * N_j + j, Gcyc_downk(j));
                triplets.emplace_back(row, idx, -(W_rek(j) + M_lsk(j) / dt_));
                b(row) = -M_lsk0(j) * Xca_km0(Ncell_, j) / dt_;

            }
            else if (i == Ncell_ + 1) {
                // 旋风分离器
                triplets.emplace_back(row, 0 * N_j + j, G_outk(j));
                triplets.emplace_back(row, idx, -(Gcyc_flyk(j) + Gcyc_downk(j) + M_cyck(j) / dt_));
                b(row) = -M_cyck0(j) * Xca_km0(Ncell_ + 1, j) / dt_;

            }
            else {
                // 中间小室
                triplets.emplace_back(row, (i - 1) * N_j + j, Gsk(i - 1, j));
                triplets.emplace_back(row, (i + 1) * N_j + j, G_downk(i, j));
                triplets.emplace_back(row, idx, -G_upk(i, j) - Gsk(i, j) - Gd(i, j) - M_km(i, j) / dt_);
                triplets.emplace_back(row, Ncell_ * N_j + j, Gre(i, j));
                b(row) = -(Gfa(i, j) * Xfaca_km(i, j) - Rca(i, j) + M_km0(i, j) * Xca_km0(i, j) / dt_);
            }

            row++;
        }
    }

    SparseMatrix A(N_total, N_total);
    A.setFromTriplets(triplets.begin(), triplets.end());

    return std::make_pair(A, b);
}

SolidMass::VectorXd SolidMass::safeSolve(const SparseMatrix& A, const VectorXd& b) const {
    try {
        // 首先尝试使用SparseLU求解器
        Eigen::SparseLU<SparseMatrix> solver;
        solver.compute(A);

        if (solver.info() != Eigen::Success) {
            std::cout << "警告：LU分解失败，尝试使用QR分解..." << std::endl;

            // 回退到SparseQR求解器
            Eigen::SparseQR<SparseMatrix, Eigen::COLAMDOrdering<int>> qr_solver;
            qr_solver.compute(A);

            if (qr_solver.info() != Eigen::Success) {
                std::cout << "警告：QR分解也失败，使用迭代求解器..." << std::endl;

                // 最终回退到迭代求解器
                Eigen::BiCGSTAB<SparseMatrix> iterative_solver;
                iterative_solver.compute(A);
                VectorXd x = iterative_solver.solve(b);

                // 确保解在物理合理范围内 [0,1]
                x = x.cwiseMax(0.0).cwiseMin(1.0);
                return x;
            }

            VectorXd x = qr_solver.solve(b);
            x = x.cwiseMax(0.0).cwiseMin(1.0);
            return x;
        }

        VectorXd x = solver.solve(b);

        // 确保解在物理合理范围内 [0,1]
        x = x.cwiseMax(0.0).cwiseMin(1.0);
        return x;

    }
    catch (const std::exception& e) {
        std::cout << "求解失败: " << e.what() << std::endl;
        // 返回零向量作为回退
        return VectorXd::Zero(b.size());
    }
}

void SolidMass::cleanNonFiniteValues(SparseMatrix& A, VectorXd& b) const {
    // 清理b向量中的非有限值
    for (int i = 0; i < b.size(); ++i) {
        if (!std::isfinite(b(i))) {
            b(i) = 0.0;
        }
    }

    // 清理稀疏矩阵A中的非有限值
    for (int k = 0; k < A.outerSize(); ++k) {
        for (SparseMatrix::InnerIterator it(A, k); it; ++it) {
            if (!std::isfinite(it.value())) {
                // 注意：这里不能直接修改，需要重建矩阵
                std::cout << "发现矩阵中的非有限值，需要重建矩阵" << std::endl;
                break;
            }
        }
    }
}

SolidMass::MatrixXd SolidMass::rollMatrix(const MatrixXd& matrix, int shift) const {
    if (shift == 0) return matrix;

    int rows = static_cast<int>(matrix.rows());
    int cols = static_cast<int>(matrix.cols());

    MatrixXd result(rows, cols);

    for (int i = 0; i < rows; ++i) {
        int new_i = (i + shift) % rows;
        if (new_i < 0) new_i += rows;
        result.row(new_i) = matrix.row(i);
    }

    return result;
}