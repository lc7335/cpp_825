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

    // ��ʼ���������
    Gsk = MatrixXd::Zero(Ncell_, n_dp);
    Gs = VectorXd::Zero(Ncell_);

    // ����������� Gsk
    for (int i = 0; i < Ncell_; ++i) {
        for (int j = 0; j < n_dp; ++j) {
            if (i == 0) {
                // ��һ��С��
                Gsk(i, j) = G_downk(i, j) + Gfa(i, j) + Gre(i, j) - G_outk(j)
                    - Gd(i, j) - (M_km(i, j) - M_km0(i, j)) / dt_;
            }
            else if (i < (Ncell_ - 1)) {
                // �м�С��
                Gsk(i, j) = Gsk(i - 1, j) + G_downk(i, j) + Gfa(i, j) + Gre(i, j)
                    - G_upk(i, j) - Gd(i, j) - (M_km(i, j) - M_km0(i, j)) / dt_;
            }
            else {
                // �ײ�С�ҳ�����Ϊ0
                Gsk(i, j) = 0.0;
            }
        }
    }

    // ����ÿ��С�ҵ��ܳ�����
    for (int i = 0; i < Ncell_; ++i) {
        Gs(i) = Gsk.row(i).sum();
    }

    // �������˵��µ�̼������ת�� G_tck
    MatrixXd G_tck = MatrixXd::Zero(Ncell_, n_dp);
    for (int j = 1; j < n_dp; ++j) {
        for (int i = 0; i < Ncell_; ++i) {
            G_tck(i, j) = (1.0 / 3.0) * dp[j] / (dp[j] - dp[j - 1]) * Rc(i, j);
        }
    }

    // ���������̼��������������ϵͳ
    auto [A_c, b_c] = buildLinearSystemXc(dp, M_km, M_km0, G_upk, G_downk, Gfa, Gre, Gd,
        G_outk, Rc, Xfac_km, Xc_km0, Gsk, Gcyc_downk,
        Gcyc_flyk, M_cyck, M_cyck0, W_rek, M_lsk, M_lsk0, G_tck);

    VectorXd Xc_km_flat = safeSolve(A_c, b_c);

    // ��������ת��Ϊ������ʽ
    Xc_km = MatrixXd::Zero(Ncell_ + 2, n_dp);
    for (int i = 0; i < Ncell_ + 2; ++i) {
        for (int j = 0; j < n_dp; ++j) {
            Xc_km(i, j) = Xc_km_flat(i * n_dp + j);
        }
    }

    // �����Ϲ������У�����װ����1�������������2
    Xc_km = rollMatrix(Xc_km, 2);

    // �����������������������������ϵͳ
    auto [A_ca, b_ca] = buildLinearSystemXca(dp, M_km, M_km0, G_upk, G_downk, Gfa, Gre, Gd,
        G_outk, Rca, Xfaca_km, Xca_km0, Gsk, Gcyc_downk,
        Gcyc_flyk, M_cyck, M_cyck0, W_rek, M_lsk, M_lsk0);

    VectorXd Xca_km_flat = safeSolve(A_ca, b_ca);

    // ��������ת��Ϊ������ʽ
    Xca_km = MatrixXd::Zero(Ncell_ + 2, n_dp);
    for (int i = 0; i < Ncell_ + 2; ++i) {
        for (int j = 0; j < n_dp; ++j) {
            Xca_km(i, j) = Xca_km_flat(i * n_dp + j);
        }
    }

    // �����Ϲ�������
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
    int N_i = Ncell_ + 2;  // С������
    int N_total = N_i * N_j;  // �ܱ�����

    std::vector<Triplet> triplets;
    VectorXd b = VectorXd::Zero(N_total);

    int row = 0;

    // ��ȼ��С�ҷ���
    for (int i = 0; i < N_i; ++i) {
        for (int j = 0; j < N_j; ++j) {
            int idx = i * N_j + j;

            if (i == 0) {
                // ��һ��С��
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
                // ���һ��С��
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
                // ����װ��
                triplets.emplace_back(row, (Ncell_ + 1) * N_j + j, Gcyc_downk(j));
                triplets.emplace_back(row, idx, -(W_rek(j) + M_lsk(j) / dt_));
                b(row) = -M_lsk0(j) * Xc_km0(Ncell_, j) / dt_;

            }
            else if (i == Ncell_ + 1) {
                // ���������
                triplets.emplace_back(row, 0 * N_j + j, G_outk(j));
                triplets.emplace_back(row, idx, -(Gcyc_flyk(j) + Gcyc_downk(j) + M_cyck(j) / dt_));
                b(row) = -M_cyck0(j) * Xc_km0(Ncell_ + 1, j) / dt_;

            }
            else {
                // �м�С��
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
    int N_i = Ncell_ + 2;  // С������
    int N_total = N_i * N_j;  // �ܱ�����

    std::vector<Triplet> triplets;
    VectorXd b = VectorXd::Zero(N_total);

    int row = 0;

    // ��ȼ��С�ҷ���
    for (int i = 0; i < N_i; ++i) {
        for (int j = 0; j < N_j; ++j) {
            int idx = i * N_j + j;

            if (i == 0) {
                // ��һ��С��
                triplets.emplace_back(row, (i + 1) * N_j + j, G_downk(i, j));
                triplets.emplace_back(row, idx, -G_outk(j) - Gsk(i, j) - Gd(i, j) - M_km(i, j) / dt_);
                triplets.emplace_back(row, Ncell_ * N_j + j, Gre(i, j));
                b(row) = -(Gfa(i, j) * Xfaca_km(i, j) - Rca(i, j) + M_km0(i, j) * Xca_km0(i, j) / dt_);

            }
            else if (i == Ncell_ - 1) {
                // ���һ��С��
                triplets.emplace_back(row, (i - 1) * N_j + j, Gsk(i - 1, j));
                triplets.emplace_back(row, idx, -G_upk(i, j) - Gd(i, j) - M_km(i, j) / dt_);
                triplets.emplace_back(row, Ncell_ * N_j + j, Gre(i, j));
                b(row) = -(Gfa(i, j) * Xfaca_km(i, j) - Rca(i, j) + M_km0(i, j) * Xca_km0(i, j) / dt_);

            }
            else if (i == Ncell_) {
                // ����װ��
                triplets.emplace_back(row, (Ncell_ + 1) * N_j + j, Gcyc_downk(j));
                triplets.emplace_back(row, idx, -(W_rek(j) + M_lsk(j) / dt_));
                b(row) = -M_lsk0(j) * Xca_km0(Ncell_, j) / dt_;

            }
            else if (i == Ncell_ + 1) {
                // ���������
                triplets.emplace_back(row, 0 * N_j + j, G_outk(j));
                triplets.emplace_back(row, idx, -(Gcyc_flyk(j) + Gcyc_downk(j) + M_cyck(j) / dt_));
                b(row) = -M_cyck0(j) * Xca_km0(Ncell_ + 1, j) / dt_;

            }
            else {
                // �м�С��
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
        // ���ȳ���ʹ��SparseLU�����
        Eigen::SparseLU<SparseMatrix> solver;
        solver.compute(A);

        if (solver.info() != Eigen::Success) {
            std::cout << "���棺LU�ֽ�ʧ�ܣ�����ʹ��QR�ֽ�..." << std::endl;

            // ���˵�SparseQR�����
            Eigen::SparseQR<SparseMatrix, Eigen::COLAMDOrdering<int>> qr_solver;
            qr_solver.compute(A);

            if (qr_solver.info() != Eigen::Success) {
                std::cout << "���棺QR�ֽ�Ҳʧ�ܣ�ʹ�õ��������..." << std::endl;

                // ���ջ��˵����������
                Eigen::BiCGSTAB<SparseMatrix> iterative_solver;
                iterative_solver.compute(A);
                VectorXd x = iterative_solver.solve(b);

                // ȷ�������������Χ�� [0,1]
                x = x.cwiseMax(0.0).cwiseMin(1.0);
                return x;
            }

            VectorXd x = qr_solver.solve(b);
            x = x.cwiseMax(0.0).cwiseMin(1.0);
            return x;
        }

        VectorXd x = solver.solve(b);

        // ȷ�������������Χ�� [0,1]
        x = x.cwiseMax(0.0).cwiseMin(1.0);
        return x;

    }
    catch (const std::exception& e) {
        std::cout << "���ʧ��: " << e.what() << std::endl;
        // ������������Ϊ����
        return VectorXd::Zero(b.size());
    }
}

void SolidMass::cleanNonFiniteValues(SparseMatrix& A, VectorXd& b) const {
    // ����b�����еķ�����ֵ
    for (int i = 0; i < b.size(); ++i) {
        if (!std::isfinite(b(i))) {
            b(i) = 0.0;
        }
    }

    // ����ϡ�����A�еķ�����ֵ
    for (int k = 0; k < A.outerSize(); ++k) {
        for (SparseMatrix::InnerIterator it(A, k); it; ++it) {
            if (!std::isfinite(it.value())) {
                // ע�⣺���ﲻ��ֱ���޸ģ���Ҫ�ؽ�����
                std::cout << "���־����еķ�����ֵ����Ҫ�ؽ�����" << std::endl;
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