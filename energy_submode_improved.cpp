#include "energy_submode_improved.hpp"
#include <Eigen/SparseLU>
#include <iostream>

Energy::Energy(int N, int N_cyc, double cp_p)
    : N(N), N_cyc(N_cyc), cp_p(cp_p), cp_g(34.56) {
}

Energy::EnergyResult Energy::energy_conservation(
    const Eigen::VectorXd& M_p0,
    const Eigen::VectorXd& M_g0,
    const Eigen::VectorXd& T0,
    const Eigen::VectorXd& M_p,
    const Eigen::VectorXd& M_g,
    const Eigen::VectorXd& W_p,
    const Eigen::VectorXd& G_d,
    const Eigen::VectorXd& G_fly,
    const Eigen::VectorXd& W_g,
    const Eigen::VectorXd& W_air,
    const Eigen::VectorXd& T_air,
    const Eigen::VectorXd& W_fa,
    const Eigen::VectorXd& T_fa,
    double W_rec_p,
    const Eigen::VectorXd& E_p,
    const Eigen::VectorXd& heat_combus,
    const Eigen::VectorXd& U_trans_w,
    const Eigen::VectorXd& U_trans_t,
    const Eigen::VectorXd& A_w,
    const Eigen::VectorXd& A_t,
    const Eigen::VectorXd& T_w,
    const Eigen::VectorXd& T_t,
    double dt) {

    // ��������ϵͳ
    LinearSystem system = build_linear_system_T(
        M_p0, M_g0, T0, M_p, M_g, W_p, G_d, G_fly, W_g, W_air, T_air,
        W_fa, T_fa, W_rec_p, E_p, heat_combus, U_trans_w, U_trans_t,
        A_w, A_t, T_w, T_t, dt
    );

    // ʹ��ϡ���������� Ax = b
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(system.A);

    if (solver.info() != Eigen::Success) {
        std::cerr << "����ֽ�ʧ��!" << std::endl;
        throw std::runtime_error("Matrix decomposition failed");
    }

    Eigen::VectorXd T = solver.solve(system.b);

    if (solver.info() != Eigen::Success) {
        std::cerr << "���ʧ��!" << std::endl;
        throw std::runtime_error("Solving failed");
    }

    // ���㴫����
    Eigen::VectorXd q_heat_w = U_trans_w.cwiseProduct(A_w).cwiseProduct(T - T_w);
    Eigen::VectorXd q_heat_t = U_trans_t.cwiseProduct(A_t).cwiseProduct(T - T_t);

    EnergyResult result;
    result.T = T;
    result.q_heat_w = q_heat_w;
    result.q_heat_t = q_heat_t;

    return result;
}

Energy::LinearSystem Energy::build_linear_system_T(
    const Eigen::VectorXd& M_p0,
    const Eigen::VectorXd& M_g0,
    const Eigen::VectorXd& T0,
    const Eigen::VectorXd& M_p,
    const Eigen::VectorXd& M_g,
    const Eigen::VectorXd& W_p,
    const Eigen::VectorXd& G_d,
    const Eigen::VectorXd& G_fly,
    const Eigen::VectorXd& W_g,
    const Eigen::VectorXd& W_air,
    const Eigen::VectorXd& T_air,
    const Eigen::VectorXd& W_fa,
    const Eigen::VectorXd& T_fa,
    double W_rec_p,
    const Eigen::VectorXd& E_p,
    const Eigen::VectorXd& heat_combus,
    const Eigen::VectorXd& U_trans_w,
    const Eigen::VectorXd& U_trans_t,
    const Eigen::VectorXd& A_w,
    const Eigen::VectorXd& A_t,
    const Eigen::VectorXd& T_w,
    const Eigen::VectorXd& T_t,
    double dt) {

    // ʹ��triplet��ʽ����ϡ�����
    std::vector<Eigen::Triplet<double>> triplets;
    Eigen::VectorXd b = Eigen::VectorXd::Zero(N);

    for (int i = 0; i < N; ++i) {
        // ��������T[i]��ϵ��
        double q_transfer_T = -(U_trans_w(i) * A_w(i) + U_trans_t(i) * A_t(i));
        double q_transfer_const = -(U_trans_w(i) * A_w(i) * T_w(i) + U_trans_t(i) * A_t(i) * T_t(i));

        // ���Խ���Ԫ�أ�T[i]��ϵ����
        double diagonal_coeff = -(M_p(i) * cp_p + M_g(i) * cp_g) / dt
            - W_g(i) * cp_g - W_p(i) * cp_p
            - E_p(i) * cp_p - G_d(i) * cp_p - G_fly(i) * cp_p + q_transfer_T;

        triplets.emplace_back(i, i, diagonal_coeff);

        // ����ǶԽ���Ԫ��
        if (i == N - 1) {
            // ���һ��С�ң�ֻ������i-1�ķ���
            triplets.emplace_back(i, i - 1, E_p(i - 1) * cp_p);
            b(i) = -(M_p0(i) * cp_p + M_g0(i) * cp_g) * T0(i) / dt
                - W_air(i) * T_air(i) * cp_g
                - W_fa(i) * T_fa(i) * cp_p
                - heat_combus(i) + q_transfer_const;
        }
        else if (i == N_cyc - 1) {
            // ��ѭ��С�ң�����i+1������ + ����i-1�ķ��� + ����T[0]����ѭ��
            triplets.emplace_back(i, i + 1, W_g(i + 1) * cp_g + W_p(i + 1) * cp_p);
            triplets.emplace_back(i, i - 1, E_p(i - 1) * cp_p);
            triplets.emplace_back(i, 0, W_rec_p * cp_p + W_air(i) * cp_g);
            b(i) = -(M_p0(i) * cp_p + M_g0(i) * cp_g) * T0(i) / dt
                - W_fa(i) * T_fa(i) * cp_p
                - heat_combus(i) + q_transfer_const;
        }
        else if (i == 0) {
            // ����װ�ã�ֻ������������������������Ҳ������������������������
            triplets.emplace_back(i, i, -W_air(i) * cp_g);  // ������ӵ��Խ���Ԫ��
            triplets.emplace_back(i, i + 1, W_p(i + 1) * cp_p);
            b(i) = -(M_p0(i) * cp_p + M_g0(i) * cp_g) * T0(i) / dt
                - W_air(i) * T_air(i) * cp_g
                - W_fa(i) * T_fa(i) * cp_p
                - heat_combus(i) + q_transfer_const;
        }
        else {
            // ��ͨС�ң�����i+1������ + ����i-1�ķ���
            triplets.emplace_back(i, i + 1, W_g(i + 1) * cp_g + W_p(i + 1) * cp_p);
            triplets.emplace_back(i, i - 1, E_p(i - 1) * cp_p);
            b(i) = -(M_p0(i) * cp_p + M_g0(i) * cp_g) * T0(i) / dt
                - W_air(i) * T_air(i) * cp_g
                - W_fa(i) * T_fa(i) * cp_p
                - heat_combus(i) + q_transfer_const;
        }
    }

    // ����ϡ�����
    Eigen::SparseMatrix<double> A(N, N);
    A.setFromTriplets(triplets.begin(), triplets.end());
    A.makeCompressed();

    LinearSystem system;
    system.A = A;
    system.b = b;

    return system;
}