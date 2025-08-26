#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <omp.h>

#include "flow_submodel.hpp"
#include "gas_mass_balance.hpp"
#include "cyc_submodel.hpp"
#include "solid_balance_submodel.hpp"
#include "heat_transfer_submode.hpp"
#include "energy_submode_improved.hpp"
#include "mode_coeffs.hpp"

using namespace Eigen;
using namespace std;

const double M_PI = 3.14159265358979323846;

int main() {
    // Constants
    const double cp_p = 840;
    const double rho_p = 1800;
    const double rho_c = 1200;
    const double R = 8.314;
    const double g = 9.81;
    const double dt = 1;

    // Model parameters
    const int N = 37;
    const int N_fur = 35;
    const int N_cyc = 26;
    const int N_t = 8;
    const int N_vm = 35;
    const int Num_cyc = 2;
    const double D_bed = 2 * 5.28 * 10.27 / (5.28 + 10.27);
    const double A_bed = 5.28 * 10.27;
    const double A_plate = 8.77 * 2.8;
    const double A_cyc = 3.05 * 5.49;
    const double H_bed = 33.1;
    const double H_out = 29.8;
    const double N_bed = 952;
    const double H_con = 11.4;
    const double H_up = 4.4;
    const double D_up = 7.87;
    const double D_down = 1.34;

    // Load coefficients from mode_coeffs
    MatrixXi A_coeffs = ModeCoeffs::A;
    MatrixXi B_coeffs = ModeCoeffs::B;
    MatrixXi C_coeffs = ModeCoeffs::C;
    MatrixXi D_coeffs = ModeCoeffs::D;
    MatrixXi E_coeffs = ModeCoeffs::E;
    MatrixXi F_coeffs = ModeCoeffs::F;
    MatrixXi G_coeffs = ModeCoeffs::G;
    MatrixXi R_coeffs = ModeCoeffs::R;
    MatrixXd A_cab = ModeCoeffs::A_cab;
    VectorXd A_w = ModeCoeffs::A_w;
    VectorXd A_t = ModeCoeffs::A_t;
    MatrixXd h_cell = ModeCoeffs::h_cell;

    // Calculate volumes
    VectorXd v_cabs(N_fur);
    for (int i = 0; i < N_fur; ++i) {
        v_cabs(i) = A_cab.row(i).mean() * (h_cell(i, 0) - h_cell(i, 1));
    }
    double v_cyc = M_PI * pow(D_up / 2, 2) * H_up + M_PI / 12 * (pow(D_up, 2) + D_up * D_down + pow(D_down, 2)) * H_con;
    double v_ls = pow(D_down, 2) / 4 * M_PI * (20.5 - 8.3);

    // Initialize parameters
    const double h_cab_den = 0.3;
    const double phi = 0.85;
    const double P0 = 101325;
    const double T_ref = 273.15;

    // Particle size distribution
    VectorXd dp(12);
    dp << 0.06, 0.09, 0.125, 0.16, 0.2, 0.25, 0.3, 0.4, 0.5, 1.0, 2.0, 3.0;
    dp *= 0.001;

    VectorXd X0(12);
    X0 << 0.1, 0.226, 0.027, 0.027, 0.03, 0.09, 0.09, 0.25, 0.06, 0.05, 0.025, 0.025;

    // Time series parameters
    const int cal_len = 10000;
    VectorXd W_fa0 = VectorXd::Constant(cal_len, 10.0);
    double dTNh_s = P0 / (T_ref * R * 3600);
    VectorXd Q3_air = VectorXd::Constant(cal_len, 7140) * dTNh_s;
    VectorXd Q1_air = (238000 * P0 / ((T_ref + 20) * R * 3600) * VectorXd::Ones(cal_len) - Q3_air) / 2;
    VectorXd Q2_air = (238000 * P0 / ((T_ref + 20) * R * 3600) * VectorXd::Ones(cal_len) - Q3_air) / 2;
    VectorXd dP_fur = VectorXd::Constant(cal_len, 10000);

    double T0_air = T_ref + 20;
    VectorXd T1_air = VectorXd::Constant(cal_len, 400);
    VectorXd T2_air = VectorXd::Constant(cal_len, 450);
    VectorXd T3_air = VectorXd::Constant(cal_len, T0_air);
    VectorXd T_fa0 = VectorXd::Constant(cal_len, 300);

    // Material parameters
    const double X_VM = 0.293;
    VectorXd M_S = W_fa0 * 0.008 * 1000 / 32;
    VectorXd n_Mass = W_fa0 * 0.00443 * 1000 / 14;

    // Initialize air and fuel arrays
    MatrixXd Q_air = MatrixXd::Zero(cal_len, N - 2);
    Q_air.col(N - 3) = Q1_air;
    Q_air.col(22) = Q2_air;
    Q_air.col(23) = Q3_air;

    MatrixXd W_air = MatrixXd::Zero(cal_len, N);
    W_air.col(N - 1) = Q1_air;
    W_air.col(24) = Q2_air;
    W_air.col(25) = Q3_air;
    W_air.col(0) = Q3_air;

    MatrixXd T_air = MatrixXd::Zero(cal_len, N);
    T_air.col(N - 1) = T1_air;
    T_air.col(24) = T2_air;
    T_air.col(0) = T3_air;

    MatrixXd W_fa = MatrixXd::Zero(cal_len, N);
    W_fa.col(24) = W_fa0;

    MatrixXd G_fa = MatrixXd::Zero(N_fur, dp.size());

    MatrixXd T_fa = MatrixXd::Zero(cal_len, N);
    T_fa.col(24) = T_fa0;

    // Initialize material fractions
    MatrixXd X_fa_km = MatrixXd::Zero(N_fur, X0.size());
    for (int i = 0; i < N_fur; ++i) {
        X_fa_km.row(i) = X0;
    }

    MatrixXd Xfac_km = MatrixXd::Zero(N_fur, dp.size());
    Xfac_km.row(22).setConstant(0.5221);

    MatrixXd Xfaca_km = MatrixXd::Zero(N_fur, dp.size());
    MatrixXd Xca_km = MatrixXd::Zero(N, dp.size());

    // Initialize flow arrays
    MatrixXd G_hk_up = MatrixXd::Zero(h_cell.rows(), dp.size());
    MatrixXd G_hk_down = MatrixXd::Zero(h_cell.rows(), dp.size());
    MatrixXd e_k = MatrixXd::Zero(h_cell.rows(), dp.size());
    VectorXd e_k_ksf = VectorXd::Zero(h_cell.rows());
    double h_den = 3.9;

    // Initialize carbon content
    MatrixXd carbon_content = MatrixXd::Constant(N, dp.size(), 0.02);
    VectorXd n_mass = VectorXd::Zero(N_fur);
    VectorXd m_s = VectorXd::Zero(N_fur);
    VectorXd x_vm = VectorXd::Zero(N_fur);

    MatrixXd Xc_km0 = carbon_content;
    MatrixXd Xca_km0 = MatrixXd::Zero(N, dp.size());
    VectorXd M_cyck0 = VectorXd::Zero(dp.size());
    VectorXd M_lsk0 = VectorXd::Zero(dp.size());
    MatrixXd W_rek = MatrixXd::Zero(N_fur, dp.size());
    MatrixXd Gd = MatrixXd::Zero(N_fur, dp.size());
    double Vin = 16 * A_cyc;

    // Initialize temperatures
    VectorXd T = VectorXd::LinSpaced(N, 900 + T_ref, 850 + T_ref);
    VectorXd T0 = T;
    VectorXd T_w = VectorXd::Constant(N, 308 + T_ref);
    VectorXd T_secondary_w = VectorXd::Constant(N, 308 + T_ref);
    VectorXd T_t = VectorXd::Constant(N, 419 + T_ref);
    VectorXd T_cor = T, T_ann = T, T_den = T, T_secondary_den = T;
    double T_cyc_w = 308 + T_ref;

    // Initialize pressure
    VectorXd P = VectorXd::Constant(N_fur, 101325);
    VectorXd dP = VectorXd::Zero(N_fur);

    // Initialize gas composition
    VectorXd y_gas_0(7);
    y_gas_0 << 0.05, 0.01, 0.15, 0.001, 0.0004, 0.7386, 0.05;
    MatrixXd Y_gas_0 = MatrixXd::Zero(N_fur, 7);
    for (int i = 0; i < N_fur; ++i) {
        Y_gas_0.row(i) = y_gas_0;
    }

    VectorXd G_gas_0 = VectorXd::Constant(N_fur, 2756);
    double M_cyc0 = 2580.0 / Num_cyc;
    double M_ls0 = 35668.0 / Num_cyc;
    double dp_ls0 = 200;
    double W_re0 = W_fa0(0) * 20 / Num_cyc;

    // Initialize output arrays
    VectorXd total_G_outk = VectorXd::Zero(cal_len);
    MatrixXd total_heat_w = MatrixXd::Zero(cal_len, N_fur);
    VectorXd total_h_den = VectorXd::Zero(cal_len);
    MatrixXd total_M_g = MatrixXd::Zero(cal_len, N_fur);
    MatrixXd total_M_p = MatrixXd::Zero(cal_len, N_fur);
    MatrixXd total_X_km = MatrixXd::Zero(cal_len, dp.size());
    MatrixXd total_Y_o2 = MatrixXd::Zero(cal_len, N_fur);
    MatrixXd total_Y_co = MatrixXd::Zero(cal_len, N_fur);
    MatrixXd total_Y_co2 = MatrixXd::Zero(cal_len, N_fur);
    MatrixXd total_Y_so2 = MatrixXd::Zero(cal_len, N_fur);
    MatrixXd total_Y_no = MatrixXd::Zero(cal_len, N_fur);
    MatrixXd total_Y_n2 = MatrixXd::Zero(cal_len, N_fur);
    MatrixXd total_Y_h2o = MatrixXd::Zero(cal_len, N_fur);
    MatrixXd total_W_g = MatrixXd::Zero(cal_len, N_fur);
    Tensor<double, 3> total_e_k(cal_len, N_fur, dp.size());
    total_e_k.setZero();
    VectorXd total_M_cyc = VectorXd::Zero(cal_len);
    VectorXd total_M_ls = VectorXd::Zero(cal_len);
    VectorXd R_cyc = VectorXd::Zero(cal_len);
    MatrixXd total_T = MatrixXd::Zero(cal_len, N);
    MatrixXd total_P = MatrixXd::Zero(cal_len, N_fur);
    MatrixXd Q_heat_w = MatrixXd::Zero(cal_len, N);
    MatrixXd Q_heat_t = MatrixXd::Zero(cal_len, N);

    // Create model instances
    Flow flow(dp, phi, rho_p, rho_c, D_bed, A_bed, A_plate, H_bed, H_out, N_bed, X0, A_cyc);
    GAS gas(N_fur, dt, dp, rho_p, rho_c, N_vm, H_bed);
    CycMassEnergy cyc;
    vector<double> v_cabs_vec(v_cabs.data(), v_cabs.data() + v_cabs.size());
    SolidMass solid(N_fur, v_cabs_vec, rho_p, dt);
    HeatTransfer hr(cp_p, rho_p, dp.mean(), N, N_t);
    Energy eng(N, N_cyc, cp_p);

    // Lambda function for gas properties update
    auto update_gas_properties = [](const VectorXd& T, const VectorXd& P) {
        double rho_g = 0.316;
        double mu_g = 4.5e-5;
        double cp_g = 34.56;
        double thermo_g = 0.25;
        double K_g = 0.06;
        return make_tuple(rho_g, mu_g, cp_g, thermo_g, K_g);
        };

    // Declare variables that will be used across iterations
    VectorXd X_km, G_denk, G_Hk, effk_cyc, G_TDHk, G_densk, e_TDHk, e_denk, ak, U_mfk, e_mfk, u_tk, G_outk;
    double Wdr, eff_bed_out;
    VectorXd Wdrk, G_ksf_up, G_ksf_down;
    VectorXd X_cyc0, X_ls0;

    // Lambda function for flow parameters calculation
    auto calculate_flow_parameters = [&](const MatrixXd& h_cell, int n_den) {
        double ep_den_ksf = ((1 - e_denk.array()) * X_km.array()).sum();
        Wdrk = Wdr * X_km;

        for (int i = 0; i < h_cell.rows(); ++i) {
            if (i < h_cell.rows() - n_den) {
                for (int j = 0; j < dp.size(); ++j) {
                    auto integrand = [&](double h_val) {
                        return e_TDHk(j) + (e_denk(j) - e_TDHk(j)) * exp(-ak(j) * h_val);
                        };

                    // Numerical integration (trapezoidal rule)
                    int n_steps = 100;
                    double h_start = h_cell(i, 1) - h_den;
                    double h_end = h_cell(i, 0) - h_den;
                    double h_step = (h_end - h_start) / n_steps;
                    double integral = 0;
                    for (int k = 0; k <= n_steps; ++k) {
                        double h = h_start + k * h_step;
                        double weight = (k == 0 || k == n_steps) ? 0.5 : 1.0;
                        integral += weight * integrand(h) * h_step;
                    }
                    e_k(i, j) = integral / (h_cell(i, 0) - h_cell(i, 1));
                }
                e_k_ksf(i) = (e_k.row(i).array() * X_km.array()).sum();
                dP(i) = g * rho_p * (1 - e_k_ksf(i)) * (h_cell(i, 0) - h_cell(i, 1));
                P(i) = P0 + dP.head(i).sum();
                G_hk_up.row(i) = ((G_TDHk.array() + (G_densk.array() - G_TDHk.array()) *
                    (-ak.array() * (h_cell(i, 0) - h_den)).exp()) * X_km.array() * A_cab(i, 1)).matrix();
                G_hk_down.row(i) = ((G_TDHk.array() + (G_densk.array() - G_TDHk.array()) *
                    (-ak.array() * (h_cell(i, 1) - h_den)).exp()) * X_km.array() * A_cab(i, 0)).matrix();
            }
            else {
                dP(i) = g * rho_p * ep_den_ksf * (h_cell(i, 0) - h_cell(i, 1));
                P(i) = P0 + dP.head(i).sum();
                e_k.row(i) = e_denk;
                e_k_ksf(i) = (e_denk.array() * X_km.array()).sum();
                G_hk_up.row(i) = G_densk.array() * X_km.array() * A_cab(i, 1);
                G_hk_down.row(i) = G_densk.array() * X_km.array() * A_cab(i, 0);
            }
        }
        G_ksf_down = G_hk_down.rowwise().sum();
        G_ksf_up = G_hk_up.rowwise().sum();

        return make_tuple(P, dP, e_k, e_k_ksf, G_hk_up, G_hk_down, Wdrk, G_ksf_up, G_ksf_down);
        };

    // Main calculation loop
    for (int i = 0; i < cal_len; ++i) {
        cout << "i " << i << endl;

        // Update gas properties
        auto [rho_g, mu_g, cp_g, thermo_g_mode, K_g_mode] = update_gas_properties(T.segment(2, N_fur), P);
        double rho_g_flow = rho_g;
        double mu_g_flow = mu_g;
        double thermo_g = thermo_g_mode;
        double K_g = K_g_mode;

        double u0 = (Q1_air(i) + Q2_air(i) + Q3_air(i)) * R * (T_ref + 850) / (P0 * A_bed);
        double u0_den = u0;

        if (i == 0) {
            // Initial flow calculation
            auto result = flow.solveFlowSubmodel(u0, u0_den, Vin, rho_g_flow, mu_g_flow, h_den, 0.25 * W_fa0(i), dP_fur(i));

            X_km = result.X_km;
            h_den = result.h_den;
            G_denk = result.G_denk;
            G_Hk = result.G_Hk;
            effk_cyc = result.effk_cyc;
            eff_bed_out = result.eff_bed_out;
            G_TDHk = result.G_TDHk;
            G_densk = result.G_densk;
            e_TDHk = result.e_TDHk;
            e_denk = result.e_denk;
            ak = result.ak;
            Wdr = result.Wdr;
            U_mfk = result.U_mfk;
            e_mfk = result.e_mfk;
            u_tk = result.U_tk;
            G_outk = result.G_outk;

            int n_den = static_cast<int>(h_den / h_cab_den);
            auto [P_calc, dP_calc, e_k_calc, e_k_ksf_calc, G_hk_up_calc, G_hk_down_calc, Wdrk_calc, G_ksf_up_calc, G_ksf_down_calc] = calculate_flow_parameters(h_cell, n_den);
            P = P_calc;
            dP = dP_calc;
            e_k = e_k_calc;
            e_k_ksf = e_k_ksf_calc;
            G_hk_up = G_hk_up_calc;
            G_hk_down = G_hk_down_calc;

            VectorXd M_g_0 = (P.array() * e_k_ksf.array() * v_cabs.array() / (R * T.segment(2, N_fur).array())).matrix();
            total_h_den(i) = h_den;
            total_M_g.row(i) = M_g_0;
            VectorXd M_p_0 = rho_p * v_cabs.cwiseProduct(VectorXd::Ones(N_fur) - e_k_ksf);
            total_M_p.row(i) = M_p_0;
            VectorXd X_km_0 = X_km;
            total_X_km.row(i) = X_km;
            MatrixXd e_k0 = e_k;
            for (int j = 0; j < N_fur; ++j) {
                for (int k = 0; k < dp.size(); ++k) {
                    total_e_k(i, j, k) = e_k(j, k);
                }
            }
            total_P.row(i) = P;
            total_G_outk(i) = G_outk.sum();
            total_M_cyc(i) = M_cyc0;
            total_M_ls(i) = M_ls0;
            X_cyc0 = X_km;
            X_ls0 = X_km;
            continue;
        }

        // Main flow calculation for i > 0
        auto flow_result = flow.solveFlowSubmodel(u0, u0_den, Vin, rho_g_flow, mu_g_flow, h_den, 0.25 * W_fa0(i), dP_fur(i));
        X_km = flow_result.X_km;
        h_den = flow_result.h_den;
        G_denk = flow_result.G_denk;
        G_Hk = flow_result.G_Hk;
        effk_cyc = flow_result.effk_cyc;
        eff_bed_out = flow_result.eff_bed_out;
        G_TDHk = flow_result.G_TDHk;
        G_densk = flow_result.G_densk;
        e_TDHk = flow_result.e_TDHk;
        e_denk = flow_result.e_denk;
        ak = flow_result.ak;
        Wdr = flow_result.Wdr;
        U_mfk = flow_result.U_mfk;
        e_mfk = flow_result.e_mfk;
        u_tk = flow_result.U_tk;
        G_outk = flow_result.G_outk;

        total_h_den(i) = h_den;
        total_G_outk(i) = G_outk.sum();

        int n_den = ceil(h_den / h_cab_den);
        int N_dil = N - n_den;

        auto [P_calc, dP_calc, e_k_calc, e_k_ksf_calc, G_hk_up_calc, G_hk_down_calc, Wdrk_calc, G_ksf_up_calc, G_ksf_down_calc] =
            calculate_flow_parameters(h_cell, n_den);
        P = P_calc;
        dP = dP_calc;
        e_k = e_k_calc;
        e_k_ksf = e_k_ksf_calc;
        G_hk_up = G_hk_up_calc;
        G_hk_down = G_hk_down_calc;
        Wdrk = Wdrk_calc;
        VectorXd G_ksf_up = G_ksf_up_calc;
        VectorXd G_ksf_down = G_ksf_down_calc;

        VectorXd epsilon_p_cl = VectorXd::Ones(N_fur) - e_k_ksf;
        VectorXd epsilon_p_c = VectorXd::Ones(N_fur) - e_k_ksf;

        VectorXd M_g = (P.array() * e_k_ksf.array() * v_cabs.array() / (R * T.segment(2, N_fur).array())).matrix();
        VectorXd M_p = rho_p * v_cabs.cwiseProduct(VectorXd::Ones(N_fur) - e_k_ksf);

        total_M_g.row(i) = M_g;
        total_M_p.row(i) = M_p;
        total_X_km.row(i) = X_km;
        for (int j = 0; j < N_fur; ++j) {
            for (int k = 0; k < dp.size(); ++k) {
                total_e_k(i, j, k) = e_k(j, k);
            }
        }
        total_P.row(i) = P;

        VectorXd M_g_0 = total_M_g.row(i - 1);
        VectorXd M_p_0 = total_M_p.row(i - 1);
        VectorXd X_km0 = total_X_km.row(i - 1);
        MatrixXd e_k0(N_fur, dp.size());
        for (int j = 0; j < N_fur; ++j) {
            for (int k = 0; k < dp.size(); ++k) {
                e_k0(j, k) = total_e_k(i - 1, j, k);
            }
        }

        VectorXd W_p = G_ksf_up;
        n_mass.tail(N_vm).setConstant(n_Mass(i) / N_vm);
        m_s.tail(N_vm).setConstant(M_S(i) / 19);
        x_vm.tail(N_vm).setConstant(X_VM);

        MatrixXd M_km = MatrixXd::Zero(N_fur, dp.size());
        MatrixXd M_km0 = MatrixXd::Zero(N_fur, dp.size());
        for (int j = 0; j < N_fur; ++j) {
            M_km.row(j) = (1 - e_k.row(j).array()).abs() * X_km.array() * rho_p * v_cabs(j);
            M_km0.row(j) = (1 - e_k0.row(j).array()).abs() * X_km0.array() * rho_p * v_cabs(j);
        }

        // Gas mass balance
        MatrixXd A_coeffs_d = A_coeffs.cast<double>();
        MatrixXd B_coeffs_d = B_coeffs.cast<double>();
        MatrixXd C_coeffs_d = C_coeffs.cast<double>();
        MatrixXd D_coeffs_d = D_coeffs.cast<double>();
        MatrixXd E_coeffs_d = E_coeffs.cast<double>();
        MatrixXd F_coeffs_d = F_coeffs.cast<double>();
        MatrixXd G_coeffs_d = G_coeffs.cast<double>();
        MatrixXd R_coeffs_d = R_coeffs.cast<double>();

        auto gas_result = gas.total_gas_mass_balance(
            M_g_0, M_g, Y_gas_0, G_gas_0, P, e_k, e_k_ksf, e_mfk.sum(),
            X_km, u_tk.mean(), carbon_content.block(2, 0, N_fur, dp.size()),
            Xca_km.block(2, 0, N_fur, dp.size()), T.segment(2, N_fur),
            x_vm, m_s, n_mass, W_fa0(i), Q_air.row(i).transpose().head(N_fur),
            v_cabs, mu_g, rho_g,
            A_coeffs_d, B_coeffs_d, C_coeffs_d, D_coeffs_d,
            E_coeffs_d, F_coeffs_d, G_coeffs_d, R_coeffs_d, h_cell
        );

        VectorXd W_g = gas_result.G_gas;
        VectorXd heat_cabs = gas_result.heat_cabs;
        Y_gas_0 = gas_result.Y_gas_pre;
        G_gas_0 = gas_result.G_gas_pre;
        MatrixXd R_char = gas_result.R_char;
        MatrixXd R_ca = gas_result.R_ca;

        double heat_gas = heat_cabs.sum();
        VectorXd R_chark = R_char.rowwise().sum() * 1000;
        VectorXd V_g = W_g.array() * R * T.segment(2, N_fur).array() / P.array();
        Vin = V_g(0) / Num_cyc;
        double u_in = Vin / A_cyc;

        total_Y_o2.row(i) = gas_result.Y_o2;
        total_Y_co.row(i) = gas_result.Y_co;
        total_Y_co2.row(i) = gas_result.Y_co2;
        total_Y_so2.row(i) = gas_result.Y_so2;
        total_Y_no.row(i) = gas_result.Y_no;
        total_Y_n2.row(i) = gas_result.Y_n2;
        total_Y_h2o.row(i) = gas_result.Y_h2o;
        total_W_g.row(i) = W_g;

        // Cyclone calculations
        auto cyc_result = cyc.calculateCycParameters(
            u_in, u_tk, G_outk / Num_cyc, effk_cyc,
            M_cyc0, X_cyc0, X_km, dP_fur(i),
            M_ls0, dp_ls0, W_re0, X_ls0,
            (e_mfk.array() * X_km.array()).sum(), u0
        );

        double M_cyc = cyc_result.M_cyc;
        VectorXd X_cyc = cyc_result.X_cyc;
        double dp_cyc = cyc_result.dp_cyc;
        double M_ls = cyc_result.M_ls;
        double dp_ls = cyc_result.dp_ls;
        double W_re = cyc_result.W_re;
        VectorXd X_ls = cyc_result.X_ls;
        VectorXd w_rek = cyc_result.W_rek;
        VectorXd G_downk = cyc_result.G_downk;
        VectorXd G_flyK = cyc_result.G_flyK;
        VectorXd M_cyck = cyc_result.M_cyck;
        VectorXd M_lsk = cyc_result.M_lsk;
        double dp_sp = cyc_result.dp_sp;

        double r_cyc = W_re * Num_cyc / W_fa0(i);
        R_cyc(i) = r_cyc;
        total_M_cyc(i) = M_cyc;
        total_M_ls(i) = M_ls;
        M_cyc0 = M_cyc;
        M_ls0 = M_ls;
        X_cyc0 = X_cyc;
        dp_ls0 = dp_ls;
        W_re0 = W_re;
        X_ls0 = X_ls;

        G_fa.row(22) = W_fa0(i) * X_fa_km.row(22);
        W_rek.row(23) = w_rek * Num_cyc;
        Gd.row(N_fur - 1) = Wdrk.transpose();

        // Solid mass balance
        VectorXd Gs, Gsk_sum;
        MatrixXd Gsk;
        vector<double> dp_vec(dp.data(), dp.data() + dp.size());
        solid.solidMass(
            dp_vec, M_km, M_km0, G_hk_up, G_hk_down,
            G_fa, W_rek, Gd, G_outk, R_char,
            Xfac_km, Xc_km0, R_ca, Xfaca_km, Xca_km0,
            G_downk * Num_cyc, G_flyK * Num_cyc,
            M_cyck * Num_cyc, M_cyck0 * Num_cyc,
            w_rek * Num_cyc, M_lsk * Num_cyc, M_lsk0 * Num_cyc,
            carbon_content, Xca_km, Gs, Gsk
        );

        VectorXd X_c = VectorXd::Zero(N_fur);
        for (int j = 0; j < N_fur; ++j) {
            double numerator = 0, denominator = 1 - e_k_ksf(j);
            for (int k = 0; k < dp.size(); ++k) {
                numerator += (1 - e_k(j, k)) * X_km(k) * carbon_content(j + 2, k);
            }
            X_c(j) = numerator / denominator * 100;
        }

        Xc_km0 = carbon_content;
        Xca_km0 = Xca_km;
        M_cyck0 = M_cyck;
        M_lsk0 = M_lsk;

        double rho_susp_cyc = M_cyc / v_cyc;
        double rho_susp_ls = M_ls / v_ls;
        VectorXd rho_susp = M_p.array() / v_cabs.array();

        // Prepare extended vectors for heat transfer
        VectorXd rho_susp_extended(N);
        rho_susp_extended(0) = rho_susp_cyc;
        rho_susp_extended(1) = rho_susp_ls;
        rho_susp_extended.segment(2, N_fur) = rho_susp;

        VectorXd epsilon_p_cl_extended(N);
        epsilon_p_cl_extended(0) = 1 - e_mfk.sum();
        epsilon_p_cl_extended(1) = 1 - e_mfk.sum();
        epsilon_p_cl_extended.segment(2, N_fur) = epsilon_p_cl;

        VectorXd epsilon_p_c_extended(N);
        epsilon_p_c_extended(0) = 1 - e_mfk.sum();
        epsilon_p_c_extended(1) = 1 - e_mfk.sum();
        epsilon_p_c_extended.segment(2, N_fur) = epsilon_p_c;

        // Heat transfer calculations
        vector<double> T_secondary_w_vec(T_secondary_w.data(), T_secondary_w.data() + T_secondary_w.size());
        vector<double> T_secondary_den_vec(T_secondary_den.data(), T_secondary_den.data() + T_secondary_den.size());
        vector<double> T_w_vec(T_w.data(), T_w.data() + T_w.size());
        vector<double> T_cor_vec(T_cor.data(), T_cor.data() + T_cor.size());
        vector<double> T_ann_vec(T_ann.data(), T_ann.data() + T_ann.size());
        vector<double> T_den_vec(T_den.data(), T_den.data() + T_den.size());
        vector<double> T_t_vec(T_t.data(), T_t.data() + T_t.size());
        vector<double> epsilon_p_cl_vec(epsilon_p_cl_extended.data(), epsilon_p_cl_extended.data() + epsilon_p_cl_extended.size());
        vector<double> epsilon_p_c_vec(epsilon_p_c_extended.data(), epsilon_p_c_extended.data() + epsilon_p_c_extended.size());
        vector<double> rho_susp_vec(rho_susp_extended.data(), rho_susp_extended.data() + rho_susp_extended.size());

        auto [heat_w_vec, heat_t_vec] = hr.heat_transfer_cabs(
            N_dil, X_km, T_secondary_w_vec, T_cyc_w, u0, U_mfk.mean(),
            T_secondary_den_vec, e_mfk.sum(), T_w_vec, T_cor_vec, T_ann_vec,
            epsilon_p_cl_vec, rho_susp_vec, T_den_vec, epsilon_p_c_vec,
            T_t_vec, mu_g, u_tk.mean(), cp_g, thermo_g, K_g, rho_g
        );

        VectorXd heat_w(N);
        VectorXd heat_t(N);
        for (int j = 0; j < N; ++j) {
            heat_w(j) = heat_w_vec[j];
            heat_t(j) = heat_t_vec[j];
        }
        total_heat_w.row(i) = heat_w.segment(2, N_fur);

        // Prepare for energy calculations
        VectorXd G_d = Gd.rowwise().sum();
        VectorXd G_fly = VectorXd::Zero(N);
        G_fly(1) = G_flyK.sum() * 2;

        // Extend vectors for energy calculations
        VectorXd M_p_0_extended(N);
        M_p_0_extended(0) = total_M_cyc(i - 1) * Num_cyc;
        M_p_0_extended(1) = total_M_ls(i - 1) * Num_cyc;
        M_p_0_extended.segment(2, N_fur) = M_p_0;

        VectorXd M_g_0_extended(N);
        M_g_0_extended(0) = 0.0;
        M_g_0_extended(1) = 0.0;
        M_g_0_extended.segment(2, N_fur) = M_g_0;

        VectorXd M_p_extended(N);
        M_p_extended(0) = M_cyc * Num_cyc;
        M_p_extended(1) = M_ls * Num_cyc;
        M_p_extended.segment(2, N_fur) = M_p;

        VectorXd M_g_extended(N);
        M_g_extended(0) = 0.0;
        M_g_extended(1) = 0.0;
        M_g_extended.segment(2, N_fur) = M_g;

        VectorXd W_p_extended(N);
        W_p_extended(0) = G_downk.sum() * Num_cyc;
        W_p_extended(1) = W_re * Num_cyc;
        W_p_extended(2) = G_outk.sum();
        W_p_extended.segment(3, N_fur - 1) = W_p.tail(N_fur - 1);

        VectorXd G_d_extended(N);
        G_d_extended(0) = 0;
        G_d_extended(1) = 0;
        G_d_extended.segment(2, N_fur) = G_d;

        VectorXd W_g_extended(N);
        W_g_extended(0) = W_g(0);
        W_g_extended(1) = 0.0;
        W_g_extended.segment(2, N_fur) = W_g;

        VectorXd Gs_extended(N);
        Gs_extended(0) = 0.0;
        Gs_extended(1) = 0.0;
        Gs_extended.segment(2, N_fur) = Gs;

        VectorXd heat_cabs_extended(N);
        heat_cabs_extended(0) = 0.0;
        heat_cabs_extended(1) = 0.0;
        heat_cabs_extended.segment(2, N_fur) = heat_cabs;

        // Energy conservation
        auto energy_result = eng.energy_conservation(
            M_p_0_extended, M_g_0_extended, T0, M_p_extended, M_g_extended,
            W_p_extended, G_d_extended, G_fly, W_g_extended,
            W_air.row(i).transpose(), T_air.row(i).transpose(),
            W_fa.row(i).transpose(), T_fa.row(i).transpose(),
            W_re * Num_cyc, Gs_extended, heat_cabs_extended,
            heat_w, heat_t, A_w, A_t, T_w, T_t, dt
        );

        T = energy_result.T;
        VectorXd q_heat_w = energy_result.q_heat_w;
        VectorXd q_heat_t = energy_result.q_heat_t;

        total_T.row(i) = T;
        double current_state3 = T(2) - T_ref;
        T0 = T;
        T_cor = T;
        T_ann = T;
        T_secondary_den = T;
        T_den = T;
        Q_heat_w.row(i) = q_heat_w;
        Q_heat_t.row(i) = q_heat_t;

        double current_state1 = gas_result.Y_o2(0) * 100;
        double current_state2 = r_cyc;
        double current_state4 = M_cyc * 2;
    }

    return 0;
}