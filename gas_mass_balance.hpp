#ifndef GAS_MASS_BALANCE_HPP
#define GAS_MASS_BALANCE_HPP

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

class GAS {
public:
    // 构造函数
    GAS(int N, double dt,
        const Eigen::VectorXd& dp, double rho_p, double rho_c,
        int N_vm, double H_bed);

    // 主要计算函数
    struct GasBalanceResult {
        Eigen::VectorXd G_gas;
        Eigen::VectorXd heat_cabs;
        Eigen::VectorXd Y_o2, Y_co, Y_co2, Y_so2, Y_no, Y_n2, Y_h2o;
        Eigen::MatrixXd Y_gas_pre;
        Eigen::VectorXd G_gas_pre;
        Eigen::MatrixXd R_char;
        Eigen::MatrixXd R_ca;
    };

    GasBalanceResult total_gas_mass_balance(
        const Eigen::VectorXd& M_g_pre,
        const Eigen::VectorXd& M_g,
        const Eigen::MatrixXd& Y_gas_pre,
        const Eigen::VectorXd& G_gas_pre,
        const Eigen::VectorXd& P,
        const Eigen::MatrixXd& e_k,
        const Eigen::VectorXd& e_k_ksf,
        double e_mfk,
        const Eigen::VectorXd& X_km,
        double u_tk,
        const Eigen::MatrixXd& carbon_content,
        const Eigen::MatrixXd& Xca_km,
        const Eigen::VectorXd& T,
        const Eigen::VectorXd& X_VM,
        const Eigen::VectorXd& M_S,
        const Eigen::VectorXd& n_Mass,
        double W_fa,
        const Eigen::VectorXd& Q_air,
        const Eigen::VectorXd& v_cabs,
        double mu_g,
        double rho_g,
        const Eigen::MatrixXd& A_coeffs,
        const Eigen::MatrixXd& B_coeffs,
        const Eigen::MatrixXd& C_coeffs,
        const Eigen::MatrixXd& D_coeffs,
        const Eigen::MatrixXd& E_coeffs,
        const Eigen::MatrixXd& F_coeffs,
        const Eigen::MatrixXd& G_coeffs,
        const Eigen::MatrixXd& R_coeffs,
        const Eigen::MatrixXd& h_cell
    );

private:
    // 成员变量
    int N;
    double dt;
    Eigen::VectorXd dp;
    double rho_p, rho_c;
    int N_vm;
    double H_bed;

    // 常数
    const double g = 9.81;
    const double R = 8.314;
    const double K_cd = 1.5;
    const double small_threshold = 0.05e-3;
    const double large_threshold = 1e-3;
    const double mol_c = 12.0;
    const double mol_ca = 100.0;

    // 摩尔质量数组
    Eigen::VectorXd molar_mass;

    // 系数矩阵
    Eigen::MatrixXd coefficients;
    Eigen::VectorXd heat_value_stdmol;

    // 热量参数
    double heat_co2 = 393.5;
    double heat_co = 110.5;
    double heat_c_co2 = -172.5;
    double heat_c_co = 283.0;
    double constant = 859.0;
    double energy_ratio = 19500.0;
    double T_ref = 298.0;
    double P_ref = 101300.0;
    double D_ref = 2.01e-5;
    double coefficients_caco3 = 0.035;
    double heat_so2 = 308.4;
    double heating_no_consumption = 373.3;
    double heating_no_generation = 226.0;
    double k1_no0 = 120.0;
    double k2_no0 = 64.0;
    double q1_cof = 15800.0;
    double q2_cof = 14700.0;
    double no_order = 1.0;
    double m1_order = 0.3;
    double m2_order = 0.45;
    double al2o3_ratio = 0.0;

    // 中间变量（用于方程组求解）
    double reaction_char_o2, reaction_char_co, reaction_char_co2;
    double reaction_co2_co, reaction_co2;
    double reaction_co, reaction_co_o2, reaction_co_co2;
    double reaction_so2, reaction_so2_o2, reaction_so2_co2;
    double reaction_no, reaction_no_co, reaction_no_co2, reaction_no_n2;
    double m_NO;

    // 内部计算函数
    struct GasCabResult {
        Eigen::VectorXd Y_gas_current;
        double G_gas_current;
        Eigen::VectorXd Y_gas_pre;
        double G_gas_pre;
        double heat;
        Eigen::VectorXd r_char;
        Eigen::VectorXd r_ca;
        Eigen::VectorXd single_char;
        double d_o2_char, d_o2_co, d_o2_vm, d_o2_so2;
    };

    GasCabResult gas_cab_constant(
        double t, double p, double v_cab,
        const Eigen::VectorXd& e_k, double e_k_ksf, double e_mfk,
        const Eigen::VectorXd& X_km, double u_tk,
        const Eigen::VectorXd& carbon_content,
        const Eigen::VectorXd& Y_gas_pre, double G_gas_pre,
        double M_g_pre, double M_g,
        double x_vm, double m_S, double n_mass, double W_fa,
        const Eigen::VectorXd& Y_gas_up, double G_gas_up,
        double Q_air, const Eigen::VectorXd& Xca_km,
        double mu_g, double rho_g, const Eigen::VectorXd& h_cell,
        const Eigen::VectorXd& a_coeffs,
        const Eigen::VectorXd& b_coeffs,
        const Eigen::VectorXd& c_coeffs,
        const Eigen::VectorXd& d_coeffs,
        const Eigen::VectorXd& e_coeffs,
        const Eigen::VectorXd& f_coeffs,
        const Eigen::VectorXd& g_coeffs,
        const Eigen::VectorXd& r_coeffs,
        double ratio_o2 = 0.21, double ratio_n2 = 0.79
    );

    void calculate_no_generation(double x_vm, double n_mass,
        const Eigen::VectorXd& poly_coeffs,
        double ratio_o2_no = 0.9);

    struct CharCombustionResult {
        double delta_O2, delta_CO, delta_CO2, total_heat;
        Eigen::VectorXd reaction_rate;
        double heat_c, heat_co2, heat_co;
        Eigen::VectorXd M_factor, k_total, k_react, k_diff;
        double k_co2, k_co;
        double d_o2_char, d_o2_co;
        Eigen::VectorXd single_char;
    };

    CharCombustionResult combustion_char(
        double t, double p, double v_cab,
        const Eigen::VectorXd& e_k, double e_k_ksf, double e_mfk,
        const Eigen::VectorXd& X_km, double u_tk,
        const Eigen::VectorXd& carbon_content,
        const Eigen::VectorXd& Y_gas,
        double mu_g, double rho_g
    );

    struct SO2CombustionResult {
        double delta_so2, delta_o2, delta_co2, total_heat;
        Eigen::VectorXd reaction_rate_ca;
    };

    SO2CombustionResult combustion_so2(
        double t, double p, double v_cab,
        const Eigen::VectorXd& Y_gas,
        const Eigen::VectorXd& Xca_km,
        const Eigen::VectorXd& e_k,
        const Eigen::VectorXd& X_km
    );

    struct NOCombustionResult {
        double delta_no, delta_co, delta_n2, delta_co2, total_heat;
    };

    NOCombustionResult combustion_no(
        double t, double v_cab,
        const Eigen::VectorXd& carbon_content,
        const Eigen::VectorXd& Y_gas,
        const Eigen::VectorXd& e_k
    );

    struct VMCombustionResult {
        double delta_O2, delta_CO2, delta_H2O, total_heat_vm;
    };

    VMCombustionResult combustion_vm(double x_vm, double W_fa,
        const Eigen::VectorXd& h_cell);

    Eigen::VectorXd _calculate_mechanical_factor(double CO_ratio);

    // 非线性方程求解器
    Eigen::VectorXd solve_nonlinear_equations(
        const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& func,
        const Eigen::VectorXd& x0,
        const Eigen::VectorXd& lb,
        const Eigen::VectorXd& ub
    );
};

#endif // GAS_MASS_BALANCE_HPP