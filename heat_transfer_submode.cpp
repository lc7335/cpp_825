#include "heat_transfer_submode.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <iostream>

// 构造函数
HeatTransfer::HeatTransfer(double cp_p, double rho_p, double dp, int N, int N_t)
    : cp_p(cp_p), rho_p(rho_p), dp(dp), N(N), N_t(N_t),
    e_den(0.9), e_w(0.85), e_p(0.8), g(9.81),
    sigma(5.67e-8), B(0.667) {
}

// 简单的数值求解器实现fsolve功能
double HeatTransfer::fsolve(std::function<double(double)> func, double x0, double tol, int max_iter) {
    double x = x0;
    double fx = func(x);

    for (int i = 0; i < max_iter; ++i) {
        if (std::abs(fx) < tol) {
            return x;
        }

        // 使用数值微分计算导数
        double h = 1e-8;
        double fx_h = func(x + h);
        double df_dx = (fx_h - fx) / h;

        if (std::abs(df_dx) < 1e-12) {
            throw std::runtime_error("Derivative too small in fsolve");
        }

        // 牛顿法更新
        x = x - fx / df_dx;
        fx = func(x);
    }

    throw std::runtime_error("fsolve did not converge");
}

// 主程序：计算各小室与水冷壁和屏式换热器的传热系数
std::tuple<std::vector<double>, std::vector<double>> HeatTransfer::heat_transfer_cabs(
    int N_dil,
    const Eigen::VectorXd& X_km,
    const std::vector<double>& T_secondary_w,
    double T_cyc_w,
    double u0,
    double u_mf,
    const std::vector<double>& T_secondary_den,
    double epsilon_mf,
    const std::vector<double>& T_w,
    const std::vector<double>& T_cor,
    const std::vector<double>& T_ann,
    const std::vector<double>& epsilon_p_cl,
    const std::vector<double>& rho_susp,
    const std::vector<double>& T_den,
    const std::vector<double>& epsilon_p_c,
    const std::vector<double>& T_t,
    double mu_g,
    double U_t,
    double cp_g,
    double thermo_g,
    double K_g,
    double rho_g) {

    std::vector<double> heat_transfer_w(N, 0.0);
    std::vector<double> heat_transfer_t(N, 0.0);

    // 计算与水冷壁的传热系数
    for (size_t i = 0; i < T_t.size(); ++i) {
        if (i == 0) {
            heat_transfer_w[i] = coefficient_den(X_km, T_secondary_w[i], u0, u_mf,
                epsilon_mf, T_secondary_den[i], cp_g, thermo_g, rho_g);
        }
        else if (i == 1) {
            heat_transfer_w[i] = coefficient_dil(X_km, T_cyc_w, T_cor[i], T_ann[i],
                epsilon_p_cl[i], rho_susp[i], rho_g, cp_g, K_g);
        }
        else if (i >= 2 && i < N_dil) {
            heat_transfer_w[i] = coefficient_dil(X_km, T_w[i], T_cor[i], T_ann[i],
                epsilon_p_cl[i], rho_susp[i], rho_g, cp_g, K_g);
        }
        else {
            heat_transfer_w[i] = coefficient_den(X_km, T_w[i], u0, u_mf,
                epsilon_mf, T_den[i], cp_g, thermo_g, rho_g);
        }
    }

    // 计算与屏式换热器的传热系数
    for (size_t j = 0; j < T_t.size(); ++j) {
        if (j >= 2 && j < N_t) {
            heat_transfer_t[j] = coefficient_t(X_km, T_cor[j], T_ann[j], epsilon_p_c[j],
                T_t[j], mu_g, U_t, cp_g, thermo_g, rho_g, K_g);
        }
        else {
            heat_transfer_t[j] = 0.0;
        }
    }

    return std::make_tuple(heat_transfer_w, heat_transfer_t);
}

// 密相区&外置换热器传热系数
double HeatTransfer::coefficient_den(
    const Eigen::VectorXd& X_km,
    double T_w,
    double u0,
    double u_mf,
    double epsilon_mf,
    double T_den,
    double cp_g,
    double thermo_g,
    double rho_g,
    double thermo_p,
    double pipe_factor) {

    // 对流换热
    double ratio_u = u0 / u_mf;
    double f_b = 0.08553 * std::pow(u_mf * u_mf * (ratio_u - 1) * (ratio_u - 1) / (g * dp), 0.1948);
    double t_e = 8.932 * std::pow(g * dp / (u_mf * u_mf * (ratio_u - 1) * (ratio_u - 1)), 0.0756)
        * std::sqrt(dp / 0.025);

    double rho_e = rho_p * (1 - epsilon_mf);
    double cp_e = cp_p * (1 - epsilon_mf) + cp_g * epsilon_mf;

    double thermo_e0 = thermo_g * (1 + (1 - epsilon_mf) * (1 - thermo_g / thermo_p) /
        (thermo_g / thermo_p + 0.28 * std::pow(epsilon_mf,
            0.63 * std::pow(thermo_g / thermo_p, -0.18))));

    double thermo_e = thermo_e0 + 0.1 * dp * u_mf * rho_g * cp_g;

    double R_e = std::sqrt(PI * t_e / (thermo_e * rho_e * cp_e));
    double R_k = dp / (3.75 * thermo_e);

    // 计算对流传热系数数组
    Eigen::VectorXd coef_con_e = X_km * ((1 - f_b) / (R_k + 0.45 * R_e));

    // 辐射换热
    double coef_rad_e = sigma * (T_den * T_den + T_w * T_w) * (T_den + T_w) /
        (1 / e_den + 1 / e_w - 1);

    // 密相区总传热系数
    double coef_den = pipe_factor * (coef_con_e.sum() + coef_rad_e);

    return coef_den;
}

// 稀相区&环形区悬挂受热面传热系数
double HeatTransfer::coefficient_dil(
    const Eigen::VectorXd& X_km,
    double T_w,
    double T_cor,
    double T_ann,
    double epsilon_p_cl,
    double rho_susp,
    double rho_g,
    double cp_g,
    double K_g,
    double K_p,
    double factor_cor,
    double n,
    double u_max) {

    // 对流换热 - 求解颗粒团贴壁下滑时间
    auto equation_tc = [this, u_max, rho_susp](double x) -> double {
        return u_max * u_max / g * (std::exp(-g * x / u_max) - 1) + u_max * x
            - 0.0178 * std::pow(rho_susp, 0.596);
        };

    double t_c = fsolve(equation_tc, 0.1);

    double rho_c = rho_p * epsilon_p_cl + rho_g * (1 - epsilon_p_cl);
    double cp_c = cp_p * epsilon_p_cl * (rho_p / rho_c) + cp_g * (1 - epsilon_p_cl) * rho_g / rho_c;

    double K_c = K_g * (1 + epsilon_p_cl * (1 - K_g / K_p) /
        (K_g / K_p + 0.28 * std::pow(1 - epsilon_p_cl,
            0.63 * std::pow(K_g / K_p, -0.18))));

    double R_e = std::sqrt(PI * t_c / (4 * K_c * rho_c * cp_c));
    double R_w = dp / (n * K_g);
    double coef_con = 1 / (R_e + R_w);

    // 辐射换热
    double e_cl = 0.5 * (1 - e_p);
    double e_d = std::sqrt(e_p / ((1 - e_p) * B) * (e_p / ((1 - e_p) * B) + 2))
        - e_p / ((1 - e_p) * B);

    double coef_rad = sigma * (T_ann * T_ann + T_w * T_w) * (T_ann + T_w) / (1 / e_cl + 1 / e_w - 1) +
        factor_cor * sigma * (T_cor * T_cor + T_w * T_w) * (T_cor + T_w) /
        (1 / e_d + 1 / e_w - 1);

    // 稀相区&环形区悬挂受热面总传热系数
    Eigen::VectorXd coef_con_vec = X_km * coef_con;
    double coef_dilute = coef_con_vec.sum() + coef_rad;

    return coef_dilute;
}

// 炉内(核心区)悬挂受热面传热系数
double HeatTransfer::coefficient_t(
    const Eigen::VectorXd& X_km,
    double T_cor,
    double T_ann,
    double epsilon_p_c,
    double T_t,
    double mu_g,
    double U_t,
    double cp_g,
    double thermo_g,
    double rho_g,
    double K_g,
    double factor_acor,
    double e_t) {

    // 对流传热
    double rho_dis = epsilon_p_c * rho_p + (1 - epsilon_p_c) * rho_g;
    double Pr = mu_g * cp_g / thermo_g;

    double coef_con_c = K_g / dp * cp_p / cp_g * std::pow(rho_dis / rho_p, 0.3) *
        std::pow(U_t * U_t / (g * dp), 0.21) * Pr;

    // 辐射换热
    double e_cl = 0.5 * (1 - e_p);
    double e_d = std::sqrt(e_p / ((1 - e_p) * B) * (e_p / ((1 - e_p) * B) + 2))
        - e_p / ((1 - e_p) * B);

    double coef_rad_c = sigma * (T_cor * T_cor + T_t * T_t) * (T_cor + T_t) / (1 / e_d + 1 / e_t - 1) +
        factor_acor * sigma * (T_ann * T_ann + T_t * T_t) * (T_ann + T_t) /
        (1 / e_cl + 1 / e_t - 1);

    // 核心区悬挂受热面总传热系数
    Eigen::VectorXd coef_con_c_vec = X_km * coef_con_c;
    double coef_cor_t = coef_con_c_vec.sum() + coef_rad_c;

    return coef_cor_t;
}