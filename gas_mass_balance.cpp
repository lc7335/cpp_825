#include "gas_mass_balance.hpp"
#include <functional>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>
const double PI = 3.14159265358979323846;

// 构造函数实现
GAS::GAS(int N, double dt, const Eigen::VectorXd& dp,
    double rho_p, double rho_c, int N_vm, double H_bed)
    : N(N), dt(dt), dp(dp), rho_p(rho_p), rho_c(rho_c),
    N_vm(N_vm), H_bed(H_bed) {

    // 初始化摩尔质量
    molar_mass.resize(6);
    molar_mass << 16, 2, 28, 12.913, 18, 44;

    // 初始化系数矩阵
    coefficients.resize(6, 3);
    coefficients << 0.241, -0.469, 0.201,
        1.388, -0.868, 0.157,
        4.845, -2.653, 0.428,
        -12.887, 7.279, -0.325,
        4.554, -2.389, 0.409,
        1.906, -0.9, 0.135;

    // 初始化热值
    heat_value_stdmol.resize(4);
    heat_value_stdmol << 802.3, 285.8, 283.0, 530.6;
}

// 总气体质量平衡计算
GAS::GasBalanceResult GAS::total_gas_mass_balance(
    const Eigen::VectorXd& M_g_pre,
    const Eigen::VectorXd& M_g,
    const Eigen::MatrixXd& Y_gas_pre_input,
    const Eigen::VectorXd& G_gas_pre_input,
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
    const Eigen::MatrixXd& h_cell) {

    GasBalanceResult result;

    // 初始化结果向量
    result.heat_cabs = Eigen::VectorXd::Zero(N);
    result.G_gas = Eigen::VectorXd::Zero(N);
    result.R_char = Eigen::MatrixXd::Zero(N, dp.size());
    result.R_ca = Eigen::MatrixXd::Zero(N, dp.size());
    result.Y_o2 = Eigen::VectorXd::Zero(N);
    result.Y_co = Eigen::VectorXd::Zero(N);
    result.Y_co2 = Eigen::VectorXd::Zero(N);
    result.Y_so2 = Eigen::VectorXd::Zero(N);
    result.Y_no = Eigen::VectorXd::Zero(N);
    result.Y_n2 = Eigen::VectorXd::Zero(N);
    result.Y_h2o = Eigen::VectorXd::Zero(N);

    // 复制输入以便修改
    result.Y_gas_pre = Y_gas_pre_input;
    result.G_gas_pre = G_gas_pre_input;

    Eigen::VectorXd Y_gas_up = Eigen::VectorXd::Zero(7);
    double G_gas_up = 0.0;

    // 从最后一个小室开始计算
    for (int i = N - 1; i >= 0; i--) {
        double t = T(i);
        double p = P(i);
        double v_cab = v_cabs(i);
        Eigen::VectorXd y_gas_pre = result.Y_gas_pre.row(i).transpose();
        double g_gas_pre = result.G_gas_pre(i);
        double m_g_pre = M_g_pre(i);
        double m_g = M_g(i);
        double x_vm = X_VM(i);
        double m_S = M_S(i);
        double n_mass = n_Mass(i);
        double q_air = Q_air(i);

        Eigen::VectorXd a = A_coeffs.row(i).transpose();
        Eigen::VectorXd b = B_coeffs.row(i).transpose();
        Eigen::VectorXd c = C_coeffs.row(i).transpose();
        Eigen::VectorXd d = D_coeffs.row(i).transpose();
        Eigen::VectorXd e = E_coeffs.row(i).transpose();
        Eigen::VectorXd f = F_coeffs.row(i).transpose();
        Eigen::VectorXd g = G_coeffs.row(i).transpose();
        Eigen::VectorXd r = R_coeffs.row(i).transpose();

        GasCabResult gas_result = gas_cab_constant(
            t, p, v_cab, e_k.row(i).transpose(), e_k_ksf(i), e_mfk,
            X_km, u_tk, carbon_content.row(i).transpose(),
            y_gas_pre, g_gas_pre, m_g_pre, m_g, x_vm, m_S, n_mass, W_fa,
            Y_gas_up, G_gas_up, q_air, Xca_km.row(i).transpose(),
            mu_g, rho_g, h_cell.row(i).transpose(),
            a, b, c, d, e, f, g, r
        );

        Y_gas_up = gas_result.Y_gas_current;
        G_gas_up = gas_result.G_gas_current;
        result.Y_gas_pre.row(i) = gas_result.Y_gas_pre.transpose();
        result.G_gas_pre(i) = gas_result.G_gas_pre;

        result.Y_o2(i) = gas_result.Y_gas_current(0);
        result.Y_co(i) = gas_result.Y_gas_current(1);
        result.Y_co2(i) = gas_result.Y_gas_current(2);
        result.Y_so2(i) = gas_result.Y_gas_current(3);
        result.Y_no(i) = gas_result.Y_gas_current(4);
        result.Y_n2(i) = gas_result.Y_gas_current(5);
        result.Y_h2o(i) = gas_result.Y_gas_current(6);

        result.G_gas(i) = gas_result.G_gas_current;
        result.heat_cabs(i) = gas_result.heat;
        result.R_char.row(i) = gas_result.r_char.transpose();
        result.R_ca.row(i) = gas_result.r_ca.transpose();
    }

    return result;
}

// 单个小室气体质量平衡模型
GAS::GasCabResult GAS::gas_cab_constant(
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
    double ratio_o2, double ratio_n2) {

    GasCabResult result;

    // 计算挥发分燃烧
    VMCombustionResult result_vm = combustion_vm(x_vm, W_fa, h_cell);

    // 定义方程组
    auto equations_gas = [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
        Eigen::VectorXd Y_gas = x.head(7);
        double G_gas = x(7);

        // 计算NO生成
        Eigen::VectorXd poly_coeffs(4);
        poly_coeffs << -2.8412e-4, 0.01364, -0.3063, 15.756;
        calculate_no_generation(x_vm, n_mass, poly_coeffs);

        // 计算各种燃烧反应
        CharCombustionResult char_result = combustion_char(
            t, p, v_cab, e_k, e_k_ksf, e_mfk, X_km, u_tk,
            carbon_content, Y_gas, mu_g, rho_g
        );

        SO2CombustionResult so2_result = combustion_so2(
            t, p, v_cab, Y_gas, Xca_km, e_k, X_km
        );

        NOCombustionResult no_result = combustion_no(
            t, v_cab, carbon_content, Y_gas, e_k
        );

        Eigen::VectorXd eqs(8);

        // 质量守恒方程
        double f_o2 = ((M_g_pre * Y_gas_pre(0) - M_g * Y_gas(0)) / dt +
            G_gas_up * Y_gas_up(0) - G_gas * Y_gas(0) -
            a_coeffs(0) * reaction_char_o2 - a_coeffs(1) * result_vm.delta_O2 -
            a_coeffs(2) * reaction_co_o2 + a_coeffs(3) * Q_air * ratio_o2 -
            a_coeffs(4) * reaction_so2_o2 - a_coeffs(5) * m_S);

        double f_co = ((M_g_pre * Y_gas_pre(1) - M_g * Y_gas(1)) / dt +
            G_gas_up * Y_gas_up(1) - G_gas * Y_gas(1) -
            b_coeffs(0) * reaction_co + b_coeffs(1) * reaction_co2_co +
            b_coeffs(2) * reaction_char_co - b_coeffs(3) * reaction_no_co);

        double f_co2 = ((M_g_pre * Y_gas_pre(2) - M_g * Y_gas(2)) / dt +
            G_gas_up * Y_gas_up(2) - G_gas * Y_gas(2) +
            c_coeffs(0) * reaction_co_co2 + c_coeffs(1) * result_vm.delta_CO2 +
            c_coeffs(2) * reaction_char_co2 - c_coeffs(3) * reaction_co2 +
            c_coeffs(4) * reaction_so2_co2 + c_coeffs(5) * reaction_no_co2);

        double f_so2 = ((M_g_pre * Y_gas_pre(3) - M_g * Y_gas(3)) / dt +
            G_gas_up * Y_gas_up(3) - G_gas * Y_gas(3) +
            d_coeffs(0) * m_S - d_coeffs(1) * reaction_so2);

        double f_no = ((M_g_pre * Y_gas_pre(4) - M_g * Y_gas(4)) / dt +
            G_gas_up * Y_gas_up(4) - G_gas * Y_gas(4) +
            e_coeffs(0) * m_NO - e_coeffs(1) * reaction_no);

        double f_n2 = ((M_g_pre * Y_gas_pre(5) - M_g * Y_gas(5)) / dt +
            G_gas_up * Y_gas_up(5) - G_gas * Y_gas(5) +
            f_coeffs(0) * reaction_no_n2 + f_coeffs(1) * Q_air * ratio_n2);

        double f_h2o = ((M_g_pre * Y_gas_pre(6) - M_g * Y_gas(6)) / dt +
            G_gas_up * Y_gas_up(6) - G_gas * Y_gas(6) +
            g_coeffs(0) * result_vm.delta_H2O);

        double f_sum = Y_gas.sum() - 1.0;

        eqs << f_o2, f_co, f_co2, f_so2, f_no, f_n2, f_h2o, f_sum;

        return eqs;
        };

    // 初始猜测
    Eigen::VectorXd x0(8);
    x0.head(7) = Y_gas_pre;
    x0(7) = G_gas_pre;

    // 边界约束
    Eigen::VectorXd lb(8);
    lb.head(7).setConstant(0.0);
    lb(7) = 1e-10;

    Eigen::VectorXd ub(8);
    ub.head(7).setConstant(1.0);
    ub(7) = std::numeric_limits<double>::infinity();

    // 求解非线性方程组
    Eigen::VectorXd solution = solve_nonlinear_equations(equations_gas, x0, lb, ub);

    result.Y_gas_current = solution.head(7);
    result.G_gas_current = solution(7);
    result.Y_gas_pre = result.Y_gas_current;
    result.G_gas_pre = result.G_gas_current;

    // 重新计算以获取热量等结果
    CharCombustionResult char_final = combustion_char(
        t, p, v_cab, e_k, e_k_ksf, e_mfk, X_km, u_tk,
        carbon_content, result.Y_gas_current, mu_g, rho_g
    );

    SO2CombustionResult so2_final = combustion_so2(
        t, p, v_cab, result.Y_gas_current, Xca_km, e_k, X_km
    );

    NOCombustionResult no_final = combustion_no(
        t, v_cab, carbon_content, result.Y_gas_current, e_k
    );

    result.heat = r_coeffs(0) * result_vm.total_heat_vm +
        r_coeffs(1) * char_final.total_heat +
        r_coeffs(2) * so2_final.total_heat +
        r_coeffs(3) * no_final.total_heat;

    result.r_char = char_final.reaction_rate;
    result.r_ca = so2_final.reaction_rate_ca;
    result.single_char = char_final.single_char;
    result.d_o2_char = char_final.d_o2_char;
    result.d_o2_co = char_final.d_o2_co;
    result.d_o2_vm = result_vm.delta_O2;
    result.d_o2_so2 = so2_final.delta_o2;

    return result;
}

// NO生成计算
void GAS::calculate_no_generation(double x_vm, double n_mass,
    const Eigen::VectorXd& poly_coeffs,
    double ratio_o2_no) {
    double z = 100 * x_vm - 30.637;
    double fitted_ratio = 0.0;
    for (int i = 0; i < poly_coeffs.size(); i++) {
        fitted_ratio += poly_coeffs(i) * std::pow(z, poly_coeffs.size() - 1 - i);
    }
    fitted_ratio = fitted_ratio * ratio_o2_no / 100;
    m_NO = n_mass * fitted_ratio;
}

// 焦炭燃烧计算
GAS::CharCombustionResult GAS::combustion_char(
    double t, double p, double v_cab,
    const Eigen::VectorXd& e_k, double e_k_ksf, double e_mfk,
    const Eigen::VectorXd& X_km, double u_tk,
    const Eigen::VectorXd& carbon_content,
    const Eigen::VectorXd& Y_gas,
    double mu_g, double rho_g) {

    CharCombustionResult result;

    double T_char = t + 66 * Y_gas(0) * p / (R * t);
    double T_mean = (t + T_char) / 2;
    double CO_ratio = 2512 * std::exp(-6240 / t);

    // 计算机械因子
    Eigen::VectorXd M_factor = _calculate_mechanical_factor(CO_ratio);

    // 扩散系数
    double diffusivity = D_ref * std::pow(t / T_ref, 1.5) * (P_ref / p);

    // 舍伍德数计算
    double schmidt_num = mu_g / (rho_g * diffusivity);
    Eigen::VectorXd reynolds_num = u_tk * dp.array() * rho_g / mu_g;
    Eigen::VectorXd sherwood_num = 2.0 + 0.064 * (reynolds_num.array() / e_mfk).sqrt() *
        std::pow(schmidt_num, 1.0 / 3.0);

    // 焦炭反应系数计算
    Eigen::VectorXd k_diff = sherwood_num.array() * diffusivity / dp.array();
    Eigen::VectorXd k_react = Eigen::VectorXd::Constant(dp.size(),
        constant * T_mean * std::exp(-energy_ratio / T_char));
    Eigen::VectorXd k_total = (k_react.array().inverse() +
        (M_factor.array() * k_diff.array()).inverse()).inverse();

    // CO2反应系数
    double k_co2 = 4.1e8 * std::exp(-2.478e8 / (R * 1000 * t));

    // CO反应速率
    double k_co = 3.0e8 * std::pow(p / (R * 1000 * t), 1.8) * std::exp(-8056 / t) *
        std::sqrt(Y_gas(6)) * Y_gas(1) * 17.5 * Y_gas(0) / (1 + 24.7 * Y_gas(0));

    // 焦炭总反应速率
    result.single_char = PI * dp.array().square() * k_total.array() * p / (R * t) * Y_gas(0);

    Eigen::VectorXd a_m = (6 * rho_p * (1 - e_k.array()).abs() * X_km.array() /
        (dp.array() * rho_c));

    result.reaction_rate = a_m.array() * v_cab * k_total.array() * p / (R * t) *
        carbon_content.array() * Y_gas(0);

    reaction_char_o2 = (result.reaction_rate.array() / M_factor.array()).sum();
    reaction_char_co = ((2 - 2.0 / M_factor.array()) * result.reaction_rate.array()).sum();
    reaction_char_co2 = ((2.0 / M_factor.array() - 1) * result.reaction_rate.array()).sum();

    // CO2反应
    Eigen::VectorXd reaction_rate_co2 = a_m.array() * v_cab * k_co2 * p / (R * t) *
        carbon_content.array() * Y_gas(2);
    reaction_co2_co = (2 * reaction_rate_co2.array()).sum();
    reaction_co2 = reaction_rate_co2.sum();

    // CO反应
    double reaction_rate_co = v_cab * e_k_ksf * k_co;
    reaction_co = reaction_rate_co;
    reaction_co_o2 = reaction_rate_co * 0.5;
    reaction_co_co2 = reaction_rate_co;

    // 产物计算
    result.delta_O2 = -reaction_char_o2 - reaction_co_o2;
    result.delta_CO = reaction_char_co + reaction_co2_co - reaction_co;
    result.delta_CO2 = reaction_char_co2 - reaction_co2 + reaction_co_co2;

    double heat_c = (reaction_char_co2 * heat_co2 + reaction_char_co * heat_co) * 1000;
    double heat_co2_r = heat_c_co2 * reaction_co2 * 1000;
    double heat_co_r = heat_c_co * reaction_rate_co * 1000;
    result.total_heat = heat_c + heat_co2_r + heat_co_r;

    result.heat_c = heat_c;
    result.heat_co2 = heat_co2_r;
    result.heat_co = heat_co_r;
    result.M_factor = M_factor;
    result.k_total = k_total;
    result.k_react = k_react;
    result.k_diff = M_factor.array() * k_diff.array();
    result.k_co2 = k_co2;
    result.k_co = k_co;
    result.d_o2_char = reaction_char_o2;
    result.d_o2_co = reaction_co_o2;

    return result;
}

// 机械因子计算
Eigen::VectorXd GAS::_calculate_mechanical_factor(double CO_ratio) {
    Eigen::VectorXd M_factor(dp.size());

    for (int i = 0; i < dp.size(); i++) {
        if (dp(i) <= small_threshold) {
            M_factor(i) = (2 * CO_ratio + 2) / (CO_ratio + 2 + 1e-8);
        }
        else if (dp(i) <= large_threshold) {
            M_factor(i) = (2 * CO_ratio + 2 - (CO_ratio / 0.00095) *
                (dp(i) - small_threshold)) / (CO_ratio + 2 + 1e-8);
        }
        else {
            M_factor(i) = 1.0;
        }
    }

    return M_factor;
}

// SO2燃烧计算
GAS::SO2CombustionResult GAS::combustion_so2(
    double t, double p, double v_cab,
    const Eigen::VectorXd& Y_gas,
    const Eigen::VectorXd& Xca_km,
    const Eigen::VectorXd& e_k,
    const Eigen::VectorXd& X_km) {

    SO2CombustionResult result;

    // 计算表面积修正因子
    double s_g = (t > 1253) ? (35.9 * t - 3.67e4) : (-38.4 * t + 5.6e4);

    // 计算反应速率常数
    double kvl = 490 * std::exp(-7.33e7 / (R * t)) * s_g * coefficients_caco3;

    // 计算SO2反应速率
    result.reaction_rate_ca = v_cab * (1 - e_k.array()).abs() * X_km.array() *
        Xca_km.array() * kvl * Y_gas(3) * p / (R * t);

    double reaction_rate_so2 = result.reaction_rate_ca.sum();
    reaction_so2 = reaction_rate_so2;
    reaction_so2_o2 = 0.5 * reaction_rate_so2;
    reaction_so2_co2 = reaction_rate_so2;

    result.delta_so2 = -reaction_rate_so2;
    result.delta_o2 = 0.5 * reaction_rate_so2;
    result.delta_co2 = reaction_rate_so2;
    result.total_heat = heat_so2 * reaction_rate_so2 * 1000;

    return result;
}

// NO燃烧计算
GAS::NOCombustionResult GAS::combustion_no(
    double t, double v_cab,
    const Eigen::VectorXd& carbon_content,
    const Eigen::VectorXd& Y_gas,
    const Eigen::VectorXd& e_k) {

    NOCombustionResult result;

    // 计算反应速率常数
    double k1_no = k1_no0 * std::exp(-q1_cof / t);
    double k2_no = k2_no0 * std::exp(-q2_cof / t);

    // 计算颗粒数量密度
    Eigen::VectorXd n_p = 6 * v_cab * (1 - e_k.array()) / (PI * dp.array().cube());

    // 计算单个颗粒表面积
    Eigen::VectorXd particle_area = PI * dp.array().square();

    // 焦炭表面反应速率
    double coke_rate = (n_p.array() * rho_p / rho_c * carbon_content.array() /
        (1 - carbon_content.array()) * particle_area.array() * k1_no *
        std::pow(Y_gas(4), no_order) * std::pow(Y_gas(1), m1_order)).sum();

    // Al2O3表面反应速率
    double al2o3_rate = (al2o3_ratio * n_p.array() * particle_area.array() * k2_no *
        std::pow(Y_gas(4), no_order) * std::pow(Y_gas(1), m2_order)).sum();

    double reaction_rate_no = coke_rate + al2o3_rate;
    reaction_no = reaction_rate_no;
    reaction_no_co = reaction_rate_no;
    reaction_no_co2 = reaction_rate_no;
    reaction_no_n2 = reaction_rate_no * 0.5;

    result.delta_no = -reaction_rate_no;
    result.delta_co = -reaction_rate_no;
    result.delta_n2 = reaction_rate_no * 0.5;
    result.delta_co2 = reaction_rate_no;
    result.total_heat = (heating_no_consumption + heating_no_generation) *
        reaction_rate_no * 1000;

    return result;
}

// 挥发分燃烧计算
GAS::VMCombustionResult GAS::combustion_vm(double x_vm, double W_fa,
    const Eigen::VectorXd& h_cell) {
    VMCombustionResult result;

    // 计算各成分质量
    Eigen::VectorXd fitted_mass(molar_mass.size());
    for (int j = 0; j < molar_mass.size(); j++) {
        double coeff_result = 0.0;
        for (int k = 0; k < 3; k++) {
            coeff_result += coefficients(j, k) * std::pow(x_vm, 3 - k - 1);
        }
        fitted_mass(j) = coeff_result * x_vm * W_fa * 1000 / H_bed *
            (h_cell(0) - h_cell(1));
    }

    // 计算氧气消耗
    result.delta_O2 = fitted_mass(0) / molar_mass(0) * 2 +
        fitted_mass(1) / molar_mass(1) * 0.5 +
        fitted_mass(2) / molar_mass(2) * 0.5 +
        fitted_mass(3) / molar_mass(3) * 1.165;

    // 计算CO2生成
    result.delta_CO2 = fitted_mass(0) / molar_mass(0) +
        fitted_mass(2) / molar_mass(2) +
        fitted_mass(3) / molar_mass(3) +
        fitted_mass(5) / molar_mass(5);

    // 计算H2O生成
    result.delta_H2O = fitted_mass(0) / molar_mass(0) * 2 +
        fitted_mass(1) / molar_mass(1) +
        fitted_mass(3) / molar_mass(3) * 0.345 +
        fitted_mass(4) / molar_mass(4);

    // 计算总放热量
    result.total_heat_vm = 0;
    for (int i = 0; i < molar_mass.size() - 2; i++) {
        result.total_heat_vm += fitted_mass(i) / molar_mass(i) *
            heat_value_stdmol(i) * 1000;
    }

    return result;
}

// 非线性方程求解器（简化版本，使用牛顿迭代法）
Eigen::VectorXd GAS::solve_nonlinear_equations(
    const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& func,
    const Eigen::VectorXd& x0,
    const Eigen::VectorXd& lb,
    const Eigen::VectorXd& ub) {

    Eigen::VectorXd x = x0;
    const double tol = 1e-8;
    const int max_iter = 10000;
    const double eps = 1e-8;

    for (int iter = 0; iter < max_iter; iter++) {
        // 确保x在边界内
        for (int i = 0; i < x.size(); i++) {
            x(i) = std::max(lb(i), std::min(ub(i), x(i)));
        }

        Eigen::VectorXd f = func(x);

        // 检查收敛
        if (f.norm() < tol) {
            break;
        }

        // 计算雅可比矩阵（数值微分）
        Eigen::MatrixXd J(f.size(), x.size());
        for (int j = 0; j < x.size(); j++) {
            Eigen::VectorXd x_plus = x;
            x_plus(j) += eps;
            Eigen::VectorXd f_plus = func(x_plus);
            J.col(j) = (f_plus - f) / eps;
        }

        // 牛顿步
        Eigen::VectorXd dx = -J.fullPivLu().solve(f);

        // 线搜索
        double alpha = 1.0;
        while (alpha > 1e-10) {
            Eigen::VectorXd x_new = x + alpha * dx;

            // 确保在边界内
            bool in_bounds = true;
            for (int i = 0; i < x_new.size(); i++) {
                if (x_new(i) < lb(i) || x_new(i) > ub(i)) {
                    in_bounds = false;
                    break;
                }
            }

            if (in_bounds) {
                Eigen::VectorXd f_new = func(x_new);
                if (f_new.norm() < f.norm()) {
                    x = x_new;
                    break;
                }
            }

            alpha *= 0.5;
        }
    }

    return x;
}