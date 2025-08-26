#ifndef HEAT_TRANSFER_HPP
#define HEAT_TRANSFER_HPP

#include <vector>
#include <tuple>
#include <functional>
#include <Eigen/Dense>
const double PI = 3.14159265358979323846;
class HeatTransfer {
private:
    // 材料属性
    double cp_p;    // 固体颗粒定压比热容
    double rho_p;   // 固体颗粒的密度，kg/m3
    double dp;      // 颗粒直径
    int N;          // 总小室数
    int N_t;        // 屏式换热器小室数

    // 物理常数
    double e_den;   // 床层发射率，0.8~0.95
    double e_w;     // 壁面发射率，0.8~0.9
    double e_p;     // 固体颗粒发射率
    double g;       // 重力加速度
    double sigma;   // 斯蒂芬-玻尔兹曼常数
    double B;       // 反射系数：各向同性反射B=0.5，漫反射B=0.667

    // 数值求解器 - 简单的牛顿法实现fsolve功能
    double fsolve(std::function<double(double)> func, double x0,
        double tol = 1e-8, int max_iter = 100);

public:
    // 构造函数
    HeatTransfer(double cp_p, double rho_p, double dp, int N, int N_t);

    // 主程序：计算各小室与水冷壁和屏式换热器的传热系数
    std::tuple<std::vector<double>, std::vector<double>> heat_transfer_cabs(
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
        double rho_g
    );

    // 密相区&外置换热器传热系数
    double coefficient_den(
        const Eigen::VectorXd& X_km,
        double T_w,
        double u0,
        double u_mf,
        double epsilon_mf,
        double T_den,
        double cp_g,
        double thermo_g,
        double rho_g,
        double thermo_p = 1.0,
        double pipe_factor = 0.9
    );

    // 稀相区&环形区悬挂受热面传热系数
    double coefficient_dil(
        const Eigen::VectorXd& X_km,
        double T_w,
        double T_cor,
        double T_ann,
        double epsilon_p_cl,
        double rho_susp,
        double rho_g,
        double cp_g,
        double K_g,
        double K_p = 1.0,
        double factor_cor = 0.5,
        double n = 2.5,
        double u_max = 1.26
    );

    // 炉内(核心区)悬挂受热面传热系数
    double coefficient_t(
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
        double factor_acor = 0.5,
        double e_t = 0.87
    );
};

#endif // HEAT_TRANSFER_HPP