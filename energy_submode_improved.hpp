#ifndef ENERGY_HPP
#define ENERGY_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

class Energy {
public:
    // 构造函数
    Energy(int N, int N_cyc, double cp_p);

    // 能量守恒方程求解
    struct EnergyResult {
        Eigen::VectorXd T;        // 温度分布
        Eigen::VectorXd q_heat_w; // 炉膛给水冷壁的传热量
        Eigen::VectorXd q_heat_t; // 炉膛给屏式换热器的传热量
    };

    EnergyResult energy_conservation(
        const Eigen::VectorXd& M_p0,      // 上一时刻各小室固体总质量，kg
        const Eigen::VectorXd& M_g0,      // 上一时刻各小室气体总质量，mol
        const Eigen::VectorXd& T0,        // 上一时刻各小室温度，K
        const Eigen::VectorXd& M_p,       // 当前时刻各小室固体总质量，kg
        const Eigen::VectorXd& M_g,       // 当前时刻各小室气体总质量，mol
        const Eigen::VectorXd& W_p,       // 当前时刻各小室固体流率，kg/s
        const Eigen::VectorXd& G_d,       // G_d参数
        const Eigen::VectorXd& G_fly,     // G_fly参数
        const Eigen::VectorXd& W_g,       // 当前时刻各小室气体流率，mol/s
        const Eigen::VectorXd& W_air,     // 给风流率，kg/s
        const Eigen::VectorXd& T_air,     // 给风温度，K
        const Eigen::VectorXd& W_fa,      // 给煤（固体）流率，kg/s
        const Eigen::VectorXd& T_fa,      // 给煤温度，K
        double W_rec_p,                   // 再循环固体流量，kg/s
        const Eigen::VectorXd& E_p,       // 各小室返混流量，kg/s
        const Eigen::VectorXd& heat_combus, // 小室内因燃烧导致的热量变化，W
        const Eigen::VectorXd& U_trans_w,  // 水冷壁传热系数，W/(m²·K)
        const Eigen::VectorXd& U_trans_t,  // 屏式换热器传热系数，W/(m²·K)
        const Eigen::VectorXd& A_w,        // 水冷壁面积，m²
        const Eigen::VectorXd& A_t,        // 屏式换热器面积，m²
        const Eigen::VectorXd& T_w,        // 水冷壁温度，K
        const Eigen::VectorXd& T_t,        // 屏式换热器温度，K
        double dt = 1.0                    // 时间步长，s
    );

private:
    int N;          // 小室总数
    int N_cyc;      // 再循环小室编号
    double cp_p;    // 固体比热容，J/(kg·K)
    double cp_g;    // 气体摩尔热容，J/(mol·K)

    // 建立线性系统的内部函数
    struct LinearSystem {
        Eigen::SparseMatrix<double> A;
        Eigen::VectorXd b;
    };

    LinearSystem build_linear_system_T(
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
        double dt
    );
};

#endif // ENERGY_HPP