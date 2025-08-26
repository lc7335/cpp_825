#ifndef HEAT_TRANSFER_HPP
#define HEAT_TRANSFER_HPP

#include <vector>
#include <tuple>
#include <functional>
#include <Eigen/Dense>
const double PI = 3.14159265358979323846;
class HeatTransfer {
private:
    // ��������
    double cp_p;    // ���������ѹ������
    double rho_p;   // ����������ܶȣ�kg/m3
    double dp;      // ����ֱ��
    int N;          // ��С����
    int N_t;        // ��ʽ������С����

    // ������
    double e_den;   // ���㷢���ʣ�0.8~0.95
    double e_w;     // ���淢���ʣ�0.8~0.9
    double e_p;     // �������������
    double g;       // �������ٶ�
    double sigma;   // ˹�ٷ�-������������
    double B;       // ����ϵ��������ͬ�Է���B=0.5��������B=0.667

    // ��ֵ����� - �򵥵�ţ�ٷ�ʵ��fsolve����
    double fsolve(std::function<double(double)> func, double x0,
        double tol = 1e-8, int max_iter = 100);

public:
    // ���캯��
    HeatTransfer(double cp_p, double rho_p, double dp, int N, int N_t);

    // �����򣺼����С����ˮ��ں���ʽ�������Ĵ���ϵ��
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

    // ������&���û���������ϵ��
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

    // ϡ����&���������������洫��ϵ��
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

    // ¯��(������)���������洫��ϵ��
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