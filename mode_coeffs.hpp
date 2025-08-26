#ifndef MODE_COEFFS_HPP
#define MODE_COEFFS_HPP

#include <Eigen/Dense>
#include <vector>


class ModeCoeffs {
public:
    // ��������ƽ�ⷽ��ϵ������
    static const Eigen::MatrixXi A;  // [35 x 6] �������ϵ��
    static const Eigen::MatrixXi B;  // [35 x 4] CO���ϵ��  
    static const Eigen::MatrixXi C;  // [35 x 6] CO2���ϵ��
    static const Eigen::MatrixXi D;  // [35 x 2] SO2���ϵ��
    static const Eigen::MatrixXi E;  // [35 x 2] NO���ϵ��
    static const Eigen::MatrixXi F;  // [35 x 2] N2���ϵ��
    static const Eigen::MatrixXi G;  // [35 x 1] H2O���ϵ��
    static const Eigen::MatrixXi R;  // [35 x 4] ��Ӧ�������ϵ��

    // ���β���
    static const Eigen::MatrixXd A_cab;  // [35 x 2] ��С�Һ�������m2
    static const Eigen::VectorXd A_t;    // [37] ��С����ʽ���������������m2
    static const Eigen::VectorXd A_w;    // [37] ��С��ˮ��ڴ��������m2
    static const Eigen::MatrixXd h_cell; // [35 x 2] ��С�Ҹ߶�(���½���)��m

private:
    // ��ʼ������
    static Eigen::MatrixXi initA();
    static Eigen::MatrixXi initB();
    static Eigen::MatrixXi initC();
    static Eigen::MatrixXi initD();
    static Eigen::MatrixXi initE();
    static Eigen::MatrixXi initF();
    static Eigen::MatrixXi initG();
    static Eigen::MatrixXi initR();
    static Eigen::MatrixXd initA_cab();
    static Eigen::VectorXd initA_t();
    static Eigen::VectorXd initA_w();
    static Eigen::MatrixXd initH_cell();
};

#endif // MODE_COEFFS_HPP