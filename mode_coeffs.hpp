#ifndef MODE_COEFFS_HPP
#define MODE_COEFFS_HPP

#include <Eigen/Dense>
#include <vector>


class ModeCoeffs {
public:
    // 气体质量平衡方程系数矩阵
    static const Eigen::MatrixXi A;  // [35 x 6] 氧气相关系数
    static const Eigen::MatrixXi B;  // [35 x 4] CO相关系数  
    static const Eigen::MatrixXi C;  // [35 x 6] CO2相关系数
    static const Eigen::MatrixXi D;  // [35 x 2] SO2相关系数
    static const Eigen::MatrixXi E;  // [35 x 2] NO相关系数
    static const Eigen::MatrixXi F;  // [35 x 2] N2相关系数
    static const Eigen::MatrixXi G;  // [35 x 1] H2O相关系数
    static const Eigen::MatrixXi R;  // [35 x 4] 反应速率相关系数

    // 几何参数
    static const Eigen::MatrixXd A_cab;  // [35 x 2] 各小室横截面积，m2
    static const Eigen::VectorXd A_t;    // [37] 各小室屏式过热器传热面积，m2
    static const Eigen::VectorXd A_w;    // [37] 各小室水冷壁传热面积，m2
    static const Eigen::MatrixXd h_cell; // [35 x 2] 各小室高度(上下界面)，m

private:
    // 初始化函数
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