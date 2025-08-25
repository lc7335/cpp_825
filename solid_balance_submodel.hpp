#ifndef SOLID_BALANCE_SUBMODEL_HPP
#define SOLID_BALANCE_SUBMODEL_HPP

#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <memory>

/**
 * 固体质量平衡子模型
 * 采用清华的逻辑
 *
 * 方程数：3种物料*n个控制体（注意：旋风分离器和回料装置也建立质量守恒的控制体）
 * 变量：灰方程的变量为沉降流率，碳和氧化钙方程的变量是二者占灰的质量百分数
 * 不分环核，给小室建立总的质量平衡方程
 */
class SolidMass {
private:
    int Ncell_;     // 小室数
    double dt_;     // 时间步长
    double rho_p_;  // 颗粒密度
    std::vector<double> Vcell_;  // 小室体积

    // 类型定义
    using MatrixXd = Eigen::MatrixXd;
    using VectorXd = Eigen::VectorXd;
    using SparseMatrix = Eigen::SparseMatrix<double>;
    using Triplet = Eigen::Triplet<double>;

    // 内部辅助函数
    std::pair<SparseMatrix, VectorXd> buildLinearSystemXc(
        const std::vector<double>& dp,
        const MatrixXd& M_km,
        const MatrixXd& M_km0,
        const MatrixXd& G_upk,
        const MatrixXd& G_downk,
        const MatrixXd& Gfa,
        const MatrixXd& Gre,
        const MatrixXd& Gd,
        const VectorXd& G_outk,
        const MatrixXd& Rc,
        const MatrixXd& Xfac_km,
        const MatrixXd& Xc_km0,
        const MatrixXd& Gsk,
        const VectorXd& Gcyc_downk,
        const VectorXd& Gcyc_flyk,
        const VectorXd& M_cyck,
        const VectorXd& M_cyck0,
        const VectorXd& W_rek,
        const VectorXd& M_lsk,
        const VectorXd& M_lsk0,
        const MatrixXd& G_tck) const;

    std::pair<SparseMatrix, VectorXd> buildLinearSystemXca(
        const std::vector<double>& dp,
        const MatrixXd& M_km,
        const MatrixXd& M_km0,
        const MatrixXd& G_upk,
        const MatrixXd& G_downk,
        const MatrixXd& Gfa,
        const MatrixXd& Gre,
        const MatrixXd& Gd,
        const VectorXd& G_outk,
        const MatrixXd& Rca,
        const MatrixXd& Xfaca_km,
        const MatrixXd& Xca_km0,
        const MatrixXd& Gsk,
        const VectorXd& Gcyc_downk,
        const VectorXd& Gcyc_flyk,
        const VectorXd& M_cyck,
        const VectorXd& M_cyck0,
        const VectorXd& W_rek,
        const VectorXd& M_lsk,
        const VectorXd& M_lsk0) const;

    VectorXd safeSolve(const SparseMatrix& A, const VectorXd& b) const;
    void cleanNonFiniteValues(SparseMatrix& A, VectorXd& b) const;
    MatrixXd rollMatrix(const MatrixXd& matrix, int shift) const;

public:
    /**
     * 构造函数
     * @param Ncell 小室数
     * @param Vcell 小室体积向量
     * @param rho_p 颗粒密度
     * @param dt 时间步长
     */
    SolidMass(int Ncell, const std::vector<double>& Vcell, double rho_p, double dt);

    /**
     * 计算各小室灰的沉降量,含碳的质量百分数，含氧化钙的质量百分数
     *
     * @param dp 颗粒粒径
     * @param M_km 当前时刻各小室灰质量
     * @param M_km0 上一时刻各小室灰质量
     * @param G_upk K档颗粒各小室上界面向上混合的质量流率
     * @param G_downk K档颗粒各小室下界面向上混合的质量流率
     * @param Gfa 给料速率
     * @param Gre 从返料装置返料速率
     * @param Gd 各小室排渣速率
     * @param G_outk K档颗粒离开主燃室的流率
     * @param Rc k档颗粒在各小室中的碳反应速率
     * @param Xfac_km 给料中碳的质量百分数
     * @param Xc_km0 上一时刻碳的质量分数
     * @param Rca k档颗粒在各小室中的氧化钙反应速率
     * @param Xfaca_km 给料中氧化钙的质量百分数
     * @param Xca_km0 上一时刻氧化钙的质量分数
     * @param Gcyc_downk 旋风分离器k档颗粒当前时刻进入返料装置流率
     * @param Gcyc_flyk 旋风分离器k档颗粒当前时刻飞灰流率
     * @param M_cyck 旋风分离器k档颗粒当前时刻滞留量
     * @param M_cyck0 旋风分离器k档颗粒上一时刻滞留量
     * @param W_rek 返料装置返回k档颗粒流率
     * @param M_lsk 返料装置k档颗粒当前时刻滞留量
     * @param M_lsk0 返料装置k档颗粒上一时刻滞留量
     * @param Xc_km 输出：碳的质量分数
     * @param Xca_km 输出：氧化钙的质量分数
     * @param Gs 输出：各小室的总沉降量
     * @param Gsk 输出：各小室灰的沉降量
     */
    void solidMass(
        const std::vector<double>& dp,
        const MatrixXd& M_km,
        const MatrixXd& M_km0,
        const MatrixXd& G_upk,
        const MatrixXd& G_downk,
        const MatrixXd& Gfa,
        const MatrixXd& Gre,
        const MatrixXd& Gd,
        const VectorXd& G_outk,
        const MatrixXd& Rc,
        const MatrixXd& Xfac_km,
        const MatrixXd& Xc_km0,
        const MatrixXd& Rca,
        const MatrixXd& Xfaca_km,
        const MatrixXd& Xca_km0,
        const VectorXd& Gcyc_downk,
        const VectorXd& Gcyc_flyk,
        const VectorXd& M_cyck,
        const VectorXd& M_cyck0,
        const VectorXd& W_rek,
        const VectorXd& M_lsk,
        const VectorXd& M_lsk0,
        MatrixXd& Xc_km,
        MatrixXd& Xca_km,
        VectorXd& Gs,
        MatrixXd& Gsk) const;

    // 获取参数的接口
    int getNcell() const { return Ncell_; }
    double getDt() const { return dt_; }
    double getRhoP() const { return rho_p_; }
    const std::vector<double>& getVcell() const { return Vcell_; }
};

#endif // SOLID_BALANCE_SUBMODEL_HPP