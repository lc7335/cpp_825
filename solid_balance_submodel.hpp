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
 * ��������ƽ����ģ��
 * �����廪���߼�
 *
 * ��������3������*n�������壨ע�⣺����������ͻ���װ��Ҳ���������غ�Ŀ����壩
 * �������ҷ��̵ı���Ϊ�������ʣ�̼�������Ʒ��̵ı����Ƕ���ռ�ҵ������ٷ���
 * ���ֻ��ˣ���С�ҽ����ܵ�����ƽ�ⷽ��
 */
class SolidMass {
private:
    int Ncell_;     // С����
    double dt_;     // ʱ�䲽��
    double rho_p_;  // �����ܶ�
    std::vector<double> Vcell_;  // С�����

    // ���Ͷ���
    using MatrixXd = Eigen::MatrixXd;
    using VectorXd = Eigen::VectorXd;
    using SparseMatrix = Eigen::SparseMatrix<double>;
    using Triplet = Eigen::Triplet<double>;

    // �ڲ���������
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
     * ���캯��
     * @param Ncell С����
     * @param Vcell С���������
     * @param rho_p �����ܶ�
     * @param dt ʱ�䲽��
     */
    SolidMass(int Ncell, const std::vector<double>& Vcell, double rho_p, double dt);

    /**
     * �����С�һҵĳ�����,��̼�������ٷ������������Ƶ������ٷ���
     *
     * @param dp ��������
     * @param M_km ��ǰʱ�̸�С�һ�����
     * @param M_km0 ��һʱ�̸�С�һ�����
     * @param G_upk K��������С���Ͻ������ϻ�ϵ���������
     * @param G_downk K��������С���½������ϻ�ϵ���������
     * @param Gfa ��������
     * @param Gre �ӷ���װ�÷�������
     * @param Gd ��С����������
     * @param G_outk K�������뿪��ȼ�ҵ�����
     * @param Rc k�������ڸ�С���е�̼��Ӧ����
     * @param Xfac_km ������̼�������ٷ���
     * @param Xc_km0 ��һʱ��̼����������
     * @param Rca k�������ڸ�С���е������Ʒ�Ӧ����
     * @param Xfaca_km �����������Ƶ������ٷ���
     * @param Xca_km0 ��һʱ�������Ƶ���������
     * @param Gcyc_downk ���������k��������ǰʱ�̽��뷵��װ������
     * @param Gcyc_flyk ���������k��������ǰʱ�̷ɻ�����
     * @param M_cyck ���������k��������ǰʱ��������
     * @param M_cyck0 ���������k��������һʱ��������
     * @param W_rek ����װ�÷���k����������
     * @param M_lsk ����װ��k��������ǰʱ��������
     * @param M_lsk0 ����װ��k��������һʱ��������
     * @param Xc_km �����̼����������
     * @param Xca_km ����������Ƶ���������
     * @param Gs �������С�ҵ��ܳ�����
     * @param Gsk �������С�һҵĳ�����
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

    // ��ȡ�����Ľӿ�
    int getNcell() const { return Ncell_; }
    double getDt() const { return dt_; }
    double getRhoP() const { return rho_p_; }
    const std::vector<double>& getVcell() const { return Vcell_; }
};

#endif // SOLID_BALANCE_SUBMODEL_HPP