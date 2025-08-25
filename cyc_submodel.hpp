#ifndef CYC_SUBMODEL_H
#define CYC_SUBMODEL_H

#include <Eigen/Dense>
#include <vector>
#include <functional>

class CycMassEnergy {
private:
    // Physical constants and parameters
    double Vls;
    double rho_p;
    double dt;
    double ain;
    double bin;
    double dout;
    double H_con;
    double H_up;
    double D_up;
    double D_down;
    double K;
    double rho_g;
    double mu_g;
    double Nc;
    double n;
    double D_sp;

    // Derived parameters
    double Ain;
    double d_in;
    double V;

    // Particle diameter array
    Eigen::VectorXd dp;

    // Helper function for nonlinear equation solving
    template<typename Func>
    Eigen::VectorXd solveNonlinear(Func equations, const Eigen::VectorXd& x0,
        int maxIter = 1000, double tol = 1e-10);

    template<typename Func>
    Eigen::VectorXd solveBoundedNonlinear(Func equations, const Eigen::VectorXd& x0,
        const Eigen::VectorXd& lb, const Eigen::VectorXd& ub,
        int maxIter = 1000, double tol = 1e-10);

public:
    // Constructor
    CycMassEnergy();

    // Main calculation functions
    Eigen::VectorXd cycEff(double Vin, const Eigen::VectorXd& dp, const Eigen::VectorXd& M_outk);

    struct CycMassResult {
        double dp_cyc;
        Eigen::VectorXd G_downk;
        Eigen::VectorXd G_flyK;
        Eigen::VectorXd X_cyc;
        Eigen::VectorXd M_cyck;
        double M_cyc;
    };

    CycMassResult cycMass(double ug, const Eigen::VectorXd& u_tk, const Eigen::VectorXd& G_outk,
        const Eigen::VectorXd& effk_cyc, double M_cyc0, const Eigen::VectorXd& X_cyc0,
        const Eigen::VectorXd& X_km, double U0, double ksee = 1.0);

    struct LoopSealResult {
        Eigen::VectorXd W_rek;
        Eigen::VectorXd M_lsk;
        double M_ls;
        Eigen::VectorXd X_ls;
        double dp_ls;
        double W_re;
        double dp_sp;
    };

    LoopSealResult loopsealMass(double dp_fur, double dp_cyc, double M_ls0, double dp_ls0,
        double W_re0, const Eigen::VectorXd& X_ls0, double e_mfk,
        const Eigen::VectorXd& G_downk, double C_ls = 0.02, double V_ls = 1.0);

    struct CycParametersResult {
        double M_cyc;
        Eigen::VectorXd X_cyc;
        double dp_cyc;
        double M_ls;
        double dp_ls;
        double W_re;
        Eigen::VectorXd X_ls;
        Eigen::VectorXd W_rek;
        Eigen::VectorXd G_downk;
        Eigen::VectorXd G_flyK;
        Eigen::VectorXd M_cyck;
        Eigen::VectorXd M_lsk;
        double dp_sp;
    };

    CycParametersResult calculateCycParameters(double ug, const Eigen::VectorXd& u_tk,
        const Eigen::VectorXd& G_outk, const Eigen::VectorXd& effk_cyc,
        double M_cyc0, const Eigen::VectorXd& X_cyc0,
        const Eigen::VectorXd& X_km, double dp_fur,
        double M_ls0, double dp_ls0, double W_re0,
        const Eigen::VectorXd& X_ls0, double e_mfk, double U0);
};

#endif // CYC_SUBMODEL_H