#include "cyc_submodel.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

const double PI = 3.14159265358979323846;
const double G_GRAVITY = 9.81;

CycMassEnergy::CycMassEnergy() {
    // Initialize physical constants and parameters
    Vls = std::pow(1.34, 2) / 4 * PI * (20.5 - 8.3);
    rho_p = 1800.0;
    dt = 1.0;
    ain = 3.05;
    bin = 5.49;
    dout = 4.0;
    H_con = 11.4;
    H_up = 4.4;
    D_up = 7.87;
    D_down = 1.34;
    K = 16.0;
    rho_g = 0.3;
    mu_g = 4.5e-5;
    Nc = 4.0;
    n = 0.6;
    D_sp = 1.34;

    // Calculate derived parameters
    Ain = ain * bin;
    d_in = 2.0 * ain * bin / (ain + bin);
    V = PI * std::pow(D_up / 2, 2) * H_up +
        PI / 12 * (std::pow(D_up, 2) + D_up * D_down + std::pow(D_down, 2)) * H_con;

    // Initialize particle diameter array
    dp.resize(12);
    dp << 0.06, 0.09, 0.125, 0.16, 0.2, 0.25, 0.3, 0.4, 0.5, 1.0, 2.0, 3.0;
    dp *= 0.001; // Convert to meters
}

template<typename Func>
Eigen::VectorXd CycMassEnergy::solveNonlinear(Func equations, const Eigen::VectorXd& x0,
    int maxIter, double tol) {
    Eigen::VectorXd x = x0;
    Eigen::VectorXd fx, dx;
    Eigen::MatrixXd J;

    for (int iter = 0; iter < maxIter; ++iter) {
        fx = equations(x);

        if (fx.norm() < tol) {
            break;
        }

        // Numerical Jacobian calculation
        J.resize(fx.size(), x.size());
        double h = 1e-8;

        for (int j = 0; j < x.size(); ++j) {
            Eigen::VectorXd x_plus = x;
            x_plus[j] += h;
            Eigen::VectorXd fx_plus = equations(x_plus);
            J.col(j) = (fx_plus - fx) / h;
        }

        // Solve J * dx = -fx using Eigen's solver
        dx = J.colPivHouseholderQr().solve(-fx);
        x += dx;

        if (dx.norm() < tol) {
            break;
        }
    }

    return x;
}

template<typename Func>
Eigen::VectorXd CycMassEnergy::solveBoundedNonlinear(Func equations, const Eigen::VectorXd& x0,
    const Eigen::VectorXd& lb, const Eigen::VectorXd& ub,
    int maxIter, double tol) {
    Eigen::VectorXd x = x0;

    // Ensure initial guess is within bounds
    for (int i = 0; i < x.size(); ++i) {
        x[i] = std::max(lb[i], std::min(ub[i], x[i]));
    }

    Eigen::VectorXd fx, dx;
    Eigen::MatrixXd J;

    for (int iter = 0; iter < maxIter; ++iter) {
        fx = equations(x);

        if (fx.norm() < tol) {
            break;
        }

        // Numerical Jacobian calculation
        J.resize(fx.size(), x.size());
        double h = 1e-8;

        for (int j = 0; j < x.size(); ++j) {
            Eigen::VectorXd x_plus = x;
            x_plus[j] += h;
            Eigen::VectorXd fx_plus = equations(x_plus);
            J.col(j) = (fx_plus - fx) / h;
        }

        // Solve J * dx = -fx
        dx = J.colPivHouseholderQr().solve(-fx);

        // Apply bounds to the step
        Eigen::VectorXd x_new = x + dx;
        for (int i = 0; i < x_new.size(); ++i) {
            x_new[i] = std::max(lb[i], std::min(ub[i], x_new[i]));
        }

        x = x_new;

        if (dx.norm() < tol) {
            break;
        }
    }

    return x;
}

Eigen::VectorXd CycMassEnergy::cycEff(double Vin, const Eigen::VectorXd& dp, const Eigen::VectorXd& M_outk) {
    // Calculate cyclone separation efficiency
    double d50 = std::sqrt(9 * mu_g * bin / (2 * PI * Nc * (rho_p - rho_g) * (Vin / Ain)));

    Eigen::VectorXd mink = M_outk / (Vin * rho_g);
    Eigen::VectorXd mlimk(dp.size());
    Eigen::VectorXd effk_cyc1(dp.size());
    Eigen::VectorXd effk_cyc2(dp.size());
    Eigen::VectorXd effk_cyc(dp.size());

    for (int i = 0; i < dp.size(); ++i) {
        mlimk[i] = 0.025 * (d50 / dp[i]) * std::exp(0.15 * std::log(10 * mink[i]));

        if (mink[i] > mlimk[i]) {
            effk_cyc1[i] = 1.0 - mlimk[i] / mink[i];
        }
        else {
            effk_cyc1[i] = 0.0;
        }

        effk_cyc2[i] = 1.0 - std::exp(-0.693 * std::pow(dp[i] / d50, 1.0 / (1 + n)));

        if (mink[i] > mlimk[i]) {
            effk_cyc[i] = effk_cyc1[i] + mlimk[i] / mink[i] * effk_cyc2[i];
        }
        else {
            effk_cyc[i] = effk_cyc2[i];
        }
    }

    return effk_cyc;
}

CycMassEnergy::CycMassResult CycMassEnergy::cycMass(double ug, const Eigen::VectorXd& u_tk,
    const Eigen::VectorXd& G_outk,
    const Eigen::VectorXd& effk_cyc, double M_cyc0,
    const Eigen::VectorXd& X_cyc0,
    const Eigen::VectorXd& X_km, double U0, double ksee) {
    double eps = K * ain * bin / (dout * dout);
    double dp_cyc = eps * ug * ug / 2 * rho_g;

    Eigen::VectorXd Re_s = rho_g * ug * dp / mu_g;
    double tao_g = V / (ain * bin * ug);

    Eigen::VectorXd tao_sk(u_tk.size());
    for (int i = 0; i < u_tk.size(); ++i) {
        if (U0 > u_tk[i]) {
            tao_sk[i] = tao_g * 0.032 * std::pow(Re_s[i], 0.43) *
                std::pow((ug - u_tk[i]) / u_tk[i], 0.7) *
                std::pow((rho_p - rho_g) / rho_g, 0.42) *
                std::pow((H_up + H_con) / H_con, -1.76);
        }
        else {
            tao_sk[i] = tao_g;
        }
    }

    Eigen::VectorXd G_flyK = G_outk.cwiseProduct(Eigen::VectorXd::Ones(G_outk.size()) - effk_cyc);

    auto equations_Xcyc = [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
        Eigen::VectorXd x_cyc = x.head(x.size() - 1);
        double M_cyc = x[x.size() - 1];

        Eigen::VectorXd eqs(x.size());

        // Mass conservation equations (12 equations)
        for (int k = 0; k < u_tk.size(); ++k) {
            eqs[k] = G_outk[k] - G_flyK[k] - ksee * M_cyc * x_cyc[k] / tao_sk[k] -
                (M_cyc * x_cyc[k] - M_cyc0 * X_cyc0[k]) / dt;
        }

        // Mass fraction sum constraint
        eqs[x.size() - 1] = x_cyc.sum() - 1.0;

        return eqs;
        };

    // Initial guess
    Eigen::VectorXd x0(X_cyc0.size() + 1);
    x0.head(X_cyc0.size()) = X_cyc0;
    x0[x0.size() - 1] = M_cyc0;

    // Solve system of equations
    Eigen::VectorXd solution = solveNonlinear(equations_Xcyc, x0);

    Eigen::VectorXd X_cyc = solution.head(solution.size() - 1);
    double M_cyc = solution[solution.size() - 1];

    Eigen::VectorXd G_downk(u_tk.size());
    for (int i = 0; i < u_tk.size(); ++i) {
        if (ug > u_tk[i]) {
            G_downk[i] = ksee * M_cyc * X_cyc[i] / tao_sk[i];
        }
        else {
            G_downk[i] = G_outk[i];
        }
    }

    Eigen::VectorXd M_cyck = M_cyc * X_cyc;

    CycMassResult result;
    result.dp_cyc = dp_cyc;
    result.G_downk = G_downk;
    result.G_flyK = G_flyK;
    result.X_cyc = X_cyc;
    result.M_cyck = M_cyck;
    result.M_cyc = M_cyc;

    return result;
}

CycMassEnergy::LoopSealResult CycMassEnergy::loopsealMass(double dp_fur, double dp_cyc, double M_ls0,
    double dp_ls0, double W_re0,
    const Eigen::VectorXd& X_ls0, double e_mfk,
    const Eigen::VectorXd& G_downk,
    double C_ls, double V_ls) {
    auto equations_Mls = [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
        double dp_ls = x[0];
        double W_re = x[1];
        double M_ls = x[2];
        Eigen::VectorXd X_ls = x.tail(x.size() - 3);

        double dp_ls_safe = std::max(dp_ls, 1e-10);
        double W_re_safe = std::max(W_re, 1e-10);

        Eigen::VectorXd eqs(x.size());

        // Return flow rate equation
        eqs[0] = W_re_safe / (0.25 * PI * D_sp * D_sp) - std::sqrt(dp_ls_safe) / C_ls;

        // Pressure balance equation
        eqs[1] = (M_ls - rho_p * (1 - e_mfk) * Vls) * G_GRAVITY / (0.25 * PI * D_sp * D_sp) -
            dp_fur - dp_cyc - dp_ls_safe;

        // Sum constraint
        eqs[2] = X_ls.sum() - 1.0;

        // Mass conservation for each particle size
        for (int k = 0; k < X_ls.size(); ++k) {
            eqs[3 + k] = G_downk[k] - W_re_safe * X_ls[k] - (M_ls * X_ls[k] - M_ls0 * X_ls0[k]) / dt;
        }

        return eqs;
        };

    // Initial guess
    Eigen::VectorXd x0(3 + X_ls0.size());
    x0[0] = dp_ls0;
    x0[1] = W_re0;
    x0[2] = M_ls0;
    x0.tail(X_ls0.size()) = X_ls0;

    // Set bounds
    Eigen::VectorXd lb(x0.size());
    Eigen::VectorXd ub(x0.size());

    lb.setConstant(1e-20);
    ub.head(3).setConstant(std::numeric_limits<double>::infinity());
    ub.tail(X_ls0.size()).setOnes();

    // Solve with bounds
    Eigen::VectorXd result = solveBoundedNonlinear(equations_Mls, x0, lb, ub);

    double dp_ls = result[0];
    double W_re = result[1];
    double M_ls = result[2];
    Eigen::VectorXd X_ls = result.tail(result.size() - 3);

    // Force normalization to prevent numerical errors
    X_ls /= X_ls.sum();

    Eigen::VectorXd W_rek = W_re * X_ls;
    Eigen::VectorXd M_lsk = M_ls * X_ls;

    double dp_sp = dp_fur + dp_cyc + dp_ls;

    LoopSealResult loopResult;
    loopResult.W_rek = W_rek;
    loopResult.M_lsk = M_lsk;
    loopResult.M_ls = M_ls;
    loopResult.X_ls = X_ls;
    loopResult.dp_ls = dp_ls;
    loopResult.W_re = W_re;
    loopResult.dp_sp = dp_sp;

    return loopResult;
}

CycMassEnergy::CycParametersResult CycMassEnergy::calculateCycParameters(
    double ug, const Eigen::VectorXd& u_tk, const Eigen::VectorXd& G_outk,
    const Eigen::VectorXd& effk_cyc, double M_cyc0, const Eigen::VectorXd& X_cyc0,
    const Eigen::VectorXd& X_km, double dp_fur, double M_ls0, double dp_ls0,
    double W_re0, const Eigen::VectorXd& X_ls0, double e_mfk, double U0) {

    CycMassResult cycResult = cycMass(ug, u_tk, G_outk, effk_cyc, M_cyc0, X_cyc0, X_km, U0);

    LoopSealResult loopResult = loopsealMass(dp_fur, cycResult.dp_cyc, M_ls0, dp_ls0, W_re0, X_ls0, e_mfk, cycResult.G_downk);

    CycParametersResult result;
    result.M_cyc = cycResult.M_cyc;
    result.X_cyc = cycResult.X_cyc;
    result.dp_cyc = cycResult.dp_cyc;
    result.M_ls = loopResult.M_ls;
    result.dp_ls = loopResult.dp_ls;
    result.W_re = loopResult.W_re;
    result.X_ls = loopResult.X_ls;
    result.W_rek = loopResult.W_rek;
    result.G_downk = cycResult.G_downk;
    result.G_flyK = cycResult.G_flyK;
    result.M_cyck = cycResult.M_cyck;
    result.M_lsk = loopResult.M_lsk;
    result.dp_sp = loopResult.dp_sp;

    return result;
}