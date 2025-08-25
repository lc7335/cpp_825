#include "flow_submodel.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <stdexcept>

const double PI = 3.14159265358979323846;

Flow::Flow(const Eigen::VectorXd& dp, double phi, double rho_p, double rho_c,
    double D_bed, double A_bed, double A_plate, double H_bed, double H_out,
    double N_bed, const Eigen::VectorXd& X0, double A_cyc)
    : dp(dp), phi(phi), rho_p(rho_p), rho_c(rho_c), D_bed(D_bed), A_bed(A_bed),
    A_plate(A_plate), H_bed(H_bed), H_out(H_out), N_bed(N_bed), X0(X0), A_cyc(A_cyc) {

    K = dp.size();
    g = 9.81;
    max_iter = 10000;
    Y = 0.8;
    fw = 0.1875;

    // Initialize cyclone model
    cyc_mass_energy = std::make_unique<CycMassEnergy>();
}

template<typename Func>
double Flow::solveScalar(Func equation, double x0, int maxIter, double tol) {
    double x = x0;
    double fx, dfx, dx;
    double h = 1e-8;

    for (int iter = 0; iter < maxIter; ++iter) {
        fx = equation(x);

        if (std::abs(fx) < tol) {
            break;
        }

        // Numerical derivative
        dfx = (equation(x + h) - fx) / h;

        if (std::abs(dfx) < 1e-15) {
            throw std::runtime_error("Zero derivative in scalar solver");
        }

        dx = -fx / dfx;
        x += dx;

        if (std::abs(dx) < tol) {
            break;
        }
    }

    return x;
}

template<typename Func>
Eigen::VectorXd Flow::solveNonlinear(Func equations, const Eigen::VectorXd& x0,
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

        // Solve J * dx = -fx
        dx = J.colPivHouseholderQr().solve(-fx);
        x += dx;

        if (dx.norm() < tol) {
            break;
        }
    }

    return x;
}

double Flow::integrate(std::function<double(double)> func, double a, double b, int n) {
    // Simpson's rule for numerical integration
    if (n % 2 == 1) n++; // Ensure even number of intervals

    double h = (b - a) / n;
    double sum = func(a) + func(b);

    for (int i = 1; i < n; i += 2) {
        sum += 4.0 * func(a + i * h);
    }

    for (int i = 2; i < n; i += 2) {
        sum += 2.0 * func(a + i * h);
    }

    return sum * h / 3.0;
}

Eigen::VectorXd Flow::terminalVelocity(double rho_g, double mu_g) {
    Eigen::VectorXd U_tk(K);

    for (int i = 0; i < K; ++i) {
        auto equation_utk = [&](double U_tk_val) -> double {
            double Re = rho_g * U_tk_val * dp[i] / mu_g;
            double Cd = 24.0 / Re * (1 + 0.15 * std::pow(Re, 0.687)) +
                0.42 / (1 + 4.25e4 * std::pow(Re, -1.16));
            return U_tk_val - std::sqrt(4.0 * (rho_c - rho_g) * g * dp[i] / (3.0 * Cd * rho_g));
            };

        double u_tk_guess = (rho_c - rho_g) * g * dp[i] * dp[i] / (18.0 * mu_g);
        U_tk[i] = solveScalar(equation_utk, u_tk_guess);
    }

    return U_tk;
}

Flow::MinFluidizedResult Flow::minimumFluidizedVelocity(double rho_g, double mu_g) {
    Eigen::VectorXd U_mfk(K);
    Eigen::VectorXd e_mfk(K);

    for (int i = 0; i < K; ++i) {
        double Ar = (rho_g * (rho_c - rho_g) * g * std::pow(dp[i], 3)) / (mu_g * mu_g);
        e_mfk[i] = 0.586 * std::pow(phi, -0.72) * std::pow(Ar, -0.029) *
            std::pow(rho_g / rho_c, 0.021);

        double k1 = 150.0 * (1 - e_mfk[i]) / (3.5 * phi);
        double k2 = std::pow(e_mfk[i], 3) * phi / 1.75;
        double Re_mf = std::sqrt(k1 * k1 + k2 * Ar) - k1;
        U_mfk[i] = (Re_mf * mu_g) / (dp[i] * rho_g);
    }

    return { U_mfk, e_mfk };
}

Flow::FlowSubmodelResult Flow::solveFlowSubmodel(double U0, double U0_den, double Vin,
    double rho_g, double mu_g, double h_den,
    double Wfa, double dP, double eff_bed_out) {
    // Initialize terminal velocity
    Eigen::VectorXd U_tk = terminalVelocity(rho_g, mu_g);

    // Initialize minimum fluidization velocity
    auto mf_result = minimumFluidizedVelocity(rho_g, mu_g);
    Eigen::VectorXd U_mfk = mf_result.U_mfk;
    Eigen::VectorXd e_mfk = mf_result.e_mfk;

    // Initialize arrays
    Eigen::VectorXd G_Hk(K);
    Eigen::VectorXd G_densk = Eigen::VectorXd::Zero(K);
    Eigen::VectorXd e_denk(K);
    Eigen::VectorXd G_TDHk(K);

    // Calculate G_TDHk
    for (int i = 0; i < K; ++i) {
        if (U_tk[i] < U0) {
            G_TDHk[i] = 23.7 * rho_g * U0 * std::exp(-5.4 * U_tk[i] / U0);
        }
        else {
            G_TDHk[i] = 0.0;
        }
    }

    // Calculate void fraction distribution
    Eigen::VectorXd e_TDHk(K);
    for (int i = 0; i < K; ++i) {
        if (U_tk[i] < U0) {
            double term = U0 / U_tk[i] + 1 + G_TDHk[i] / (rho_p * U_tk[i]);
            e_TDHk[i] = 0.5 * term - std::sqrt(0.25 * term * term - U0 / U_tk[i]);
        }
        else {
            e_TDHk[i] = 1.0;
        }
    }

    // Calculate ak
    Eigen::VectorXd ak(K);
    for (int i = 0; i < K; ++i) {
        ak[i] = 200.0 * std::pow(dp[i], 0.572) / U0;  // ·¿µÂÉ½
    }

    Eigen::VectorXd X_km = X0;
    double Wdr = 0.4 * Wfa;

    // Calculate bubble parameters
    Eigen::VectorXd f_d(K);
    Eigen::VectorXd e_bk(K);
    for (int i = 0; i < K; ++i) {
        f_d[i] = 0.24 * (1.1 + 2.9 * std::exp(-3.3 * dp[i])) *
            std::pow(0.15 + U0_den - U_mfk[i], -0.33);
        e_bk[i] = 1.1 / (1 + 1.3 / f_d[i] * std::pow(U0_den - U_mfk[i], -0.8));
        e_denk[i] = (1 - e_bk[i]) * e_mfk[i] + e_bk[i];
    }

    std::cout << "e_denk: ";
    for (int i = 0; i < K; ++i) {
        std::cout << e_denk[i] << " ";
    }
    std::cout << std::endl;

    // Main iteration loop
    for (int i = 0; i < max_iter; ++i) {
        // Inner loop for height calculation
        for (int j = 0; j < max_iter; ++j) {
            // Numerical integration for each particle size
            Eigen::VectorXd integral_value(K);

            for (int ii = 0; ii < K; ++ii) {
                auto integrand = [&](double h_val) -> double {
                    double e_hk = e_TDHk[ii] + (e_denk[ii] - e_TDHk[ii]) * std::exp(-ak[ii] * h_val);
                    return (1 - e_hk) * X_km[ii];
                    };

                integral_value[ii] = integrate(integrand, 0, H_bed - h_den);
            }

            double integral_Value = integral_value.sum();
            double ep_den_ksf = ((Eigen::VectorXd::Ones(K) - e_denk).cwiseProduct(X_km)).sum();
            double h_den_new = (dP - g * rho_p * integral_Value) / (g * rho_p * ep_den_ksf);

            if (j > 0 && std::abs(h_den_new - h_den) < std::max(0.0001, 0.001 * h_den)) {
                break;
            }
            h_den = 0.5 * (h_den_new + h_den);  // Damped update
        }

        // Bubble parameter calculation
        double d_bk = 0.54 / std::pow(g, 0.2) * std::pow(U0 - U_mfk.mean(), 0.4) *
            std::pow(h_den + 4 * std::sqrt(A_bed / N_bed), 0.8);

        for (int k = 0; k < K; ++k) {
            G_densk[k] = 3.07e-9 * A_bed * d_bk * std::pow(rho_g, 3.5) * std::pow(g, 0.5) /
                std::pow(mu_g, 2.5) * std::pow(U0 - U_mfk[k], 2.5);
        }

        // Update outlet flow rates
        for (int k = 0; k < K; ++k) {
            G_Hk[k] = G_TDHk[k] + (G_densk[k] - G_TDHk[k]) *
                std::exp(-ak[k] * (H_out - h_den));
        }

        Eigen::VectorXd Wfak = Wfa * X0;

        // Mass conservation equations
        auto equations_Xkm = [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
            Eigen::VectorXd x_vars = x.head(x.size() - 1);
            double Wdr_local = x[x.size() - 1];

            Eigen::VectorXd G_outk = x_vars.cwiseProduct(G_Hk) * (1 - eff_bed_out) * A_cyc * 2;
            Eigen::VectorXd effk_cyc = cyc_mass_energy->cycEff(Vin, dp, G_outk / 2);

            Eigen::VectorXd eqs(x.size());
            for (int k = 0; k < K; ++k) {
                eqs[k] = x_vars[k] * Wdr_local + G_outk[k] * (1 - effk_cyc[k]) - Wfak[k];
            }
            eqs[K] = x_vars.sum() - 1.0;

            return eqs;
            };

        Eigen::VectorXd x0(K + 1);
        x0.head(K) = X_km;
        x0[K] = Wdr;

        Eigen::VectorXd solution = solveNonlinear(equations_Xkm, x0);
        Eigen::VectorXd X_km_new = solution.head(K);
        double Wdr_new = solution[K];

        if ((X_km_new - X_km).norm() < 0.0001 && std::abs(Wdr - Wdr_new) < 0.0001) {
            break;
        }

        X_km = X_km_new;
        Wdr = Wdr_new;
    }

    // Calculate final results
    double e_den_s = e_denk.dot(X_km);
    Eigen::VectorXd G_outk = X_km.cwiseProduct(G_Hk) * (1 - eff_bed_out) * A_cyc * 2;
    Eigen::VectorXd effk_cyc = cyc_mass_energy->cycEff(Vin, dp, G_outk / 2);

    Eigen::VectorXd G_denk(K);
    for (int k = 0; k < K; ++k) {
        G_denk[k] = (U0 - U_mfk[k]) * A_bed * fw * (1 - e_mfk[k]) * rho_p * X_km[k];
    }

    // Prepare result structure
    FlowSubmodelResult result;
    result.X_km = X_km;
    result.h_den = h_den;
    result.G_denk = G_denk;
    result.G_Hk = G_Hk;
    result.effk_cyc = effk_cyc;
    result.eff_bed_out = eff_bed_out;
    result.G_TDHk = G_TDHk;
    result.G_densk = G_densk;
    result.e_TDHk = e_TDHk;
    result.e_denk = e_denk;
    result.ak = ak;
    result.Wdr = Wdr;
    result.U_mfk = U_mfk;
    result.e_mfk = e_mfk;
    result.U_tk = U_tk;
    result.G_outk = G_outk;

    return result;
}