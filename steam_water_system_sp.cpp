#include "steam_water_system_sp.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <limits>

SteamWaterSystem::SteamWaterSystem(double R_g_val)
    : R_g(R_g_val), relaxation_p(0.1), relaxation_Tg(0.1), tolerance(1e-2), max_iter(5), Q_sf(0.0) {
}

SteamWaterSystem::SystemOutput SteamWaterSystem::steam_water_system_iterative(
    double T1_in, double D1_in, double h2_prev_fw, double rho2_prev_fw, double Tg_prev_fw,
    double D_pw, double Q_lb, double T_j_prev, double p_prev_dr, double V_l_prev, double V_v_prev, double D_qb_prev,
    double h2_prev_he1, double rho2_prev_he1, double Tg_prev_he1,
    double D_spray1, double h_spray1, double D_spray12, double h_spray12,
    double Q_p_qy, double h2_prev_pl, double rho2_prev_pl,
    double h2_prev_he2, double rho2_prev_he2, double Tg_prev_he2,
    double D_spray2, double h_spray2, double D_spray22, double h_spray22,
    double p_in_guess, double Tg_out_fw, double p_out_true, double Tg_in_true) {

    double p_in = p_in_guess;
    double Tg_out_fw_current = Tg_out_fw;

    for (int iteration = 0; iteration < max_iter; ++iteration) {
        // Call original function to calculate
        SystemOutput system_output = steam_water_system(
            T1_in, D1_in, h2_prev_fw, rho2_prev_fw, Tg_prev_fw,
            D_pw, Q_lb, T_j_prev, p_prev_dr, V_l_prev, V_v_prev, D_qb_prev,
            h2_prev_he1, rho2_prev_he1, Tg_prev_he1,
            D_spray1, h_spray1, D_spray12, h_spray12,
            Q_p_qy, h2_prev_pl, rho2_prev_pl,
            h2_prev_he2, rho2_prev_he2, Tg_prev_he2,
            D_spray2, h_spray2, D_spray22, h_spray22,
            p_in, Tg_out_fw_current
        );

        // Get calculated outlet pressure and smoke gas temperature
        double p_out_calculated = system_output.p_out;
        double Tg_in_calculated = system_output.Tg_out;

        // Calculate errors
        double error_p = std::abs(p_out_calculated - p_out_true);
        double error_Tg = std::abs(Tg_in_calculated - Tg_in_true);

        // Check convergence
        if (error_p < tolerance && error_Tg < tolerance) {
            std::cout << "Converged at iteration " << (iteration + 1) << std::endl;
            system_output.p_in_converged = p_in;
            system_output.Tg_in_converged = Tg_out_fw_current;
            return system_output;
        }

        // Update guess values (relaxation iteration)
        if (error_p >= tolerance) {
            p_in += relaxation_p * (p_out_true - p_out_calculated);
        }

        if (error_Tg >= tolerance) {
            Tg_out_fw_current += relaxation_Tg * (Tg_in_true - Tg_in_calculated);
        }
    }

    std::cout << "Warning: Convergence not reached after " << max_iter << " iterations" << std::endl;

    SystemOutput final_output = steam_water_system(
        T1_in, D1_in, h2_prev_fw, rho2_prev_fw, Tg_prev_fw,
        D_pw, Q_lb, T_j_prev, p_prev_dr, V_l_prev, V_v_prev, D_qb_prev,
        h2_prev_he1, rho2_prev_he1, Tg_prev_he1,
        D_spray1, h_spray1, D_spray12, h_spray12,
        Q_p_qy, h2_prev_pl, rho2_prev_pl,
        h2_prev_he2, rho2_prev_he2, Tg_prev_he2,
        D_spray2, h_spray2, D_spray22, h_spray22,
        p_in, Tg_out_fw_current
    );

    final_output.p_in_converged = p_in;
    final_output.Tg_in_converged = Tg_out_fw_current;
    return final_output;
}

SteamWaterSystem::SystemOutput SteamWaterSystem::steam_water_system(
    double T1_in, double D1_in, double h2_prev_fw, double rho2_prev_fw, double Tg_prev_fw,
    double D_pw, double Q_lb, double T_j_prev, double p_prev_dr, double V_l_prev, double V_v_prev, double D_qb_prev,
    double h2_prev_he1, double rho2_prev_he1, double Tg_prev_he1,
    double D_spray1, double h_spray1, double D_spray12, double h_spray12,
    double Q_p_qy, double h2_prev_pl, double rho2_prev_pl,
    double h2_prev_he2, double rho2_prev_he2, double Tg_prev_he2,
    double D_spray2, double h_spray2, double D_spray22, double h_spray22,
    double p1_in, double Tg_out_fw) {

    // Feedwater economizer
    FlowProperties feedwater_results = Feedwater_Economizer(p1_in, T1_in, D1_in, Tg_out_fw, 32000, 8, h2_prev_fw);

    // Drum
    DrumResults drum_results = run_drum(T_j_prev, Q_lb, V_l_prev, V_v_prev,
        feedwater_results.D_out, feedwater_results.h_out, D_pw, feedwater_results.p_out, D_qb_prev, p_prev_dr);

    // Low temperature superheater
    FlowProperties he_results1 = heat_exchanger(drum_results.p_Pa, drum_results.T_bh, drum_results.D_qb,
        feedwater_results.Tg_out, 520, 80, 9.9, h2_prev_he1);

    // First spray attemperator
    SprayResults spray_results1 = SprayAttemperator(he_results1.p_out, he_results1.D_out, he_results1.h_out,
        D_spray1, h_spray1);

    // Second spray attemperator
    SprayResults spray_results12 = SprayAttemperator(spray_results1.p_f2, spray_results1.D_f2, spray_results1.h_f2,
        D_spray12, h_spray12);

    // Platen superheater
    FlowProperties platen_results = Platen_Superheater(spray_results12.p_f2, spray_results12.T_f2, spray_results12.D_f2,
        Q_p_qy, 396, 2.4, 50, 0.03, 5000, 490, 623.15, 80, 1.0, h2_prev_pl, rho2_prev_pl);

    // Third spray attemperator
    SprayResults spray_results2 = SprayAttemperator(platen_results.p_out, platen_results.D_out, platen_results.h_out,
        D_spray2, h_spray2);

    // Fourth spray attemperator
    SprayResults spray_results22 = SprayAttemperator(spray_results2.p_f2, spray_results2.D_f2, spray_results2.h_f2,
        D_spray22, h_spray22);

    // High temperature superheater
    FlowProperties he_results2 = heat_exchanger(spray_results22.p_f2, spray_results22.T_f2, spray_results22.D_f2,
        he_results1.Tg_out, 450, 80, 11.78, h2_prev_he2);

    SystemOutput output;
    output.T_j = drum_results.T_j;
    output.Tj_platen = platen_results.Tj_platen;
    output.p_out = spray_results2.p_f2;
    output.T_out = spray_results2.T_f2;
    output.D_out = spray_results2.D_f2;
    output.h_out = spray_results2.h_f2;
    output.rho_out = he_results2.rho_out;
    output.Tg_out = he_results2.Tg_out;

    return output;
}

std::tuple<double, double, double> SteamWaterSystem::update_smoke(double Tg, double v_g) {
    double cp_g = 1.05 + 0.0002 * (Tg - 273.15);  // Inlet smoke gas specific heat [J/(kg·K)]
    double rho_g = 101000 * 0.029 / Tg / 8.314;   // Inlet smoke gas density [kg/m³]
    double Dg = rho_g * v_g * 4.52 * 4.52;        // Inlet smoke gas flow rate [kg/s]
    return std::make_tuple(cp_g, rho_g, Dg);
}

SteamWaterSystem::SaturatedProperties SteamWaterSystem::get_saturated_properties(double p_Pa) {
    SaturatedProperties props;
    props.h_l = PropsSI("H", "P", p_Pa, "Q", 0, "Water");
    props.h_v = PropsSI("H", "P", p_Pa, "Q", 1, "Water");
    props.rho_l = PropsSI("D", "P", p_Pa, "Q", 0, "Water");
    props.rho_v = PropsSI("D", "P", p_Pa, "Q", 1, "Water");
    props.T_bh = PropsSI("T", "P", p_Pa, "Q", 1, "Water");
    return props;
}

double SteamWaterSystem::water_wall_temperature(double T_j_prev, double Q_lb, double p_Pa,
    double C_j_lb, double C_j_z, double M_b, double M_z, double alpha_sf, double dt) {

    SaturatedProperties props = get_saturated_properties(p_Pa);
    double T_bh = props.T_bh;

    // Target equation solver using Newton-Raphson method
    auto equation = [&](double T_j) -> double {
        Q_sf = alpha_sf * std::pow(T_j - T_bh, 3);
        return (Q_lb - Q_sf) - ((M_b * C_j_lb + M_z * C_j_z) * (T_j - T_j_prev) / dt);
        };

    auto derivative = [&](double T_j) -> double {
        return -3 * alpha_sf * std::pow(T_j - T_bh, 2) - (M_b * C_j_lb + M_z * C_j_z) / dt;
        };

    double T_j = T_j_prev;
    for (int i = 0; i < 100; ++i) {
        double f = equation(T_j);
        double df = derivative(T_j);

        if (std::abs(f) < 1e-6) break;
        if (std::abs(df) < 1e-12) break;

        T_j = T_j - f / df;
    }

    return T_j;
}

std::tuple<double, double, double, double> SteamWaterSystem::steam_flow(
    double p_prev, double V_l_prev, double V_v_prev, double D_sm, double h_sm,
    double D_pw, double p_Pa, double D_qb_prev, double C_j, double M_yx, double V_total, double dt) {

    SaturatedProperties props = get_saturated_properties(p_Pa);
    SaturatedProperties props_prev = get_saturated_properties(p_prev);

    // Limit D_qb_prev and V_l_prev to reasonable ranges
    D_qb_prev = std::max(0.0, std::min(D_qb_prev, D_sm * 2));
    V_l_prev = std::max(0.1 * V_total, std::min(V_l_prev, 0.9 * V_total));

    // Define equations
    auto equations = [&](const std::vector<double>& vars) -> std::vector<double> {
        double D_qb = vars[0];
        double V_l = vars[1];
        double V_v = V_total - V_l;

        // Energy equation
        double current_energy = (V_l * props.rho_l * props.h_l + V_v * props.rho_v * props.h_v + M_yx * C_j * props.T_bh);
        double prev_energy = (V_l_prev * props_prev.rho_l * props_prev.h_l + V_v_prev * props_prev.rho_v * props_prev.h_v + M_yx * C_j * props_prev.T_bh);
        double eq_energy = (Q_sf + D_sm * h_sm - D_pw * props.h_l - D_qb * props.h_v) - (current_energy - prev_energy) / dt;

        // Mass equation
        double current_mass = V_l * props.rho_l + V_v * props.rho_v;
        double prev_mass = V_l_prev * props_prev.rho_l + V_v_prev * props_prev.rho_v;
        double eq_mass = D_sm - D_pw - (current_mass - prev_mass) / dt - D_qb;

        return { eq_energy, eq_mass };
        };

    std::vector<double> initial_guess = { D_qb_prev, V_l_prev };
    std::vector<double> lower_bounds = { 0, 0 };
    std::vector<double> upper_bounds = { 1e6, V_total };

    std::vector<double> solution = solve_least_squares(equations, initial_guess, lower_bounds, upper_bounds);

    double D_qb = solution[0];
    double V_l = solution[1];
    double V_v = V_total - V_l;

    return std::make_tuple(D_qb, V_l, V_v, props.h_v);
}

SteamWaterSystem::DrumResults SteamWaterSystem::run_drum(double T_j_prev, double Q_lb, double V_l_prev, double V_v_prev,
    double D_sm, double h_sm, double D_pw, double p_Pa, double D_qb_prev, double p_prev) {

    double T_j = water_wall_temperature(T_j_prev, Q_lb, p_Pa);
    auto [D_qb, V_l, V_v, h_v] = steam_flow(p_prev, V_l_prev, V_v_prev, D_sm, h_sm, D_pw, p_Pa, D_qb_prev);
    double T_bh = PropsSI("T", "P", p_Pa, "Q", 1, "Water");

    DrumResults results;
    results.T_j = T_j;
    results.p_Pa = p_Pa;
    results.T_bh = T_bh;
    results.D_qb = D_qb;
    results.V_l = V_l;
    results.V_v = V_v;
    results.h = h_v;

    return results;
}

SteamWaterSystem::FlowProperties SteamWaterSystem::heat_exchanger(double p1, double T1, double D1, double Tg_in,
    double A, double K, double v_g, double h2_prev) {

    double h1 = PropsSI("H", "P", p1, "T", T1, "Water");
    auto [cp_g, rho_g, Dg] = update_smoke(Tg_in, v_g);

    double T2 = T1;
    double D2 = D1;
    double p2 = p1;

    // Define equations for nonlinear solver
    auto equations = [&](const std::vector<double>& vars) -> std::vector<double> {
        double h2 = vars[0];
        double Q_qy = vars[1];
        double Tg_out = vars[2];

        // Calculate LMTD with numerical stability handling
        double deltaT1 = Tg_in - T2;
        double deltaT2 = Tg_out - T1;

        double LMTD;
        if (std::abs(deltaT1 - deltaT2) < 1e-3) {
            LMTD = (deltaT1 + deltaT2) / 2;
        }
        else if (deltaT1 * deltaT2 <= 0) {
            LMTD = (deltaT1 + deltaT2) / 2;
        }
        else {
            try {
                LMTD = (deltaT1 - deltaT2) / std::log(deltaT1 / deltaT2);
            }
            catch (...) {
                LMTD = (deltaT1 + deltaT2) / 2;
            }
        }

        double eq1 = D1 * h1 - D2 * h2 + Q_qy;
        double eq2 = Dg * cp_g * (Tg_in - Tg_out) + Q_qy;
        double eq3 = Q_qy - K * A * LMTD;

        return { eq1, eq2, eq3 };
        };

    std::vector<double> initial_guess = { h2_prev, 1000, Tg_in - 10 };
    std::vector<double> solution = solve_nonlinear_system(equations, initial_guess, 1e-3, 50);

    FlowProperties results;
    results.p_out = p2;
    results.T_out = T2;
    results.D_out = D2;
    results.h_out = solution[0];
    results.Tg_out = solution[2];
    results.Q_qy = solution[1];

    // Calculate outlet density
    results.rho_out = PropsSI("D", "P", p2, "H", solution[0], "Water");

    return results;
}

SteamWaterSystem::FlowProperties SteamWaterSystem::Feedwater_Economizer(double p1, double T1, double D1, double Tg_in,
    double K_A, double v_g, double h2_prev) {

    double h1 = PropsSI("H", "P", p1, "T", T1, "Water");
    auto [cp_g, rho_g, Dg] = update_smoke(Tg_in, v_g);

    double T2 = T1;
    double D2 = D1;
    double p2 = p1;

    auto equations = [&](const std::vector<double>& vars) -> std::vector<double> {
        double h2 = vars[0];
        double Q_qy = vars[1];
        double Tg_out = vars[2];

        double deltaT1 = Tg_in - T2;
        double deltaT2 = Tg_out - T1;

        double LMTD;
        if (std::abs(deltaT1 - deltaT2) < 1e-3) {
            LMTD = (deltaT1 + deltaT2) / 2;
        }
        else if (deltaT1 * deltaT2 <= 0) {
            LMTD = (deltaT1 + deltaT2) / 2;
        }
        else {
            try {
                LMTD = (deltaT1 - deltaT2) / std::log(deltaT1 / deltaT2);
            }
            catch (...) {
                LMTD = (deltaT1 + deltaT2) / 2;
            }
        }

        double eq1 = D1 * h1 - D2 * h2 + Q_qy;
        double eq2 = Dg * cp_g * (Tg_in - Tg_out) + Q_qy;
        double eq3 = Q_qy - K_A * LMTD;

        return { eq1, eq2, eq3 };
        };

    std::vector<double> initial_guess = { h2_prev, 1000, Tg_in - 10 };
    std::vector<double> solution = solve_nonlinear_system(equations, initial_guess, 1e-3, 50);

    FlowProperties results;
    results.p_out = p2;
    results.T_out = T2;
    results.D_out = D2;
    results.h_out = solution[0];
    results.Tg_out = solution[2];
    results.Q_qy = solution[1];
    results.rho_out = PropsSI("D", "P", p2, "H", solution[0], "Water");

    return results;
}

SteamWaterSystem::FlowProperties SteamWaterSystem::Platen_Superheater(double p1, double T1, double D1, double Q_qy,
    double A, double V, double K, double xi, double Mj_p, double Cj_p, double Tj_p_prev, double K2,
    double dt, double h2_prev, double rho2_prev) {

    double h1 = PropsSI("H", "P", p1, "T", T1, "Water");
    double rho1 = PropsSI("D", "P", p1, "T", T1, "Water");
    double p2 = p1 - xi * D1 * D1 / rho1;  // Pressure drop

    auto equations = [&](const std::vector<double>& vars) -> std::vector<double> {
        double h2 = vars[0];
        double rho2 = vars[1];
        double D2 = vars[2];
        double Q_mf = vars[3];
        double T2 = vars[4];
        double Tj_p = vars[5];

        double eq1 = h2_prev * rho2_prev + (dt / V) * (D1 * h1 - D2 * h2 + Q_mf) - rho2 * h2;
        double eq2 = rho2_prev - rho2 + (dt / V) * (D1 - D2);
        double eq3 = rho2 - PropsSI("D", "P", p2, "H", h2, "Water");
        double eq4 = Mj_p * Cj_p * (Tj_p - Tj_p_prev) / dt - Q_qy + Q_mf;
        double eq5 = Q_mf - K2 * A * (Tj_p - T2);
        double eq6 = T2 - PropsSI("T", "P", p2, "H", h2, "Water");

        return { eq1, eq2, eq3, eq4, eq5, eq6 };
        };

    std::vector<double> initial_guess = { h2_prev, rho2_prev, D1 * 0.9, Q_qy * 0.5, T1, Tj_p_prev };
    std::vector<double> solution = solve_nonlinear_system(equations, initial_guess, 1e-3, 50);

    FlowProperties results;
    results.p_out = p2;
    results.T_out = solution[4];
    results.D_out = solution[2];
    results.h_out = solution[0];
    results.rho_out = solution[1];
    results.Q_qy = Q_qy;
    results.Q_mf = solution[3];
    results.Tj_platen = solution[5];

    return results;
}

SteamWaterSystem::SprayResults SteamWaterSystem::SprayAttemperator(double p_f1, double D_f1, double h_f1,
    double D_spray, double h_spray) {

    SprayResults results;
    results.p_f2 = p_f1;  // Assume pressure loss is negligible
    results.D_f2 = D_f1 + D_spray;  // Mass balance
    results.h_f2 = (D_f1 * h_f1 + D_spray * h_spray) / results.D_f2;  // Energy balance
    results.T_f2 = PropsSI("T", "P", results.p_f2, "H", results.h_f2, "Water");

    return results;
}

SteamWaterSystem::FlowProperties SteamWaterSystem::calculate_air_preheater(double p1, double T1, double D1, double Tg_in,
    double V_g, double v_g, double A, double V, double K, double xi, double dt, double h2_prev, double rho2_prev, double Tg_prev) {

    double h1 = PropsSI("H", "P", p1, "T", T1, "Air");
    double rho1 = PropsSI("D", "P", p1, "T", T1, "Air");
    auto [cp_g, rho_g, Dg] = update_smoke(Tg_in, v_g);

    double T2 = T1;

    auto equations = [&](const std::vector<double>& vars) -> std::vector<double> {
        double p2 = vars[0];
        double T2_calc = vars[1];
        double h2 = vars[2];
        double rho2 = vars[3];
        double D2 = vars[4];
        double Q_qy = vars[5];
        double Tg_out = vars[6];

        double deltaT1 = Tg_in - T2_calc;
        double deltaT2 = Tg_out - T1;

        double LMTD;
        if (std::abs(deltaT1 - deltaT2) < 1e-3) {
            LMTD = (deltaT1 + deltaT2) / 2;
        }
        else if (deltaT1 * deltaT2 <= 0) {
            LMTD = (deltaT1 + deltaT2) / 2;
        }
        else {
            try {
                LMTD = (deltaT1 - deltaT2) / std::log(deltaT1 / deltaT2);
            }
            catch (...) {
                LMTD = (deltaT1 + deltaT2) / 2;
            }
        }

        double eq1 = h2_prev * rho2_prev + (dt / V) * (D1 * h1 - D2 * h2 + Q_qy) - rho2 * h2;
        double eq2 = V_g * cp_g * rho_g * (Tg_out - Tg_prev) / dt - Dg * cp_g * (Tg_in - Tg_out) + Q_qy;
        double eq3 = rho2_prev - rho2 + (dt / V) * (D1 - D2);
        double eq4 = Q_qy - K * A * LMTD;
        double eq5 = h2 - PropsSI("H", "P", p2, "T", T2_calc, "Air");
        double eq6 = rho2 - PropsSI("D", "P", p2, "T", T2_calc, "Air");
        double eq7 = p1 - p2 - xi * D1 * D1 / rho1;

        return { eq1, eq2, eq3, eq4, eq5, eq6, eq7 };
        };

    std::vector<double> initial_guess = { p1 * 0.9, T1 + 10, h2_prev, rho2_prev, D1 * 0.9, 1000, Tg_in - 10 };
    std::vector<double> solution = solve_nonlinear_system(equations, initial_guess, 1e-3, 50);

    FlowProperties results;
    results.p_out = solution[0];
    results.T_out = solution[1];
    results.D_out = solution[4];
    results.h_out = solution[2];
    results.rho_out = solution[3];
    results.Tg_out = solution[6];
    results.Q_qy = solution[5];

    return results;
}

// Helper function to solve nonlinear systems using Newton-Raphson method
std::vector<double> SteamWaterSystem::solve_nonlinear_system(
    std::function<std::vector<double>(const std::vector<double>&)> equations,
    const std::vector<double>& initial_guess, double tol, int max_iterations) {

    std::vector<double> x = initial_guess;
    int n = x.size();
    double h = 1e-8;

    for (int iter = 0; iter < max_iterations; ++iter) {
        std::vector<double> f = equations(x);

        // Check convergence
        double norm = 0;
        for (double val : f) {
            norm += val * val;
        }
        if (std::sqrt(norm) < tol) {
            return x;
        }

        // Calculate Jacobian numerically
        Eigen::MatrixXd J(n, n);
        for (int j = 0; j < n; ++j) {
            std::vector<double> x_plus = x;
            x_plus[j] += h;
            std::vector<double> f_plus = equations(x_plus);

            for (int i = 0; i < n; ++i) {
                J(i, j) = (f_plus[i] - f[i]) / h;
            }
        }

        // Solve J * dx = -f
        Eigen::VectorXd f_vec(n);
        for (int i = 0; i < n; ++i) {
            f_vec(i) = f[i];
        }

        Eigen::VectorXd dx = J.colPivHouseholderQr().solve(-f_vec);

        // Update x
        for (int i = 0; i < n; ++i) {
            x[i] += dx(i);
        }
    }

    return x;  // Return current solution even if not converged
}

// Helper function for least squares problems with bounds
std::vector<double> SteamWaterSystem::solve_least_squares(
    std::function<std::vector<double>(const std::vector<double>&)> equations,
    const std::vector<double>& initial_guess,
    const std::vector<double>& lower_bounds,
    const std::vector<double>& upper_bounds,
    double tol, int max_iterations) {

    std::vector<double> x = initial_guess;
    int n = x.size();
    double h = 1e-8;
    double lambda = 1e-3;  // Levenberg-Marquardt parameter

    for (int iter = 0; iter < max_iterations; ++iter) {
        std::vector<double> f = equations(x);
        int m = f.size();

        // Check convergence
        double norm = 0;
        for (double val : f) {
            norm += val * val;
        }
        if (std::sqrt(norm) < tol) {
            return x;
        }

        // Calculate Jacobian numerically
        Eigen::MatrixXd J(m, n);
        for (int j = 0; j < n; ++j) {
            std::vector<double> x_plus = x;
            x_plus[j] += h;
            std::vector<double> f_plus = equations(x_plus);

            for (int i = 0; i < m; ++i) {
                J(i, j) = (f_plus[i] - f[i]) / h;
            }
        }

        // Levenberg-Marquardt step
        Eigen::MatrixXd JtJ = J.transpose() * J;
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);

        Eigen::VectorXd f_vec(m);
        for (int i = 0; i < m; ++i) {
            f_vec(i) = f[i];
        }

        Eigen::VectorXd dx = (JtJ + lambda * I).colPivHouseholderQr().solve(-J.transpose() * f_vec);

        // Update x with bounds checking
        for (int i = 0; i < n; ++i) {
            x[i] += dx(i);

            // Apply bounds if provided
            if (!lower_bounds.empty() && x[i] < lower_bounds[i]) {
                x[i] = lower_bounds[i];
            }
            if (!upper_bounds.empty() && x[i] > upper_bounds[i]) {
                x[i] = upper_bounds[i];
            }
        }

        // Adaptive lambda
        std::vector<double> f_new = equations(x);
        double norm_new = 0;
        for (double val : f_new) {
            norm_new += val * val;
        }

        if (norm_new < norm) {
            lambda /= 10;
        }
        else {
            lambda *= 10;
        }
    }

    return x;  // Return current solution even if not converged
}