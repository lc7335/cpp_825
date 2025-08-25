#ifndef STEAM_WATER_SYSTEM_HPP
#define STEAM_WATER_SYSTEM_HPP

#include <vector>
#include <map>
#include <string>
#include <functional>
#include <eigen3/Eigen/Dense>
#include "CoolPropLib.h"

class SteamWaterSystem {
private:
    double R_g;  // Smoke gas constant [J/(kg¡¤K)]
    double relaxation_p;  // Pressure relaxation factor (0 < relaxation_p < 1)
    double relaxation_Tg;  // Smoke temperature relaxation factor (0 < relaxation_Tg < 1)
    double tolerance;  // Convergence tolerance
    int max_iter;  // Maximum iteration count
    double Q_sf;  // Heat transfer from water wall

public:
    // Constructor
    SteamWaterSystem(double R_g_val);

    // Structure to hold system output results
    struct SystemOutput {
        double T_j;  // Water wall temperature
        double Tj_platen;  // Platen superheater wall temperature
        double p_out;  // Steam outlet pressure
        double T_out;  // Steam outlet temperature
        double D_out;  // Steam outlet flow rate
        double h_out;  // Steam outlet enthalpy
        double rho_out;  // Steam outlet density
        double Tg_out;  // Smoke gas outlet temperature
        double p_in_converged;  // Converged inlet pressure
        double Tg_in_converged;  // Converged smoke gas inlet temperature
    };

    struct FlowProperties {
        double p_out;
        double T_out;
        double D_out;
        double h_out;
        double rho_out;
        double Tg_out;
        double Q_qy;
        double Q_mf;
        double T_j;
        double V_l;
        double V_v;
        double Tj_platen;
    };

    struct DrumResults {
        double T_j;
        double p_Pa;
        double T_bh;
        double D_qb;
        double V_l;
        double V_v;
        double h;
    };

    struct SprayResults {
        double p_f2;
        double D_f2;
        double T_f2;
        double h_f2;
    };

    struct SaturatedProperties {
        double h_l;
        double h_v;
        double rho_l;
        double rho_v;
        double T_bh;
    };

    // Main iterative solver
    SystemOutput steam_water_system_iterative(
        double T1_in, double D1_in, double h2_prev_fw, double rho2_prev_fw, double Tg_prev_fw,  // Economizer
        double D_pw, double Q_lb, double T_j_prev, double p_prev_dr, double V_l_prev, double V_v_prev, double D_qb_prev, // Evaporator
        double h2_prev_he1, double rho2_prev_he1, double Tg_prev_he1,  // Low temp superheater
        double D_spray1, double h_spray1, double D_spray12, double h_spray12, // Spray attemperator
        double Q_p_qy, double h2_prev_pl, double rho2_prev_pl,  // Platen superheater
        double h2_prev_he2, double rho2_prev_he2, double Tg_prev_he2,  // High temp superheater
        double D_spray2, double h_spray2, double D_spray22, double h_spray22,  // Spray attemperator
        double p_in_guess,  // Initial guess for inlet pressure [Pa]
        double Tg_out_fw,  // Initial guess for smoke gas outlet temperature [K]
        double p_out_true,  // True outlet pressure [Pa]
        double Tg_in_true   // True smoke gas temperature [K]
    );

    // Total connection function
    SystemOutput steam_water_system(
        double T1_in, double D1_in, double h2_prev_fw, double rho2_prev_fw, double Tg_prev_fw,  // Economizer
        double D_pw, double Q_lb, double T_j_prev, double p_prev_dr, double V_l_prev, double V_v_prev, double D_qb_prev, // Evaporator
        double h2_prev_he1, double rho2_prev_he1, double Tg_prev_he1,  // Low temp superheater
        double D_spray1, double h_spray1, double D_spray12, double h_spray12, // Spray attemperator
        double Q_p_qy, double h2_prev_pl, double rho2_prev_pl,  // Platen superheater
        double h2_prev_he2, double rho2_prev_he2, double Tg_prev_he2,  // High temp superheater
        double D_spray2, double h_spray2, double D_spray22, double h_spray22,  // Spray attemperator
        double p1_in, double Tg_out_fw  // Assumed steam inlet pressure and smoke gas inlet temperature
    );

    // Update smoke gas properties
    std::tuple<double, double, double> update_smoke(double Tg, double v_g);

    // Get saturated water/steam properties
    SaturatedProperties get_saturated_properties(double p_Pa);

    // Water wall temperature calculation
    double water_wall_temperature(double T_j_prev, double Q_lb, double p_Pa,
        double C_j_lb = 490, double C_j_z = 490, double M_b = 5000,
        double M_z = 3000, double alpha_sf = 5.67e-8, double dt = 1);

    // Steam flow calculation
    std::tuple<double, double, double, double> steam_flow(
        double p_prev, double V_l_prev, double V_v_prev, double D_sm, double h_sm,
        double D_pw, double p_Pa, double D_qb_prev,
        double C_j = 579, double M_yx = 8000, double V_total = 10.0, double dt = 1);

    // Run drum simulation
    DrumResults run_drum(double T_j_prev, double Q_lb, double V_l_prev, double V_v_prev,
        double D_sm, double h_sm, double D_pw, double p_Pa, double D_qb_prev, double p_prev);

    // Heat exchanger calculation
    FlowProperties heat_exchanger(double p1, double T1, double D1, double Tg_in,
        double A, double K, double v_g, double h2_prev);

    // Feedwater economizer calculation
    FlowProperties Feedwater_Economizer(double p1, double T1, double D1, double Tg_in,
        double K_A, double v_g, double h2_prev);

    // Platen superheater calculation
    FlowProperties Platen_Superheater(double p1, double T1, double D1, double Q_qy,
        double A, double V, double K, double xi, double Mj_p, double Cj_p,
        double Tj_p_prev, double K2, double dt = 1.0, double h2_prev = 0, double rho2_prev = 0);

    // Spray attemperator calculation
    SprayResults SprayAttemperator(double p_f1, double D_f1, double h_f1,
        double D_spray, double h_spray);

    // Air preheater calculation
    FlowProperties calculate_air_preheater(double p1, double T1, double D1, double Tg_in,
        double V_g, double v_g, double A, double V, double K, double xi, double dt = 1.0,
        double h2_prev = 0, double rho2_prev = 0, double Tg_prev = 0);

private:
    // Helper functions for numerical solving
    std::vector<double> solve_nonlinear_system(
        std::function<std::vector<double>(const std::vector<double>&)> equations,
        const std::vector<double>& initial_guess,
        double tol = 1e-6, int max_iterations = 1000);

    std::vector<double> solve_least_squares(
        std::function<std::vector<double>(const std::vector<double>&)> equations,
        const std::vector<double>& initial_guess,
        const std::vector<double>& lower_bounds = {},
        const std::vector<double>& upper_bounds = {},
        double tol = 1e-6, int max_iterations = 2000);
};

#endif // STEAM_WATER_SYSTEM_HPP