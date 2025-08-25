#ifndef FLOW_SUBMODEL_H
#define FLOW_SUBMODEL_H

#include <Eigen/Dense>
#include <vector>
#include <functional>
#include <memory>
#include "cyc_submodel.hpp"

class Flow {
private:
    // Physical parameters
    Eigen::VectorXd dp;      // Particle diameter array
    double phi;              // Particle sphericity
    double rho_p;            // Ash density (kg/m³)
    double rho_c;            // Carbon density (kg/m³)
    double D_bed;            // Bed equivalent diameter
    double A_bed;            // Bed cross-sectional area
    double A_plate;          // Plate area
    double H_bed;            // Bed height
    double H_out;            // Outlet height
    double N_bed;            // Number of bed nozzles
    Eigen::VectorXd X0;      // Initial component fractions
    double A_cyc;            // Cyclone area

    // Derived parameters
    int K;                   // Number of particle size classes
    double g;                // Gravitational acceleration
    int max_iter;            // Maximum iterations
    double Y;                // Bubble flow ratio
    double fw;               // Wake volume fraction

    // Cyclone model instance
    std::unique_ptr<CycMassEnergy> cyc_mass_energy;

    // Helper function for nonlinear equation solving
    template<typename Func>
    Eigen::VectorXd solveNonlinear(Func equations, const Eigen::VectorXd& x0,
        int maxIter = 10000, double tol = 1e-6);

    template<typename Func>
    double solveScalar(Func equation, double x0, int maxIter = 1000, double tol = 1e-10);

    // Numerical integration function
    double integrate(std::function<double(double)> func, double a, double b, int n = 1000);

public:
    // Constructor
    Flow(const Eigen::VectorXd& dp, double phi, double rho_p, double rho_c,
        double D_bed, double A_bed, double A_plate, double H_bed, double H_out,
        double N_bed, const Eigen::VectorXd& X0, double A_cyc);

    // Destructor
    ~Flow() = default;

    // Main calculation functions
    Eigen::VectorXd terminalVelocity(double rho_g, double mu_g);

    struct MinFluidizedResult {
        Eigen::VectorXd U_mfk;
        Eigen::VectorXd e_mfk;
    };
    MinFluidizedResult minimumFluidizedVelocity(double rho_g, double mu_g);

    struct FlowSubmodelResult {
        Eigen::VectorXd X_km;        // Mass fractions
        double h_den;                // Dense phase height
        Eigen::VectorXd G_denk;      // Dense phase mass flux
        Eigen::VectorXd G_Hk;        // Height-dependent mass flux
        Eigen::VectorXd effk_cyc;    // Cyclone efficiency
        double eff_bed_out;          // Bed outlet efficiency
        Eigen::VectorXd G_TDHk;      // TDH mass flux
        Eigen::VectorXd G_densk;     // Dense phase specific mass flux
        Eigen::VectorXd e_TDHk;      // TDH void fraction
        Eigen::VectorXd e_denk;      // Dense phase void fraction
        Eigen::VectorXd ak;          // Decay constant
        double Wdr;                  // Return flow rate
        Eigen::VectorXd U_mfk;       // Minimum fluidization velocity
        Eigen::VectorXd e_mfk;       // Minimum fluidization void fraction
        Eigen::VectorXd U_tk;        // Terminal velocity
        Eigen::VectorXd G_outk;      // Outlet mass flux
    };

    FlowSubmodelResult solveFlowSubmodel(double U0, double U0_den, double Vin,
        double rho_g, double mu_g, double h_den,
        double Wfa, double dP, double eff_bed_out = 0.2);
};

#endif // FLOW_SUBMODEL_H