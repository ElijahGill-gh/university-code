/*
 * Final Project for PHYM004 Computational Physics and Modelling
 * Title: Smooth Particle Hydrodynamics
 * Author: Elijah Gill
 * Due Date: 27/03/2026
 */

/*
 * Program description: 1D Smooth Particle Hydrodynamics (SPH) code to be used to
 *                      model a shock tube.
 *
 *
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>

// Conditional Bools
bool toggle_boundary_particles;         // Toggles the addition of unmoving boundary particles on either end of the tube
bool toggle_artificial_viscosity;       // Toggles the use of artificial viscosity
bool toggle_variable_h;                 // Toggle the use of a constant or variable smoothing length
bool export_bool;                       // Set to 'true' to allow exporting data and 'false' to not allow

// Global Constants
float length;                           // Shock tube length
float Gamma;                            // Adiabatic Index
double h_const;                         // Smoothing length
float Alpha;                            // Artificial Viscosity Constant
float Beta;                             // Artificial Viscosity Constant
float eta;                              // Variable Smoothing Length Constant
float ep;                               // Constant to stop division by zero
int export_every;                       // How many iterations between exporting data to a text file
std::string txtfile = "DATA.txt";       // Name of the text file to append data to
std::string inputfile = "INPUTS.txt";   // Name of the inputs file containing model parameters
float delta;                            // Time-step
float t_start;                          // Start time
float t_end;                            // End time
int num_particles;                      // Number of particles in system
int boundary_particles;                 // Number of boundary particles on each side of the tube
static const int max_line = 1000;       // Maximum number of characters the code will read in a file before flagging an error
const int max_iter = 500;                // Maximum number of iterations before cutting off convergence
const double tolerance = 1e-4;          // Tolerance value to meet to end convergence

// Initial system parameters
double rho_1;                           // Density in the left tube
double P_1;                             // Pressure in the left tube
double rho_2;                           // Density in the right tube
double P_2;                             // Pressure in the right tube

// ============================================================================================================================
//                                                Declaring functions and classes
// ============================================================================================================================
// Descriptions of functions can be found in the 'Functions' section as part of each function.
// Declare functions that don't rely on the System class
void read_inputs();
double avg(double A, double B);
double smoothing_kernel(double v, double h);
double grad_smoothing_kernel(double v, double h);
double density(double mass, double pos_a, double pos_b, double h);
double eos_ideal(double density, double u, float gamma);
double variable_h(float eta, double mass, double density);
double sound_speed(double P, double density, float gamma);
double find_u(float gamma, double P, double density);
double acceleration(double mass, double Pa, double Pb, double rhoa, double rhob, double arti_visc, double posa, double posb, double h);
double power(double mass, double Pa, double rhoa, double arti_visc, double vela, double velb, double posa, double posb, double h);
double artificial_viscosity(double cs_a, double cs_b, double rhoa, double rhob, double posa, double posb, double vela, double velb, double avg_h);

// System Class
class ParticleSystem {
// Public class to contain information of each particle in the system
public:
    std::vector<double> pos;     // x positions
    std::vector<double> vel;     // x velocities
    std::vector<double> acc;     // x accelerations
    std::vector<double> rho;     // densities
    std::vector<double> P;       // pressures
    std::vector<double> u;       // Specific internal energies
    std::vector<double> powr;    // power
    std::vector<bool> boundary;  // Boundary condition particle label
    std::vector<double> h;       // Smoothing lengths
    std::vector<double> cs;      // Speeds of sound
    std::vector<double> sigma;   // Grad-h-term for each particle
    double mass;                 // Mass of each particle
    double tot_E;                // Total energy of the system
    double tot_mom;              // Total linear momentum of the system

    // Initialise system
    void initialise(int N, float rho1, float rho2, float P1, float P2, float L) {
        /* Find the total number of particles to model (more if boundary condition is active)
        N : Number of particles in the system
        rho1 : Density on the left half of the tube
        rho2 : Density on the right half of the tube
        P1 : Pressure on the left half of the tube
        P2 : Pressure on the right half of the tube
        L : Total length of the tube
        */
        int total_N = 0;
        if (toggle_boundary_particles == 1) {
            total_N = N + 2*boundary_particles;
        } else {
            total_N = N;
            boundary_particles = 0;
        }
        // Make all the vectors the correct size
        pos.resize(total_N, 0.0);
        vel.resize(total_N, 0.0);
        acc.resize(total_N, 0.0);
        rho.resize(total_N, 0.0);
        P.resize(total_N, 0.0);
        u.resize(total_N, 0.0);
        powr.resize(total_N, 0.0);
        boundary.resize(total_N, false);
        h.resize(total_N, h_const);
        cs.resize(total_N, 0.0);
        sigma.resize(total_N, 1.0);

        // Split particle number into density ratio
        int half_1 = N * rho1/(rho1 + rho2);
        int half_2 = N - half_1;

        // Calculate spacing in each half of the tube
        float dx1 = L/(2*half_1);
        float dx2 = L/(2*half_2);

        // Define mass based on the smaller spacing
        if (rho2 > rho1) {
            mass = rho2 * dx2;
        } else {
            mass = rho1 * dx1;
        }

        if (toggle_boundary_particles == 1) {
            // Add boundary particles on the left of the tube
            for (int i=0; i<boundary_particles; i++) {
                pos[i] = (-L - dx1)/2 - i*dx1;
                rho[i] = rho1;
                P[i] = P1;
                u[i] = find_u(Gamma, P[i], rho[i]);
                boundary[i] = true;
            }
        }
        // Fill the left side of the tube
        for (int i=0; i<half_1; i++) {
            int index = i + boundary_particles;
            pos[index] = (-L + dx1)/2 + i*dx1;
            rho[index] = rho1;
            P[index] = P1;
            u[index] = find_u(Gamma, P[index], rho[index]);
        }
        // Fill the right side of the tube
        for (int i=0; i<half_2; i++) {
            int index = i + boundary_particles + half_1;
            pos[index] = dx2/2 + i*dx2;
            rho[index] = rho2;
            P[index] = P2;
            u[index] = find_u(Gamma, P[index], rho[index]);
        }
        if (toggle_boundary_particles == 1) {
        // Add boundary particles on the right of the tube
            for (int i=0; i<boundary_particles; i++) {
                int index = i + boundary_particles + N;
                pos[index] = (L + dx2)/2 + i*dx2;
                rho[index] = rho2;
                P[index] = P2;
                u[index] = find_u(Gamma, P[index], rho[index]);
                boundary[index] = true;
            }
        }
        // Add variable smoothing lengths and grad-h-terms if toggled
        if (toggle_variable_h == 1) {
            h_const = eta * dx1;
            for (int i=0; i<total_N; i++) {
                h[i] = variable_h(eta, mass, rho[i]);
            }
        }
        // Calculate total initial energy and linear momentum
        tot_E = total_energy(total_N);
        tot_mom = total_momentum(total_N);

        // Print a message
        std::cout << "Initialisation Complete! Created " << pos.size() << " particles. (" << pos.size()-(2*boundary_particles) << " real, " << 2*boundary_particles << " boundary)\n";
    }

    void find_params() {
        // Finds the densities, pressures and speeds of sound for each particle in the system
        // Find size of system
        int N = pos.size();
        // Find Densities and Pressures
        for (int i=0; i<N; i++) {
            if (boundary[i] == true) continue;      // Ignore boundary particles
            if (toggle_variable_h == 1) {
                h[i] = variable_h(eta, mass, rho[i]);
            }
            rho[i] = 0.0;
            for (int j=0; j<N; j++) {
                rho[i] += density(mass, pos[i], pos[j], h[i]);
            }
            // Calculate pressure from new density
            P[i] = eos_ideal(rho[i], u[i], Gamma);
            // Calculate the speeds of sound
            cs[i] = sound_speed(P[i], rho[i], Gamma);
        }
        // Find Acceleration and Power
        for (int i=0; i<N; i++) {
            double temp_acc = 0.0;
            double temp_powr = 0.0;
            for (int j=0; j<N; j++) {
                double avg_h = avg(h[i],h[j]);
                double arti_visc = 0.0;
                if (toggle_artificial_viscosity == 1) {
                    arti_visc = artificial_viscosity(cs[i], cs[j], rho[i], rho[j], pos[i], pos[j], vel[i], vel[j], avg_h);
                }
                // Accumulate acceleration and power values with a brute force search
                temp_acc += acceleration(mass, P[i], P[j], rho[i], rho[j], arti_visc, pos[i], pos[j], avg_h);
                temp_powr += power(mass, P[i], rho[i], arti_visc, vel[i], vel[j], pos[i], pos[j], avg_h);
            }

            // Once brute force search is completed, update the acceleration and power variables for each particle
            acc[i] = temp_acc;
            powr[i] = temp_powr;
        }
    }

    double total_energy(int N) {
        // Returns the total sum of kinetic and specific internal energy of each real particle in the system
        // N : Number of particles in the tube, including boundary particles
        double sum_E = 0;
        for (int i=0; i<N; i++) {
            if (boundary[i] == 1) continue;     // Ignore boundary particles
            sum_E += mass*(0.5*std::pow(vel[i],2) + u[i]);
        }
        return sum_E;
    }

    double total_momentum(int N) {
        // Returns the total sum of linear momentum of each real particle in the system
        // N : Number of particles in the tube, including boundary particles
        double sum_mom = 0;
        for (int i=0; i<N; i++) {
            if (boundary[i] == 1) continue;     // Ignore boundary particles
            sum_mom += mass*vel[i];
        }
        return sum_mom;
    }
};

// Declare functions that rely on the System class
void verlet_step(ParticleSystem& sys, float dt, int N);
void verlet(ParticleSystem& sys, float dt, float t_start, float t_end, bool export_bool, int export_every);
void export_data(std::string filename, bool restart, const ParticleSystem& sys, int N, float time);
void progress_bar(int step, int total_steps);

// ============================================================================================================================
//                                                             Main Code
// ============================================================================================================================
int main() {
    // Create some UI
    std::cout << "==================================================================================\n";
    std::cout << "===================  Smooth Particle Hydrodynamics Shock Tube  ===================\n";
    std::cout << "==================================================================================\n";
    std::cout << "Parameters are set in the 'INPUTS.txt' file.\nChange these parameters in the text file and rerun the code for different results.\n\n";
    // Read in parameters from a text file
    read_inputs();
    // User check inputs are good or not

    // Initialise system
    ParticleSystem shock_tube_sys;
    shock_tube_sys.initialise(num_particles, rho_1, rho_2, P_1, P_2, length);

    // Initialise the text file to append data to
    export_data(txtfile, true, shock_tube_sys, num_particles + 2*boundary_particles, 0.0);

    // Run Verlet integrator and append values to a text file
    verlet(shock_tube_sys, delta, t_start, t_end, export_bool, export_every);

    return 0;
}
// ============================================================================================================================
//                                                             Functions
// ============================================================================================================================

void read_inputs() {
    // Set the parameters for the model by reading the values given in the 'INPUTS.exe' file, then print them to the terminal.
    // Find the inputs file
    std::ifstream file("INPUTS.txt");
    // If file isn't opened flag an error and exit program
    if (file.is_open() != 1) {
        std::cerr << "Could not open 'INPUTS.txt' file!" << std::endl;
        std::exit(1);
    }
    if (file.is_open()) {
        // Read Condition Booleans
        file.ignore(max_line, ':') >> std::boolalpha >> toggle_boundary_particles;
        file.ignore(max_line, ':') >> std::boolalpha >> toggle_artificial_viscosity;
        file.ignore(max_line, ':') >> std::boolalpha >> toggle_variable_h;
        file.ignore(max_line, ':') >> std::boolalpha >> export_bool;
        // Read Tube Parameters
        file.ignore(max_line, ':') >> length;
        file.ignore(max_line, ':') >> Gamma;
        file.ignore(max_line, ':') >> h_const;
        file.ignore(max_line, ':') >> Alpha;
        file.ignore(max_line, ':') >> Beta;
        file.ignore(max_line, ':') >> eta;
        file.ignore(max_line, ':') >> ep;
        file.ignore(max_line, ':') >> num_particles;
        file.ignore(max_line, ':') >> boundary_particles;
        // Read Particle Parameters
        file.ignore(max_line, ':') >> rho_1;
        file.ignore(max_line, ':') >> P_1;
        file.ignore(max_line, ':') >> rho_2;
        file.ignore(max_line, ':') >> P_2;
        // Read Verlet Parameters
        file.ignore(max_line, ':') >> delta;
        file.ignore(max_line, ':') >> t_start;
        file.ignore(max_line, ':') >> t_end;
        file.ignore(max_line, ':') >> export_every;

        file.close();
    }
    // Print read values for each parameter to the terminal
    std::cout << "Current Parameters:\n";
    std::cout << "Boolean Conditions (1 -> On, 0 -> Off):\n";
    std::cout << "+-----------------------+-------------------------+------------------------------+----------------+\n";
    std::cout << "| Boundary Particles: " << toggle_boundary_particles << " | Artificial Viscosity: " << toggle_artificial_viscosity << " | Variable Smoothing Length: " << toggle_variable_h << " | Export Data: " << export_bool << " |\n";
    std::cout << "+-----------------------+-------------------------+------------------------------+----------------+\n";
    std::cout << "Tube Parameters:\n";
    std::cout << "+----------------+------------+---------------------------------------+----------+---------+----------+\n";
    std::cout << "| Tube Length: " << length << " | Gamma: " << Gamma << " | Smoothing Length (if constant): " << h_const << " | Alpha: " << Alpha << " | Beta: " << Beta << " | Eta: " << eta << " |\n";
    std::cout << "+----------------+------------+---------------------------------------+----------+---------+----------+\n";
    std::cout << "+--------------+--------------------------+----------------------------------+\n";
    std::cout << "| Epsilon: " << ep << " | Number of Particles: " << num_particles << " | Number of Boundary Particles: " << boundary_particles << " |\n";
    std::cout << "+--------------+--------------------------+----------------------------------+\n";
    std::cout << "Particle Parameters:\n";
    std::cout << "+----------------------+-----------------------+--------------------------+-----------------------------+\n";
    std::cout << "| Left Tube Density: " << rho_1 << " | Left Tube Pressure: " << P_1 << " | Right Tube Density: " << rho_2 << " | Right Tube Pressure: " << P_2 << " |\n";
    std::cout << "+----------------------+-----------------------+--------------------------+-----------------------------+\n";
    std::cout << "Verlet Parameters:\n";
    std::cout << "+-----------------+---------------+---------------+------------------------------+\n";
    std::cout << "| Timestep: " << delta << " | Start Time: " << t_start << " | End Time: " << t_end << " | Exports Every: " << export_every << " Iteration/s |\n";
    std::cout << "+-----------------+---------------+---------------+------------------------------+\n\n";
}

double avg(double A, double B) {
    /* Returns the average between two values
    A : Value 1
    B : Value 2
    */
    return (A + B)/2;
}

double smoothing_kernel(double v, double h) {
    /* Calculates the value of a cubic spline for a distance v between 2 particles
    v : Distance between the two compared particles
    h : Smoothing length
    */
    // Normalise distance
    double q = fabs(v)/h;
    // Cubic-Spline Code
    double w = 0.0;
    if (q >= 0.0 && q <= 1.0) {
        w = 2/(3*h) * (1.0 - 1.5 * std::pow(q,2) + 0.75 * std::pow(q,3));
    } else if (q > 1 && q <= 2) {
        w = 2/(3*h) * (0.25 * std::pow((2-q), 3));
    }
    return w;
}

double grad_smoothing_kernel(double v, double h) {
    /* Calculates the value of the gradient of a cubic spline for a distance v between 2 particles
    v : Distance between the two compared particles
    h : Smoothing length
    */
    // Determine sign
    double sign = (v >= 0.0) ? 1 : -1;
    // Normalise distance
    double q = fabs(v)/h;
    // Gradient of Cubic-Spline Code
    double w = 0.0;
    if (q >= 0.0 && q <= 1.0) {
        w = sign * 2/(3*h*h) * (-3 * q + 2.25 * std::pow(q,2));
    } else if (q > 1 && q <= 2) {
        w = sign * 2/(3*h*h) * (-0.75 * std::pow((2-q),2));
    }
    return w;
}

double density(double mass, double pos_a, double pos_b, double h) {
    /* Calculates the density of a particle using the smoothing kernel
    mass : Mass of particle a
    pos_a : x-position of particle a
    pos_b : x-position of particle b
    h : Smoothing length
    */
    return mass * smoothing_kernel((pos_a - pos_b), h);
}

double eos_ideal(double density, double u, float gamma) {
    /* Calculates the pressure using the ideal gas equation of state
    density : Density of the particle
    u : Specific internal energy of the particle
    gamma : Adiabatic index of the particle
    */
    return (gamma - 1.0)*density*u;
}

double variable_h(float eta, double mass, double density) {
    /* Calculates the smoothing length of a particle
    eta : Constant between 1.2 and 1.5
    mass : Mass of the particle
    density : Density of the particle
    */
    return eta * (mass/density);
}

double sound_speed(double P, double density, float gamma) {
    /* Calculates the speed of sound for a particle depending on its pressure and density
    P : Particle pressure
    density : Particle density
    gamma : Particle adiabatic index
    */
    return std::pow((gamma*P/density), 0.5);
}

double find_u(float gamma, double P, double density) {
    /* Calculates the energy due to the ideal gas equation of state
    gamma : Particle adiabatic index
    P : Particle pressure
    density : Particle density
    */
    return P/((gamma - 1.0)*density);
}

double acceleration(double mass, double Pa, double Pb, double rhoa, double rhob, double arti_visc, double posa, double posb, double h) {
    /* Calculates the acceleration of particle a due to its interaction with particle b
    mass : Mass of particle a
    Pa : Pressure of particle a
    Pb : Pressure of particle b
    rhoa : Density of particle a
    rhob : Density of particle b
    arti_visc : Artificial viscosity value
    posa : x-position of particle a
    posb : x-position of particle b
    h : Smoothing length
    */
    double acc = -mass * (Pa/std::pow(rhoa,2) + Pb/std::pow(rhob,2) + arti_visc) * grad_smoothing_kernel((posa-posb),h);
    return acc;
}

double power(double mass, double Pa, double rhoa, double arti_visc, double vela, double velb, double posa, double posb, double h) {
    /* Calculates the power of particle a due to its interaction with particle b
    mass : Mass of particle a
    Pa : Pressure of particle a
    rhoa : Density of particle a
    arti_visc : Artificial viscosity value
    vela : Velocity of particle a
    velb : Velocity of particle b
    posa : x-position of particle a
    posb : x-position of particle b
    h : Smoothing length
    */
    return mass * (Pa/std::pow(rhoa,2) + 0.5*arti_visc) * (vela-velb) * grad_smoothing_kernel((posa-posb), h);
}

double artificial_viscosity(double cs_a, double cs_b, double rhoa, double rhob, double posa, double posb, double vela, double velb, double avg_h) {
    /* Calculates the artificial viscosity components between two particles
    cs_a : Speed of sound for particle a
    cs_b : Speed of sound for particle b
    rhoa : Density of particle a
    rhob : Density of particle b
    posa : x-position of particle a
    posb : x-position of particle b
    vela : Velocity of particle a
    velb : Velocity of particle b
    avg_h : Average smoothing length between particles a and b
    */
    double artificial_visc = 0.0;
    float approach = (posa - posb) * (vela - velb);
    if (approach < 0) {
        double mu = (avg_h*approach)/(std::pow((posa-posb),2) + ep*std::pow(avg_h,2));
        artificial_visc = (-Alpha*avg(cs_a, cs_b)*mu + Beta*std::pow(mu,2))/avg(rhoa,rhob);
    }
    return artificial_visc;
}

void verlet_step(ParticleSystem& sys, float dt, int N) {
    /* Performs a single step of the Velocity-Verlet integration method.
    sys : A pointer to an initiation of the ParticleSystem class
    dt : Time step
    N : Number of particles
    */
    // Half step
    for (int i=0; i<N; i++) {
        // Skip boundary particles
        if (sys.boundary[i] == true) {
            continue;
        }
        sys.vel[i] += 0.5 * dt * sys.acc[i];
        sys.u[i] += 0.5 * dt * sys.powr[i];
        // Position full step
        sys.pos[i] += dt * sys.vel[i];
    }

    // Compute new accelerations and powers
    sys.find_params();

    // Full step
    for (int i=0; i<N; i++) {
        // Skip boundary particles
        if (sys.boundary[i] == true) {
            continue;
        }
        sys.vel[i] += 0.5 * dt * sys.acc[i];
        sys.u[i] += 0.5 * dt * sys.powr[i];
    }
    // Compute total energy and momentum
    sys.tot_E = sys.total_energy(N);
    sys.tot_mom = sys.total_momentum(N);
}

void verlet(ParticleSystem& sys, float dt, float t_start, float t_end, bool export_bool, int export_every) {
    /* Function that runs the full Velocity-Verlet integration method between specified time periods, given a time step.
    sys : A pointer to an initiation of the ParticleSystem class
    dt : Time step
    t_start : Verlet start time
    t_end : Verlet end time
    export_bool : Toggle for exporting data
    export_every : Integer representing how many iterations between an export
    */
    // Print a message once the Verlet integrator starts
    std::cout << "Starting Verlet Integration...\n";
    // Find system size
    int N = sys.pos.size();
    // Find number of steps to iterated over
    int no_steps = (t_end - t_start)/dt;

    // Find initial accelerations and power
    sys.find_params();

    // Keep track of the timestep
    float t = 0.0;
    // Run Velocity-Verlet loop
    for (int i=0; i<no_steps; i++)  {
        verlet_step(sys, dt, N);
        t += dt;
        // Exporting code
        if (i % export_every == 0) {
            if (export_bool == 1) {
                export_data(txtfile, false, sys, N, t);
            }
        }
        // Show the integrator progress
        progress_bar(i+1, no_steps);
        // Print a message once the Verlet integrator has completed
        if (i == no_steps-1) {
            std::cout << "\nSimulation Complete!\nNumber of iterations: " << i << "\n";
        }
    }
}

void export_data(std::string filename, bool restart, const ParticleSystem& sys, int N, float time) {
    /* Function that exports data to a text file
    filename : Name of the file to export to
    restart : A bool given to tell the function to restart the file writing process or append data to an existing file
    sys : A pointer to an initiation of the ParticleSystem class
    N : Number of particles in the system
    time : Time in the simulation the data is wrong
    */
    std::ofstream file;
    // If 'restart' is true, re-initialise the file that is being written to
    if (restart == 1) {
        file.open(filename, std::ios::trunc);
        // If file can't open flag an error!
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return;
        }
        // Create headers for the data
        file << "Particle\tPosition\tVelocity\tAcceleration\tDensity\tPressure\tEnergy\tTotal Energy\tTotal Momentum\tTime\n";
        // Write information to the file
        for (int i=0; i<N; i++) {
            // Ignore boundary particles
            if (sys.boundary[i] == true) {
                continue;
            }
            file << std::fixed << std::setprecision(6);
            file << i-boundary_particles << "\t" << sys.pos[i] << "\t" << sys.vel[i] << "\t" << sys.acc[i] << "\t" << sys.rho[i] << "\t" << sys.P[i] << "\t" << sys.u[i] << "\t" << sys.tot_E << "\t" << sys.tot_mom << "\t" << time << "\n";
        }
        // Close the file
        file.close();
    } else if (restart == 0) {
        // Append data to the file
        file.open(filename, std::ios::app);
        // If file can't open flag an error!
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return;
        }
        // Write information to the file
        for (int i=0; i<N; i++) {
            // Ignore boundary particles
            if (sys.boundary[i] == true) {
                continue;
            }
            file << std::fixed << std::setprecision(6);
            file << i-boundary_particles << "\t" << sys.pos[i] << "\t" << sys.vel[i] << "\t" << sys.acc[i] << "\t" << sys.rho[i] << "\t" << sys.P[i] << "\t" << sys.u[i] << "\t" << sys.tot_E << "\t" << sys.tot_mom << "\t" << time << "\n";
        }
        // Close the file
        file.close();
    } else {
        std::cerr << "Error: Expected restart to be a boolean, but this was not given." << std::endl;
        return;
    }
}

void progress_bar(int step, int total_steps) {
    /* Prints a progress bar to the terminal so that the user can see the progress of the Verlet Integrator
    step : Current step the integrator is on
    total_steps : The total number of steps the integrator has to run through
    */
    int bar_len = 50;
    std::cout << "\r[";
    int progress = (int)(bar_len * ((float)step/total_steps));
    for (int i=0; i<bar_len; i++) {
        if (i < progress) {
            std::cout << "#";
        } else {
            std::cout << "-";
        }
    }
    std::cout << "] " << int((float)step/total_steps * 100) << "%  ";
    std::cout.flush();
}
