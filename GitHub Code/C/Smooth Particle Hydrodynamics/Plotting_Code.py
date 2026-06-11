# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.animation as animation

# Functions

def sound_speed(P, rho, gamma):
    """Returns the speed of sound for a gas with a certain density and pressure.
    P : Pressure
    rho : Density
    gamma : Adiabatic Index
    """
    return (gamma*P/rho)**0.5

def find_p_iterative(P1, P2, rho1, rho2, gamma, G, BETA, tol=1e-6):
    """Iteratively solves for the pressure in the star region.
    P1 : Pressure on left side of the tube
    P2 : Pressure on right side of the tube
    rho1 : Density on left side of the tube
    rho2 : Density on right side of the tube
    gamma : Adiabatic Index
    G : Adiabatic constant 1
    BETA : Adiabatic constant 2
    tol : Convergence Tolerance
    """
    # Initial guess (using arithmetic mean or P1/P2 ratio)
    p_curr = (P1 + P2) / 2.0
    c1 = sound_speed(P1, rho1, gamma)
    # Group some constants for readability
    A = 2*c1/(gamma-1)
    B = 2/((gamma+1)*rho2)

    for _ in range(50):
        # Left side function (rarefaction)
        v3 = A*((p_curr/P1)**BETA - 1)
        # Right side function (shock)
        v4 = (p_curr - P2)*(B/(p_curr+G*P2))**0.5

        # Total function = 0
        f = v3 + v4
        
        # Numerical derivative (df/dp_curr)
        df = (A*BETA/(P1**BETA))*p_curr**(BETA-1) + (1-(p_curr-P2)/(2*(p_curr+G*P2)))*(B/(p_curr+G*P2))**0.5       
        p_new = p_curr - f / df
        if abs(p_new - p_curr) < tol:
            return p_new
        p_curr = p_new

    return p_curr


def analytical_shocktube(rho1, rho2, P1, P2, gamma, xs, t, xmid=0):
    """Finds the analytical solution to the shock tube and returns the parameters attached
    to a list of particles for plotting.
    rho1 : Density of left side of the tube
    rho2 : Density of right side of the tube
    P1 : Pressure of left side of the tube
    P2 : Pressure of right side of the tube
    gamma : Adiabatic index
    xs : List of positions of particles
    t : Time into shock simulation
    xmid : Original discontinuity point
    """

    # Useful values
    GAMMA = (gamma-1)/(gamma+1)
    BETA = (gamma-1)/(2*gamma)
    cs1 = sound_speed(P1, rho1, gamma)
    cs2 = sound_speed(P2, rho2, gamma)
    p_star = find_p_iterative(P1, P2, rho1, rho2, gamma, GAMMA, BETA)
    v_star = (2*cs1/(gamma-1))*(1 - (p_star/P1)**BETA)

    # If t<=0, throw an error
    if t <= 0:
        raise ValueError("Parameter 't' cannot be less than or equal to zero!")
    
    # Initialise output lists
    ds, ps, vs, us = np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs)

    # Section 1
    v1 = 0
    p1 = P1
    d1 = rho1
    u1 = p1/((gamma-1)*d1)
    
    # Section 2 is x-dependent so is calculated in a for-loop later
    
    # Section 3
    p3 = p_star
    v3 = v_star
    d3 = d1*(p3/p1)**(1/gamma)
    u3 = p3/((gamma-1)*d3)
    
    # Section 4
    v4 = v3
    p4 = p3
    d4 = rho2*((p4+GAMMA*P2)/(P2+GAMMA*p4))
    u4 = p4/((gamma-1)*d4)
    
    # Section 5
    v5 = 0
    p5 = P2
    d5 = rho2
    u5 = p5/((gamma-1)*d5)

    # Find wave speeds
    s12 = -cs1
    s23 = v_star - sound_speed(p3, d3, gamma)
    s34 = v_star
    s45 = v_star / (1-d5/d4)

    for i,x in enumerate(xs):
        # Find relative velocities of particles from where the original discontinuity was
        s = (x - xmid)/t

        if s <= s12:
            # In section 1
            ds[i], ps[i], vs[i], us[i] = d1, p1, v1, u1
        elif s <= s23:
            # In section 2
            vs[i] = 2/(gamma+1)*(cs1+s)
            ds[i] = d1*(1 - (gamma-1)/2 * vs[i]/cs1)**(1/(BETA*gamma))
            ps[i] = p1*(1 - (gamma-1)/2 * vs[i]/cs1)**(1/BETA)
            us[i] = ps[i]/((gamma-1)*ds[i])
        elif s <= s34:
            # In section 3
            ds[i], ps[i], vs[i], us[i] = d3, p3, v3, u3
        elif s <= s45:
            # In section 4
            ds[i], ps[i], vs[i], us[i] = d4, p4, v4, u4
        else:
            # In section 5
            ds[i], ps[i], vs[i], us[i] = d5, p5, v5, u5


    return ds, ps, vs, us

# Read output file function
def read_data(file):
    """Function to read the output data file from the SPH C++ code.
    file : Name of output file to read
    """
    df = pd.read_csv(file, sep='\t')

    # Find the number of particles from the datafile
    no_particles = max(df["Particle"]) + 1
    print(no_particles)

    # Sort data into lists
    # Make empty lists for variables
    pos = []
    vel = []
    rho = []
    P = []
    E = []
    tot_E = []
    tot_mom = []
    time = []

    # Append variables to lists
    for i in range(int(df.shape[0]/no_particles)):
        # Temporary lists for each iteration
        pos_temp = df["Position"].values[i*no_particles : no_particles+(i*no_particles)]
        vel_temp = df["Velocity"].values[i*no_particles : no_particles+(i*no_particles)]
        rho_temp = df["Density"].values[i*no_particles : no_particles+(i*no_particles)]
        P_temp = df["Pressure"].values[i*no_particles : no_particles+(i*no_particles)]
        E_temp = df["Energy"].values[i*no_particles : no_particles+(i*no_particles)]
        tot_E_temp = df["Total Energy"].values[i*no_particles : no_particles+(i*no_particles)]
        tot_mom_temp = df["Total Momentum"].values[i*no_particles : no_particles+(i*no_particles)]
        time_temp = df["Time"].values[i*no_particles : no_particles+(i*no_particles)]

        # Append iteration to the main lists
        pos.append(pos_temp)
        vel.append(vel_temp)
        rho.append(rho_temp)
        P.append(P_temp)
        E.append(E_temp)
        tot_E.append(tot_E_temp[0])
        tot_mom.append(tot_mom_temp[0])
        time.append(time_temp[0])

    # View the energy plot as percentage error from original energy
    E_err = [(i-tot_E[0])/tot_E[0]*100 for i in tot_E]

    return pos, vel, rho, P, E, E_err, tot_mom, time

# Animation function
def animate(pos, vel, rho, P, E) -> None:
    """Function to animate the progression of the shock tube simulation over time.
    pos : List of positions of each particle for every iteration
    vel : List of velocities of each particle for every iteration
    rho : List of densities of each particle for every iteration
    P : List of pressures of each particle for every iteration
    E : List of specific internal energies of each particle for every iteration
    """
    # Create figure
    fig, axs = plt.subplots(2,2, figsize=(10,8))
    # Labels
    axs[0,0].set_xlabel("Position, x"), axs[0,0].set_ylabel("Density"), axs[0,0].grid(visible=True)
    axs[0,1].set_xlabel("Position, x"), axs[0,1].set_ylabel("Velocity"), axs[0,1].grid(visible=True)
    axs[1,0].set_xlabel("Position, x"), axs[1,0].set_ylabel("Pressure"), axs[1,0].grid(visible=True)
    axs[1,1].set_xlabel("Position, x"), axs[1,1].set_ylabel("Specific Internal Energy"), axs[1,1].grid(visible=True)

    frames = []
    # Append each of the plots to the 'frames' list
    for i in range(len(pos)):
        frame = []
        frame.append(axs[0,0].scatter(pos[i], rho[i], c='b', s=10))
        frame.append(axs[0,1].scatter(pos[i], vel[i], c='b', s=10))
        frame.append(axs[1,0].scatter(pos[i], P[i], c='b', s=10))
        frame.append(axs[1,1].scatter(pos[i], E[i], c='b', s=10))
        
        frames.append(frame)
        if i % 50 == 0:
            print(f"Frame: {i}/{len(pos)}")

        
    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat=True)
    plt.tight_layout()
    plt.show()