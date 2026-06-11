# ======================================================================================
#                                     N Body Functions
# ======================================================================================

# This file contains all the classes and functions used in the N Body Problem Project 
# for computational physics and modelling assessment 2.

# All functions have a preferred type for variables when defined, acting as a bug catcher
# if the wrong data type is inputted.

# Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd

# Global Constants
G = 1.0 # [m^3 / (kg s^2)]
G_SI = 6.6743e-11 # [m^3 / (kg s^2)]
G_sol = 39.477 # [AU^3 / (M_sun yr^2)]
G_earth = 1.1857e-4 # [AU^3 / (M_earth yr^2)]
au = 1.4960e11 # [m]
ep = 1e-4


# ======================================================================================
#                             Body Class and Relevant Functions
# ======================================================================================

class Body:
    """Make an object representing a body.

    pos = A list containing the [x,y,z] coordinates of the object
    vel = A list containing the [vx,vy,vz] velocities of the object
    mass = A float value for the mass of the object
    """
    def __init__(self, pos:list, vel:list, mass:float):
        """Initialisation of attributes."""
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.mass = mass

    # Methods
    def values(self):
        """Returns a list containing the body's current position, velocity and mass."""
        return [self.pos, self.vel, self.mass]
    
    def KE(self):
        """Returns the kinetic energy, KE, of the body using its velocity and mass."""
        return 0.5 * self.mass * np.linalg.norm(self.vel)**2
    
    def GPE(self, bodies:list, G:float=G):
        """Returns the gravitational potential energy, GPE, of the body due to all other bodies in the system.

        bodies = A list of Body() objects representing all the masses in the system
        G = Gravitational constant (Default = Global Constant Value)
        """
        GP_eng = 0
        for body in bodies:
            # Check all properties to determine if the body is the same one in the list
            if (self.pos == body.pos).all() and (self.vel == body.vel).all() and self.mass == body.mass:
                continue
            else:
                r = self.pos - body.pos
                GP_eng += (G*self.mass * body.mass)/np.linalg.norm(r)
        
        return GP_eng


def find_positions(bodies:list):
    """Returns a list of the positions for each body in the system.
    
    bodies = A list of Body() objects representing all the masses in the system
    """
    return [body.pos for body in bodies]


def find_velocities(bodies:list):
    """Returns a list of the velocities for each body in the system.
    
    bodies = A list of Body() objects representing all the masses in the system
    """
    return [body.vel for body in bodies]

def find_masses(bodies:list):
    """Returns a list of the masses for each body in the system.
    
    bodies = A list of Body() objects representing all the masses in the system
    """
    return [body.mass for body in bodies]

# ======================================================================================
#                                Force and Energy Functions
# ======================================================================================

def force_gravity(pos1:np.ndarray, pos2:np.ndarray, m1:float, m2:float, G:float=G, ep:float=ep):
    """Function that returns the force due to gravity between 2 bodies.

    pos1 = Vector position of body 1
    pos2 = Vector position of body 2
    m1 = Mass of body 1
    m2 = Mass of body 2
    G = Gravitational Constant (Default = Global Constant Value)
    ep = A constant to avoid dividing by 0 when r is 0 (Default = Global Constant Value)
    """
    # Find distance between bodies
    r = pos1 - pos2
    # Calculate force due to gravity
    return -(G*m1*m2)/((np.linalg.norm(r))**3 + ep**2) * r


def system_energy(bodies:list, G:float=G):
    """Finds the total energy for an N body system.

    bodies = A list of Body() objects representing all the masses in the system
    G = Gravitational Constant (Default = Global Constant Value)
    """
    # Find the kinetic energy of the bodies
    kin_eng = 0
    for body in bodies:
        kin_eng += body.KE()

    # Find the GPE of the whole system
    gp_eng = 0
    for i in range(len(bodies)):
        r = bodies[i].pos - bodies[(i+1)%len(bodies)].pos
        gp_eng += (G*bodies[i].mass*bodies[(i+1)%len(bodies)].mass)/np.linalg.norm(r)

    return kin_eng - gp_eng


def system_ang_momentum(bodies:list):
    """Finds the total angular momentum for an N body system.
    
    bodies = A list of Body() objects representing all the masses in the system
    """
    # Split position, velocity and mass data into their own arrays
    positions = np.array(find_positions(bodies))
    velocities = np.array(find_velocities(bodies))
    masses = np.array(find_masses(bodies))

    # Calculate the centre of mass (COM) parameters
    M = np.sum(masses) # Total mass
    R = np.dot(masses, positions) / M
    V = np.dot(masses, velocities) / M

    # Find the angular momentum for each body and sum together
    RxV = np.cross(R,V)
    Rxvel = np.cross(R,velocities)
    Vxpos = np.cross(V,positions)
    posxvel = np.cross(positions,velocities)

    Ls = (RxV + Rxvel + Vxpos + posxvel) * masses.reshape(-1,1)
    L_sum = np.sum(Ls, axis=0)
    
    return L_sum


def accelerations(bodies:list, G:float=G, ep:float=ep):
    """Returns a list containing all the accelerations for each of the bodies.
    
    bodies = A list of Body() objects representing all the masses in the system. Outputted list is in same order as 'bodies'
    G = Gravitational Constant (Default = Global Constant Value)
    ep = A constant to avoid dividing by 0 when r is 0 (Default = Global Constant Value)
    """
    # Number of bodies
    n = len(bodies)
    # Make an acceleration list the same shape as the input data
    accs = [0] * n
    for i in range(n):
        accs[i] = np.zeros(3, dtype='float')
    
    for i in range(n):
        for j in range(i+1, n):
            # Calculate force due to gravity
            force = force_gravity(bodies[i].pos, bodies[j].pos, bodies[i].mass, bodies[j].mass, G, ep)
            
            # Find acceleration due to gravity on the interacting bodies
            accs[i] += force / bodies[i].mass
            accs[j] -= force / bodies[j].mass

    return accs


def find_semimajor_axis(pos_list:list, t_list:list, P:float):
    """Finds the average semimajor axis for each mass in an N-Body system.
    
    pos_list = List of positions for each timestep of the verlet method
    t_list = List of timesteps of the verlet method
    P = Time period for a single orbit given as a float
    """
    # Calculate number of orbits
    num_orbits = int(np.floor(t_list[-1]/P))
    sma = []
    for j in range(len(pos_list[0])):
        count = 1
        old_index = 0
        a = []
        while count <= num_orbits:
            # Find the index of the period
            index = next(i for i,t in enumerate(t_list) if t>P*count)
            # Find positions for this period
            ps = pos_list[old_index:index]
            # Make a list of just the x value of the positions
            x_list = []
            for i,p in enumerate(ps):
                x_list.append(p[j][0])
            # Find the semimajor axis value in this period
            a.append((max(x_list) - min(x_list))/2)

            count += 1
            old_index = index

        sma.append(sum(a)/len(a))
        
    return sma


def find_period(t_list:list, vel_list:list):
    """Finds the average orbital period for each mass in an N-Body system.
    
    t_list = List of timesteps of the verlet method
    vel_list = List of velocities for each timestep of the verlet method
    """
    # Find the times a complete orbit is made for each mass in the system
    periods = []
    counter = 0
    for i in range(len(vel_list[0])):
        temp_periods = [0]
        for j in range(len(vel_list)):
            # Check for when the x-velocity sign changes
            if np.sign(vel_list[j-1][i][0]) == np.sign(vel_list[j][i][0]):
                pass
            else:
                if counter == 1:
                    temp_periods.append(t_list[j-1])
                    counter = 0
                else:
                    counter += 1
        temp_periods.pop(0)
        periods.append(temp_periods)

    # Find the average periods for each mass
    period = []
    for i in range(len(periods)):
        p = 0
        for j in range(len(periods[0])-1):
            p += (periods[i][j+1] - periods[i][j])
        period.append(p/(len(periods[0])-1))

    return period


def check_kepler(P:float, a1:float, a2:float, G:float, m1:float, m2:float):
    """Returns both sides of the equation for Kepler's 3rd law calculated separately.
    
    P^2 / (a1 + a2)^3 : [P] Time period, [a1, a2] semimajor axes for both masses respectively
    4pi^2 / (G(m1 + m2)) : [G] Gravitational Constant, [m1, m2] masses of object 1 and 2 respectively
    """
    # LHS
    lhs = P**2 / (a1+a2)**3
    # RHS
    rhs = 4*np.pi**2 / (G*(m1+m2))

    return lhs, rhs

# ======================================================================================
#                                     Verlet Functions
# ======================================================================================

def verlet_step(bodies:list, delta:float, G:float=G, ep:float=ep) -> None:
    """Computes one time step of velocity verlet integration. Directly changes the values of the input 'bodies'.
    
    bodies = A list of Body() objects representing all the masses in the system
    delta = Timestep
    G = Gravitational Constant (Default = Global Constant Value)
    """
    # Compute the initial accelerations for the step
    init_acc = accelerations(bodies, G, ep)
    
    # Half Velocity Step
    vel_halfsteps = []
    for i,body in enumerate(bodies):
        vel_halfsteps.append(body.vel + 0.5 * delta * init_acc[i])

    # Position Step
    for i,body in enumerate(bodies):
        body.pos = body.pos + delta * vel_halfsteps[i]

    # Compute the new accelerations with the position step
    new_acc = accelerations(bodies, G, ep)

    # Velocity Step
    for i,body in enumerate(bodies):
        body.vel = vel_halfsteps[i] + 0.5 * delta * new_acc[i]

    return None


def verlet_model(bodies:list, delta:float, t_end:float, t_start:float=0.0, G:float=G, ep:float=ep):
    """Runs a full verlet model for the N body Problem.

    bodies = A list of Body() objects representing all the masses in the system.
    delta = Timestep
    t_end = Final time interval of the model
    t_start = Initial time interval
    G = Gravitational Constant (Default = Global Constant Value)
    export = Set to True if you want to export the data to a txt file
    file_name = Name of text file to export to
    """
    # Create copies of the bodies. These values will be changed each iteration of the Verlet method
    iterable_bodies = []
    for body in bodies:
        iterable_bodies.append(Body(body.pos, body.vel, body.mass))
    
    # Print the initial total energy and total angular momentum
    print(f"Initial energy: {system_energy(iterable_bodies)}")
    print(f"Initial Angular Momentum: {system_ang_momentum(iterable_bodies)}")

    # Create lists to append data to each iteration
    t_list = []
    pos_list = []
    vel_list = []
    E_list = []
    L_list = []

    # Calculate the number of iterations
    iterations = int((t_end - t_start)/delta)
    # Print a warning message if the iterations are larger than 100,000
    if iterations > 100000:
        print(f"Running for {iterations} iterations. This might take a while!")
    else:
        pass

    # Compute Verlet iterations in a while loop
    t = t_start
    iteration = 0
    while t <= t_end:
        # Append Data for the step
        t_list.append(t)
        pos_list.append(find_positions(iterable_bodies))
        vel_list.append(find_velocities(iterable_bodies))
        E_list.append(system_energy(iterable_bodies, G))
        L_list.append(system_ang_momentum(iterable_bodies))

        # Compute step
        verlet_step(iterable_bodies, delta, G, ep)
        t += delta
        iteration += 1
        # Print the current step every 1,000,000 iterations
        if iteration % 1000000 == 0:
            print(f"Iteration: {iteration}/{iterations}")
        else:
            pass

    return t_list, pos_list, vel_list, E_list, L_list

# ======================================================================================
#                                    Plotting Functions
# ======================================================================================

def plot_orbits(t_list:list, pos_list:list, E_list:list, L_list:list, labels:list=["Mass 1","Mass 2","Mass 3"], zoom=False, save=False, name="Figure") -> None:
    """Plots the orbit paths, percentage energy and angular momentum against time for an N-body system."""
    N = len(pos_list[0])

    # Colours to plot with
    main_colours = ["red","green","blue","darkorange","darkviolet","teal","gold","navy","maroon"]      # 9 for solar system plot
    path_colours = ["pink","springgreen","cyan","sandybrown","violet","lightseagreen","khaki","royalblue","indianred"]     # 9 for solar system plot

    # Make figure and axis
    fig = plt.figure(figsize=(17,8))
    fig.suptitle(f"{N}-Body Orbit")
    gs0 = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1,1], height_ratios=[1])
    gs00 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[0])
    gs01 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs0[1])

    ax1 = fig.add_subplot(gs00[0])
    ax2 = gs01.subplots(sharex=True, sharey="row")
    ax1.grid(True, zorder=1), ax2[0].grid(True, zorder=1), ax2[1].grid(True, zorder=1), ax2[2].grid(True, zorder=1), ax2[3].grid(True, zorder=1)

    # Inset Axes
    if zoom == True:
        x1, x2, y1, y2 = -2, 2, -2, 2  # subregion of the original image
        axins = ax1.inset_axes([0.6,0, 0.4, 0.4], xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
    else:
        pass

    # Split the position data into a list for each body individually, and plot
    for i in range(len(pos_list[0])):
        x_list = []
        y_list = []
        for j in range(len(pos_list)):
            x_list.append(pos_list[j][i][0])
            y_list.append(pos_list[j][i][1])
        
        ax1.plot(x_list, y_list, zorder=2, color=path_colours[i%len(path_colours)])
        ax1.scatter(x_list[-1], y_list[-1], zorder=3, label=labels[i], color=main_colours[i%len(main_colours)])
        
        if zoom == True:
            axins.plot(x_list, y_list, zorder=2, color=path_colours[i%len(path_colours)])
            axins.scatter(x_list[-1], y_list[-1], zorder=3, label=labels[i], color=main_colours[i%len(main_colours)])
        else:
            pass

    # Plot the energy error
    ax2[0].plot(t_list, [100*(i-E_list[0])/E_list[0] for i in E_list], zorder=2)

    # Split angular momentum into 3 lists for each of its components and plot
    Lx = []
    Ly = []
    Lz = []
    for L in L_list:
        Lx.append(L[0])
        Ly.append(L[1])
        Lz.append(L[2])
    # Plot each component against time
    ax2[1].plot(t_list, Lx, zorder=2), ax2[2].plot(t_list, Ly, zorder=2), ax2[3].plot(t_list, Lz, zorder=2)

    # Add title, labels, etc
    ax1.set_title("Positions"), ax2[0].set_title("Energy Error Over Time")
    ax2[1].set_title('$L_x$'), ax2[2].set_title('$L_y$'), ax2[3].set_title('$L_z$')
    ax1.set_xlabel("X"), ax1.set_ylabel("Y")
    ax2[0].set_ylabel("Energy Error [%]"), ax2[2].set_ylabel("Angular Momentum"), ax2[3].set_xlabel("Time")
    ax1.legend()

    if zoom == True:
        ax1.indicate_inset_zoom(axins, edgecolor="black")
    else:
        pass

    plt.show()

    # Saving figure code
    if save == True:
        fig.savefig(name, dpi=300)
    else:
        pass
        
    return None

# ======================================================================================
#                             Importing and Exporting Functions
# ======================================================================================

def verlet_from_table(filename:str, system_name:str, G:float=G, save:bool=False, name:str="File") -> None:
    """Runs the verlet algorithm and plots the data from an imported comma delimited 
    text file.

    filename = Name of the comma delimited file including the file extension
    system_name = Name of a system given in the imported table
    G = Gravitational Constant (Default = Global Constant Value)
    save = Bool. Set to True if you want to save the plot
    name = A string the plot will be saved as
    """
    # Import data
    data = pd.read_csv(filename, index_col=0)
    sys = data.loc[system_name]
    
    # Make bodies
    b1 = Body([sys[0],sys[1],0], [sys[6],sys[7],0], 1)
    b2 = Body([sys[2],sys[3],0], [sys[8],sys[9],0], 1)
    b3 = Body([sys[4],sys[5],0], [sys[10],sys[11],0], 1)
    bodies = [b1,b2,b3]

    t_end = sys[17] * sys[12]

    # Run Verlet Algorithm
    t_list, pos_list, vel_list, E_list, L_list = verlet_model(bodies, delta=sys[15], t_end=t_end, G=G, ep=sys[16])

    # Plot the data
    if save == True:
        plot_orbits(t_list, pos_list, E_list, L_list, save=save, name=name)
    else:
        plot_orbits(t_list, pos_list, E_list, L_list)
    
    return None