# Imports
from Plotting_Code import *

# Read data file
pos, vel, rho, P, E, E_err, tot_mom, time = read_data("DATA.txt")

# Animate data
animate(pos, vel, rho, P, E)