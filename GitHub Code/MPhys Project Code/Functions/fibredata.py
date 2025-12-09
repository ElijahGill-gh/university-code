# Imports
import numpy as np
import os
import scipy as sc
import torch

# ==============================================================
#                      Fibre Data - 42 PIMs
# ==============================================================

# Importing Data - From USB
if os.path.exists("F:/venv code/Y3 Work/Fibres/MATLAB Files"):
    PIM_data = sc.io.loadmat("F:/venv code/Y3 Work/Fibres/MATLAB Files/Modes.mat")
    modes = PIM_data["F2"].transpose(2,0,1) # Modes
    PIMs_torch = torch.tensor(modes, dtype=torch.cdouble)

    beta_data = sc.io.loadmat("F:/venv code/Y3 Work/Fibres/MATLAB Files/Beta.mat")
    beta = beta_data["Beta"] # Phase velocities for each mode
    beta_torch = torch.tensor(beta, dtype=torch.double)

    uwlm = np.genfromtxt("F:/venv code/Y3 Work/Fibres/MATLAB Files/uwlm.txt", delimiter=",", skip_header=1)

    print("Files (42) imported from USB.")

# Importing Data - From Kaggle
elif os.path.exists("/kaggle/input/fibre-data"):
    PIM_data = sc.io.loadmat("/kaggle/input/fibre-data/Modes.mat")
    modes = PIM_data["F2"].transpose(2,0,1) # Modes~
    PIMs_torch = torch.tensor(modes, dtype=torch.cfloat)

    beta_data = sc.io.loadmat("/kaggle/input/fibre-data/Beta.mat")
    beta = beta_data["Beta"] # Phase velocities for each mode
    beta_torch = torch.tensor(beta, dtype=torch.cfloat)

    print("Files (42) imported from Kaggle database.")

else:
    print("No files found.")

# ==============================================================
#                      Fibre Data - 189 PIMs
# ==============================================================

# Importing Data - From USB
if os.path.exists("F:/venv code/Y3 Work/Fibres/MATLAB Files"):
    PIM_data189 = sc.io.loadmat("F:/venv code/Y3 Work/Fibres/MATLAB Files/Modes189.mat")
    modes189 = PIM_data189["F2"].transpose(2,0,1) # Modes
    PIMs_torch189 = torch.tensor(modes189, dtype=torch.cdouble)

    beta_data189 = sc.io.loadmat("F:/venv code/Y3 Work/Fibres/MATLAB Files/Beta189.mat")
    beta189 = beta_data189["Beta"] # Phase velocities for each mode
    beta_torch189 = torch.tensor(beta189, dtype=torch.double)

    uwlm189 = np.genfromtxt("F:/venv code/Y3 Work/Fibres/MATLAB Files/uwlm189.txt", delimiter=",", skip_header=1)

    print("Files (189) imported from USB.")

# Importing Data - From Kaggle
elif os.path.exists("/kaggle/input/fibre-data"):
    PIM_data189 = sc.io.loadmat("/kaggle/input/fibre-data/Modes189.mat")
    modes189 = PIM_data189["F2"].transpose(2,0,1) # Modes~
    PIMs_torch189 = torch.tensor(modes189, dtype=torch.cfloat)

    beta_data189 = sc.io.loadmat("/kaggle/input/fibre-data/Beta189.mat")
    beta189 = beta_data189["Beta"] # Phase velocities for each mode
    beta_torch189 = torch.tensor(beta189, dtype=torch.cfloat)

    print("Files (189) imported from Kaggle database.")

else:
    print("No files found.")