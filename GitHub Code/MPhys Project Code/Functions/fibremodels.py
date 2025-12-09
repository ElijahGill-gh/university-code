# General Imports
import torch
import torch.nn as nn
import IPython
import matplotlib.pyplot as plt
from matplotlib import cm

# Imports from other python files
from fibrefunctions import *
from fibredata import *

# ==============================================================
#                         Fibre Models
# ==============================================================

# The class BendModel() can be used for any case I need, so use that one in future. Older functions kept here for older code that uses them.

class LengthModel(nn.Module):
    """Class to propagate a field through a straight section of MMF, trains one parameter 'length'."""
    def __init__(self, length=0.21):
        super(LengthModel, self).__init__()
        # Parameters
        self.length = nn.Parameter(data=torch.tensor([length], dtype=torch.float32, requires_grad=True))
    
    def forward(self, input_field):
        """Forward pass of the model."""
        output_field = propagate(input_field, PIMs_torch, beta_torch, self.length)
        return output_field
    
class BendModelOld(nn.Module):
    """Class to apply a phase plane, representing a bend in the fibre, to a field."""
    def __init__(self, angle=0.0):
        super(BendModelOld, self).__init__()

        self.wl = 633e-9 #[m]  # From MATLAB Code
        self.mask_len = 25e-6 #[m]  # From MATLAB Code
        # Parameters
        self.angle = nn.Parameter(data=torch.tensor([angle], dtype=torch.float32, requires_grad=True))
    
    def forward(self, input_field, j:str):
        """Forward pass of the model."""
        prop_field = propagate(input_field, PIMs_torch, beta_torch, 0.1)
        bend_field = apply_zernike(prop_field, j, self.angle, self.wl, self.mask_len)
        output_field = propagate(bend_field, PIMs_torch, beta_torch, 0.1)
        return output_field

class BendModel2(nn.Module):
    """Class to apply 2 phase planes, representing a bend in the fibre, to a field."""
    def __init__(self, anglex=0.0, angley=0.0):
        super(BendModel2, self).__init__()

        self.wl = 633e-9 #[m]  # From MATLAB Code
        self.mask_len = 25e-6 #[m]  # From MATLAB Code
        # Parameters
        self.anglex = nn.Parameter(data=torch.tensor([anglex], dtype=torch.float32, requires_grad=True))
        self.angley = nn.Parameter(data=torch.tensor([angley], dtype=torch.float32, requires_grad=True))
    
    def forward(self, input_field):
        """Forward pass of the model."""
        prop_field = propagate(input_field, PIMs_torch, beta_torch, 0.1)
        bend_field = apply_zernike(prop_field, 'x', self.anglex, self.wl, self.mask_len)
        bend_field = apply_zernike(bend_field, 'y', self.angley, self.wl, self.mask_len)
        output_field = propagate(bend_field, PIMs_torch, beta_torch, 0.1)
        return output_field

class BendModel(nn.Module):
    """Class to model a field propagation through an MMF with n number of bends."""
    def __init__(self, angles: np.ndarray=np.zeros((1,2)), prop_len: float=0.1):
        super(BendModel, self).__init__()

        self.N = angles.shape[0]
        #self.wl = 633e-9 #[m]  # From MATLAB Code
        #self.mask_len = 25e-6 #[m]  # From MATLAB Code
        self.prop_len = prop_len
        # Parameters
        self.angles = nn.Parameter(data=angles.clone().detach(), requires_grad=True)
    
    def forward(self, input_field):
        """Forward pass of the model."""
        endsegment_field = input_field
        # For n bends, propagate and apply bends for each segment
        for i in range(self.N):
            prop_field = propagate(endsegment_field, PIMs_torch, beta_torch, self.prop_len)
            bend_field = apply_zernike(prop_field, 'x', self.angles[i,0], wl, mask_len)
            endsegment_field = apply_zernike(bend_field, 'y', self.angles[i,1], wl, mask_len)
        # After all bends, propagate one more time
        output_field = propagate(endsegment_field, PIMs_torch, beta_torch, self.prop_len)
        return output_field
    

# ==============================================================
#                     Plotting Loss Functions
# ==============================================================

def plot_loss_length(n, dist, l_bound, u_bound, num_points):
    """Plots the loss function.
    n = Number of PIMs
    dist = Propagation distance [m]
    l_bound, u_bound = Lower and upper bounds for the loss plot
    num_points = Number of points used in the loss plot"""
    
    model = LengthModel(dist)

    fields_in = PIMs_torch[0:n]

    fields_target = []
    for i in range(n):
        fields_target.append(model(fields_in[i]))

    # Plot loss function using the target data
    loss = []
    xs = torch.from_numpy(np.linspace(l_bound, u_bound, num_points))

    for i in range(xs.size()[0]):
        model = LengthModel(xs[i])
        fields_out = []
    
        for j in range(n):
            fields_out.append(model(fields_in[j]))
    
        loss.append(loss_func(fields_out, fields_target).cpu().detach().numpy())

        if i % 10 == 0:
            print(f"[{i}/{num_points}]")
            IPython.display.clear_output(wait=True)

    plt.plot(xs,loss)
    plt.grid(visible=True)
    plt.ylabel("Loss")
    plt.xlabel("Length [m]")
    plt.title(f"Loss Function (True Length = {dist}$^o$)")

    return 0 

def loss_bend(n, alpha, j:str, l_bound, u_bound, num_points):
    """Calculates the loss depending on angle for the loss function.
    n = Number of PIMs
    alpha = Bend Angle [deg]
    j = x or y for different zernike mode
    l_bound, u_bound = Lower and upper bounds for the loss plot
    num_points = Number of points used in the loss plot"""
    
    model = BendModelOld(alpha)

    fields_in = PIMs_torch[0:n]

    fields_target = []
    for i in range(n):
        fields_target.append(model(fields_in[i], j))

    # Plot loss function using the target data
    loss = []
    xs = torch.from_numpy(np.linspace(l_bound, u_bound, num_points))

    for i in range(xs.size()[0]):
        model = BendModelOld(xs[i])
        fields_out = []
    
        for k in range(n):
            fields_out.append(model(fields_in[k], j))
    
        loss.append(loss_func(fields_out, fields_target).cpu().detach().numpy())

        #if i % 10 == 0:
        print(f"[{i}/{num_points}]")
        IPython.display.clear_output(wait=True)

    return xs, loss

def plot_loss_bend(n, alpha, j:str, l_bound, u_bound, num_points):
    """Plots the loss function.
    n = Number of PIMs
    alpha = Bend Angle [deg]
    j = x or y for different zernike mode
    l_bound, u_bound = Lower and upper bounds for the loss plot
    num_points = Number of points used in the loss plot"""
    
    xs, loss = loss_bend(n, alpha, j, l_bound, u_bound, num_points)

    plt.plot(xs,loss)
    plt.grid(visible=True)
    plt.ylabel("Loss")
    plt.xlabel("Angle [degrees]")
    plt.title(f"Loss Function (True Angle = {alpha}$^o$)")

    return 0

def plot_bendxy_loss(n, alpha, beta, l_bound, u_bound, num_points=100):
    """Returns a surface plot of the loss for a bend with both an x and y component.
    n = Number of Input Fields
    alpha, beta = Bend angles for x and y tilt respectively
    l_bound, u_bound = Lower and Upper bound for the grid
    num_points = Number of points on one axis of the grid (default is 100x100)"""

    # Compute loss for n fields
    model = BendModel2(alpha, beta)
    fields_in = PIMs_torch[:n]

    # Make target fields
    fields_target = []
    for i in range(n):
        fields_target.append(model(fields_in[i]).detach())
    fields_target = np.array(fields_target)
    fields_target = torch.from_numpy(fields_target)

    # Have a 100x100 surface plot for loss where x,y = alpha,beta and z = loss
    xs = np.linspace(l_bound, u_bound, num_points)
    ys = np.linspace(l_bound, u_bound, num_points)
    xs, ys = np.meshgrid(xs, ys)
    xs = torch.from_numpy(xs)
    ys = torch.from_numpy(ys)

    # Compute the loss for each point on the grid
    loss = np.empty(xs.shape)
    loss[:] = np.nan
    for i in range(xs.shape[0]):
        print(f"[{i}/{num_points}]")
        IPython.display.clear_output(wait=True)

        for j in range(xs.shape[0]):
            model = BendModel2(xs[i,j],ys[i,j])
            
            # Make output data
            fields_out = []
            for k in range(n):
                fields_out.append(model(fields_in[k]).detach())
            fields_out = np.array(fields_out)
            fields_out = torch.from_numpy(fields_out)
            
            loss[i,j] = loss_func(fields_out, fields_target).cpu().detach().numpy()

    fig = plt.figure(figsize=(20,5))
    fig.suptitle(r"Loss Function (True angles: $\alpha$="+str(alpha)+r", $\beta$="+ str(beta)+")", fontsize=18)
    # ================
    # First Subplot
    # ================
    ax = fig.add_subplot(1, 4, 1, projection='3d')
    surf = ax.plot_surface(xs,ys,loss, cmap=cm.jet)
    ax.set_title("Regular View")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\beta$")
    ax.set_zlabel("Loss")
    # ================
    # Directional Subplots
    # ================
    # z # top-down
    ax = fig.add_subplot(1, 4, 2, projection='3d')
    surf = ax.plot_surface(xs,ys,loss, cmap=cm.jet)
    fig.colorbar(surf, shrink=0.4, aspect=10)
    ax.view_init(elev=90, azim=0, roll=0)
    ax.zaxis.set_ticklabels([])
    ax.set_title("Top-down View")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\beta$")
    # x
    ax = fig.add_subplot(1, 4, 3, projection='3d')
    surf = ax.plot_surface(xs,ys,loss, cmap=cm.jet)
    fig.colorbar(surf, shrink=0.4, aspect=10)
    ax.view_init(elev=0, azim=0, roll=0)
    ax.xaxis.set_ticklabels([])
    ax.set_title(r"Side On View ($\beta$)")
    ax.set_ylabel(r"$\beta$")
    ax.set_zlabel("Loss")
    # y
    ax = fig.add_subplot(1, 4, 4, projection='3d')
    surf = ax.plot_surface(xs,ys,loss, cmap=cm.jet)
    fig.colorbar(surf, shrink=0.4, aspect=10)
    ax.view_init(elev=0, azim=-90, roll=0)
    ax.yaxis.set_ticklabels([])
    ax.set_title(r"Side On View ($\alpha$)")
    ax.set_xlabel(r"$\alpha$")
    ax.set_zlabel("Loss")

# ==============================================================
#                  Interpolating Loss Functions
# ==============================================================

def interp_arr(A, B):
    """Returns a new, linearly interpolated, array between arrays A and B."""
    return np.linspace(A,B,3)[1]

def interp_loss(alpha, j:str, l_bound, u_bound, num_points):
    """Function generates interpolated field data to plot a loss function with."""
    model = BendModel(alpha)

    field_in = PIMs_torch[0]
    field_target = model(field_in, j)

    # Plot loss function using the target data
    loss = []
    angles = []
    xs = torch.from_numpy(np.linspace(l_bound, u_bound, num_points))

    for i in range(xs.size()[0]):
        print(f"[{i}/{num_points}]")
        IPython.display.clear_output(wait=True)
        # Want to find the i and i+1 fields
        if i+1 < xs.size()[0]:
            model = BendModel(xs[i])
            field1 = model(field_in, j)
            model = BendModel(xs[i+1])
            field2 = model(field_in, j)

            # Linear interpolate between 2 fields
            field_interp = interp_arr(field1.detach(), field2.detach())

            # Make plotting data
            angles.append(xs[i])
            angles.append(xs[i] + (xs[i+1]-xs[i])/2)
            loss.append(loss_func_single(field1, field_target).cpu().detach().numpy())
            loss.append(loss_func_single(torch.from_numpy(field_interp), field_target).cpu().detach().numpy())
        else:
            model = BendModel(xs[i])
            field1 = model(field_in, j)
            angles.append(xs[i])
            loss.append(loss_func_single(field1, field_target).cpu().detach().numpy())
    
    return angles, loss

def compare_interp(alpha, j:str, l_bound, u_bound, num_points):
    """Function plots 3 graphs: Original plot, Interpolated Plot, Actual Plot for same number of interpolated points"""
    fig, axs = plt.subplots(1,3, figsize=(15,5))
    fig.suptitle(r"Interpolated Loss Function Comparison (True angle: $\alpha$="+str(alpha)+")", fontsize=18)
    # =================================================
    # First Subplot - Original Loss Function
    # =================================================
    xs, loss = loss_bend(1, alpha, j, l_bound, u_bound, num_points)
    axs[0].plot(xs, loss)
    axs[0].set_xlabel("Angle [degrees]"), axs[0].set_ylabel("Loss"), axs[0].set_title(f"Loss Function, {num_points} Data Points"), axs[0].grid(visible=True)
    print("1/3")
    # =================================================
    # Second Subplot - Interpolated Loss Function
    # =================================================
    angles, loss = interp_loss(alpha, j, l_bound, u_bound, num_points)
    axs[1].plot(angles, loss)
    axs[1].set_xlabel("Angle [degrees]"), axs[1].set_ylabel("Loss"), axs[1].set_title(f"Interpolated Loss Function, {2*num_points} Data Points"), axs[1].grid(visible=True)
    print("2/3")
    # =================================================
    # Third Subplot - Actual Loss Function, 2x points
    # =================================================
    angles, loss = loss_bend(1, alpha, j, l_bound, u_bound, 2*num_points)
    axs[2].plot(angles, loss)
    axs[2].set_xlabel("Angle [degrees]"), axs[2].set_ylabel("Loss"), axs[2].set_title(f"Loss Function, {2*num_points} Data Points"), axs[2].grid(visible=True)
    print("3/3")