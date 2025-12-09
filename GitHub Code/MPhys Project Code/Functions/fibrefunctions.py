# General Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib import gridspec
import random
import torch

# Imports from other python files
from fibredata import *

# ==============================================================
#                        Global Variables
# ==============================================================

wl = 633e-9                     #[m]  # From MATLAB Code
c = 299792458                   #[m/s]
freq = c/wl                     #[Hz]
n_c = 1.4570                    # Core refractive index  # From MATLAB Code
no_pix = 31
mask_len = 25e-6                #[m]  # From MATLAB Code
pix_size = mask_len / no_pix    #[m]

# ==============================================================
#                         Maths Functions
# ==============================================================

def vectorise(field):
    """Reshapes a 2D matrix into a column vector."""
    x = field.shape[0]
    y = field.shape[1]
    return field.reshape(x*y,1)

def unvectorise(field, dim):
    """Reshapes a column vector into a 2D square matrix of size (dim,dim)."""
    return field.reshape(dim,dim)

def norm(mat):
    """Normalises a given complex matrix so that radii are between values of 0 and 1, while maintaining the angles."""
    rad = torch.abs(mat)
    max_rad = torch.max(rad)
    norm_rad = rad/max_rad
    angle = mat.angle()

    return norm_rad*(torch.cos(angle) + 1j*torch.sin(angle))

def norm_nan(mat):
    """Normalises a given complex matrix so that radii are between values of 0 and 1, while maintaining the angles.
    Works with NaN elements using numpy."""
    rad = np.abs(mat)
    max_rad = np.nanmax(rad)
    norm_rad = rad/max_rad
    angle = np.angle(mat)

    return norm_rad*(np.cos(angle) + 1j*np.sin(angle))

def coupling_gaussian(l, p, l0, p0, sig_l:float=0.0, sig_p:float=0.0):
    """Finds the coupling gaussian on the PIM Power Spectrum."""
    return torch.exp(-(l-l0)**2/(2*sig_l**2+0.00001) - (p-p0)**2/(2*sig_p**2+0.00001))

# ==============================================================
#                         Misc Functions
# ==============================================================

def Complex2HSV_old(z, rmin, rmax):
    """Finds the magnitude and angle of a complex number and uses it to find a hsv colour value.
    Returns the equivalent RGB value for matplotlib.pyplot"""
    # get amplidude of z and limit to [rmin, rmax]
    amp = np.abs(z)
    amp = np.where(amp < rmin, rmin, amp)
    amp = np.where(amp > rmax, rmax, amp)
    ph = np.angle(z, deg=1)
    # HSV are values in range [0,1]
    h = (ph % 360) / 360
    s = 0.85 * np.ones_like(h)
    v = (amp - rmin) / (rmax - rmin)
    return hsv_to_rgb(np.dstack((h,s,v)))

def visualise_old(field, title=""):
    """Displays a visual plot of the field, using hsv colour mapping to demonstrate the fields phase (Hue) and amplitude (Value)."""
    # Set up plots
    fig, axs = plt.subplots(1,2, figsize=(10,5))
        
    # Plot the given field
    if field.is_cuda == True:
        axs[0].imshow(Complex2HSV_old(field.cpu(), 0, 0.065))
    else:
        axs[0].imshow(Complex2HSV_old(field, 0, 0.065))

    # Colour bar
    V, H = np.mgrid[0:1:100j, 0:1:300j]
    S = np.ones_like(V)
    HSV = np.dstack((H,S,V))
    RGB = hsv_to_rgb(HSV)

    axs[1].imshow(RGB, origin="lower", extent=[0, 2*np.pi, 0, 1], aspect=15)

    axs[1].set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], ['0', '$\pi/2$', '$\pi$', '$3\pi/2$','$2\pi$'])
    axs[1].set_yticks([0, 1], ['0', '1'])

    axs[1].set_ylabel("Amplitude")
    axs[1].set_xlabel("Phase (rad.)")
    fig.suptitle(title, fontsize=16)

    fig.show()

def Complex2HSV(z, intensity: bool=False):
    """Finds the magnitude and angle of a complex number and uses it to find a hsv colour value.
    Returns the equivalent RGB value for matplotlib.pyplot.
    If intensity = True, returns an intensity plot, if false (default) returns the regular amplitude plot."""
    # get amplidude of z and limit to [0,1]
    z = norm(z)
    amp = np.abs(z)
    amp = np.where(amp < 0, 0, amp)
    amp = np.where(amp > 1, 1, amp)
    phase = np.angle(z, deg=1)
    # HSV are values in range [0,1]
    h = (phase % 360) / 360
    #s = np.ones_like(h)
    s = np.where(np.isnan(z), 0, 1)
    if intensity == 1:
        v = amp**2
    else:
        v = amp
    return hsv_to_rgb(np.dstack((h,s,v)))

def visualise(field, title="", intensity: bool=False):
    """Displays a visual plot of the field, using hsv colour mapping to demonstrate the fields phase (Hue) and amplitude (Value)."""
    # Set up plots
    fig, axs = plt.subplots(1,2, figsize=(10,5))
        
    # Plot the given field
    if field.is_cuda == True:
        axs[0].imshow(Complex2HSV(field.cpu(), intensity))
    else:
        axs[0].imshow(Complex2HSV(field, intensity))

    # Colour bar
    V, H = np.mgrid[0:1:100j, 0:1:300j]
    S = np.ones_like(V)
    HSV = np.dstack((H,S,V))
    RGB = hsv_to_rgb(HSV)

    axs[1].imshow(RGB, origin="lower", extent=[0, 2*np.pi, 0, 1], aspect=15)

    axs[1].set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], ['0', '$\pi/2$', '$\pi$', '$3\pi/2$','$2\pi$'])
    axs[1].set_yticks([0, 1], ['0', '1'])

    axs[1].set_ylabel("Amplitude")
    axs[1].set_xlabel("Phase (rad.)")
    fig.suptitle(title, fontsize=16)

    plt.tight_layout()
    #fig.show()

def visualise2(field1, field2, main_title="", title1="Field Before", title2="Field After", intensity: bool=False):
    """Displays a visual plot of two fields, using hsv colour mapping to demonstrate the fields' phase (Hue) and amplitude (Value)."""
    # Set up plots
    fig, axs = plt.subplots(1,3, figsize=(15,5))

    # Plot the given field
    if field1.is_cuda == True and field2.is_cuda == True:
        axs[0].imshow(Complex2HSV(field1.cpu(), intensity))
        axs[1].imshow(Complex2HSV(field2.cpu(), intensity))
    elif field1.is_cuda == False and field2.is_cuda == False:
        axs[0].imshow(Complex2HSV(field1, intensity))
        axs[1].imshow(Complex2HSV(field2, intensity))
    else:
        raise TypeError(f"Fields are on different devices! Field 1 on devnum {field1.get_device()}, while Field 2 is on devnum {field2.get_device()}.")

    # Colour bar
    V, H = np.mgrid[0:1:100j, 0:1:300j]
    S = np.ones_like(V)
    HSV = np.dstack((H,S,V))
    RGB = hsv_to_rgb(HSV)

    axs[2].imshow(RGB, origin="lower", extent=[0, 2*np.pi, 0, 1], aspect=15)

    axs[2].set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], ['0', '$\pi/2$', '$\pi$', '$3\pi/2$','$2\pi$'])
    axs[2].set_yticks([0, 1], ['0', '1'])

    axs[2].set_ylabel("Amplitude")
    axs[2].set_xlabel("Phase (rad.)")
    fig.suptitle(main_title, fontsize=18)
    axs[0].set_title(title1, fontsize=15)
    axs[1].set_title(title2, fontsize=15)

    fig.show()

def pim_superposition(n, pims=PIMs_torch, rand=True):
    """Generates a field of n pims superposed on top of eachother. Can choose random pims, or the first n."""
    # Make a field of zero values with same shape as PIM fields
    field = pims[0] - pims[0]

    # Superposition
    used_pims = []
    if rand == 1:
        while len(used_pims) < n:
            index = random.randint(0,pims.shape[0]-1)
            if index in used_pims:
                pass
            else:
                field += pims[index]
                used_pims.append(index)
    else:
        for i in range(n):
            field += pims[i]
            used_pims.append(i)

    return field, used_pims

# ==============================================================
#                Compressive Sampling Functions
# ==============================================================

def PP_spectrum(z, uwlp=uwlm, PIMs=PIMs_torch, show_phase=True) -> None:
    """Plots the PIM power spectrum for a specific MMF."""
    u,w,l,p = np.hsplit(uwlp,4)

    tri = pim_matrix(PIMs) @ vectorise(z)
    tri = torch.flip(tri, [0, 1])

    p_range = int(p.max() - p.min() + 1)
    l_range = int(l.max() - l.min() + 1)

    mat = np.zeros((p_range, l_range), dtype=np.complex128)
    mat[:] = np.nan

    for i in range(l.shape[0]):
        i_index = int(-l[0][0]) + int(l[i,0])
        j_index = int(p[i,0])
        mat[j_index, i_index] = tri[i,0]
    mat = np.flipud(mat)

    if show_phase == True:
        # Custom Complex2HSV code to deal with NaNs
        # get amplidude of z and limit to [0,1]
        z = norm_nan(mat)
        amp = np.abs(z)
        amp = np.where(amp < 0, 0, amp)
        amp = np.where(amp > 1, 1, amp)
        phase = np.angle(z, deg=1)

        # HSV are values in range [0,1]
        h = (phase % 360) / 360
        s = np.where(np.isnan(z), 0, 1)
        v = amp

        # Plotting Code
        fig = plt.figure(figsize=(7,3))
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        gs = gridspec.GridSpec(1,2, width_ratios=[1,0.2])
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])
        ax0.imshow(hsv_to_rgb(np.dstack((h,s,v))))
        ax0.set_xlabel("Azimuthal Index, l"), ax0.set_ylabel("Radial Index, p")
        fig.suptitle("PIM Power Spectrum")

        # Colour bar
        V, H = np.mgrid[0:1:100j, 0:1:300j]
        S = np.ones_like(V)
        HSV = np.dstack((H,S,V))
        RGB = hsv_to_rgb(HSV)

        ax1.imshow(RGB, origin="lower", extent=[0, 2*np.pi, 0, 1], aspect=15)

        ax1.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], ['0', '$\pi/2$', '$\pi$', '$3\pi/2$','$2\pi$'])
        ax1.set_yticks([0, 1], ['0', '1'])

        ax1.set_ylabel("Amplitude")
        ax1.set_xlabel("Phase (rad.)")
    
    else:
        # Show an intensity plot of just the amplitude of each PIM present
        z = norm_nan(mat)
        amp = np.abs(z)
        amp = np.where(amp < 0, 0, amp)
        amp = np.where(amp > 1, 1, amp)

        plot = plt.imshow(amp, cmap="inferno", vmin=0, vmax=1)
        plt.xlabel("Azimuthal Index, l"), plt.ylabel("Radial Index, p")
        plt.title("PIM Power Spectrum")
        plt.colorbar(plot, fraction=0.05, aspect=4, pad=0.01)

def PP_spectrum_gauss(l0, p0, sig_l=2.5, sig_p=1, uwlp=uwlm) -> None:
    """Generates the expected PIM coupling around a specified PIM through (l0,p0), depending on standard deviations
    sig_l and sig_p"""

    # Ordered list of PIMs from uwlm
    u,w,l,p = np.hsplit(uwlp,4)
    lp_list = []
    for i in range(l.size):
        lp_list.append([l[i][0],p[i][0]])

    # Gaussian PIM coupling plot
    gauss_n = []
    for j in range(len(lp_list)): 
        lt = torch.tensor(lp_list[j][0])
        pt = torch.tensor(lp_list[j][1])
        gauss_n.append(coupling_gaussian(lt, pt, l0, p0, sig_l, sig_p))

    # Append values to a rectangular matrix
    p_range = int(p.max() - p.min() + 1)
    l_range = int(l.max() - l.min() + 1)
    mat = np.zeros((p_range, l_range), dtype=np.float64)
    mat[:] = np.nan

    for i in range(l.shape[0]):
        i_index = int(-l[0][0]) + int(l[i,0])
        j_index = int(p[i,0])
        mat[j_index, i_index] = gauss_n[i]
    mat = np.flipud(mat)

    plot = plt.imshow(mat, cmap="inferno")
    plt.xlabel("Azimuthal Index, l"), plt.ylabel("Radial Index, p")
    plt.title("PIM Power Spectrum")
    plt.colorbar(plot, fraction=0.05, aspect=4, pad=0.01)

def sampled_TM(sig_l, sig_p, uwlp=uwlm):
    """Generates a PIM basis transmission matrix that only contains an amplitude component. 
    Generated from given uwlp values and sig_l and sig_p."""
    # Ordered list of PIMs from uwlm
    u,w,l,p = np.hsplit(uwlm,4)
    lp_list = []
    for i in range(l.size):
        lp_list.append([l[i][0],p[i][0]])

    # Make fully sampled PIM basis TM
    compressive_TM = []
    for i in range(len(lp_list)):
        l0 = torch.tensor(lp_list[i][0])
        p0 = torch.tensor(lp_list[i][1])
        gauss_n = []
        for j in range(len(lp_list)):
            l = torch.tensor(lp_list[j][0])
            p = torch.tensor(lp_list[j][1])
            gauss_n.append(coupling_gaussian(l, p, l0, p0, sig_l, sig_p))
        compressive_TM.append(gauss_n)

    compressive_TM = np.array(compressive_TM)
    return torch.from_numpy(compressive_TM)

# ==============================================================
#                   Fibre Propagation Functions
# ==============================================================

def pim_matrix(PIMs_torch):
    """Creates the real space to pim space conversion matrix. Generated using external data for each PIM's shape."""
    pim = torch.column_stack((PIMs_torch[0].flatten(),PIMs_torch[1].flatten()))
    no_pims = PIMs_torch.size()[0]
    for i in np.arange(2, no_pims, 1):
        pim = torch.column_stack((pim, PIMs_torch[i].flatten()))
    # transpose
    pim = pim.transpose(0,1)
    return pim

def real_matrix(PIMs_torch):
    """Creates the pim space to real space conversion matrix. Generated using external data for each PIM's shape, from the pim_matrix() function."""
    real = torch.transpose(torch.conj(pim_matrix(PIMs_torch)), dim0=0, dim1=1)
    return real

def beta_matrix(beta_torch, length=0.0):
    """Function to make the diagonal square matrix for PIM propagation in an MMF."""
    return torch.diag(torch.exp(1j*length*beta_torch.flatten()))

def TM(PIMs_torch, beta_torch, length=0.0):
    """Creates the transmission matrix for a MMF, assumed to be perfectly straight,
    of length=length."""
    TM_part1 = beta_matrix(beta_torch, length) @ pim_matrix(PIMs_torch)
    TM = real_matrix(PIMs_torch) @ TM_part1
    return TM

def propagate(field, PIMs_torch, beta_torch, length=0.0):
    """Function to propagate an input field through a segment of perfectly
    straight optical fibre of length=length."""
    field_vec = vectorise(field)
    field_out_vec = TM(PIMs_torch, beta_torch, length) @ field_vec
    field_out = unvectorise(field_out_vec, 31)
    return field_out

# ==============================================================
#                     Fibre Bending Functions
# ==============================================================

def apply_zernike(field_in: torch.Tensor, j: str, alpha: torch.Tensor = 0.0, wl: float = 633e-9, mask_len: float = 25e-6) -> torch.Tensor:
    """Applies either the tip or tilt Zernike function to an input field (2D matrix)."""

    Nx = np.linspace(-0.5,0.5,field_in.shape[1])
    Ny = np.linspace(-0.5,0.5,field_in.shape[0])
    X,Y = np.meshgrid(Nx,Ny)
    X = torch.from_numpy(X) * mask_len
    Y = torch.from_numpy(Y) * mask_len
    #plt.imshow(Xs)
    #plt.colorbar()
    
    if j in ('X','x'):
        alpha = torch.deg2rad(90-alpha)
        rho = X
        dist = rho * torch.cos(alpha)
        weight = torch.exp(1j * 2*np.pi/wl * dist)
    elif j in ('Y','y'):
        alpha = torch.deg2rad(alpha)
        rho = Y
        dist = rho * torch.sin(alpha)
        weight = torch.exp(1j * 2*np.pi/wl * dist)
    else:
        raise ValueError("j input not valid! Only accepts 'X' or 'Y'!")
    
    #weight = torch.exp(1j * torch.deg2rad(torch.tensor(alpha)) * rho)
    #field_out = field_in * weight
    field_out = field_in * weight

    return field_out

def zernike_vis(size: int, j: str, alpha: float = 0.0, wl:float = 633e-9, mask_len: float = 25e-6):
    """Visualises the zernike phase mask. Requires the Field() class."""
    field = torch.ones(size,size)
    visualise(apply_zernike(field, j, alpha, wl, mask_len))
    return 0

def zernike_array(j: str, alpha: torch.Tensor = 0.0, size = 31, wl: float = 633e-9, mask_len: float = 25e-6):
    """Creates an array based on either the tip or tilt zernike polynomial."""
    Nx = np.linspace(-0.5,0.5,size)
    Ny = np.linspace(-0.5,0.5,size)
    X,Y = np.meshgrid(Nx,Ny)
    X = torch.from_numpy(X) * mask_len
    Y = torch.from_numpy(Y) * mask_len

    if j in ('X','x'):
        alpha = torch.deg2rad(90-alpha)
        rho = X
        dist = rho * torch.cos(alpha)
        weight = torch.exp(1j * 2*np.pi/wl * dist)
    elif j in ('Y','y'):
        alpha = torch.deg2rad(alpha)
        rho = Y
        dist = rho * torch.sin(alpha)
        weight = torch.exp(1j * 2*np.pi/wl * dist)
    else:
        raise ValueError("j input not valid! Only accepts 'X' or 'Y'!")

    return weight

# ==============================================================
#                         Loss Functions
# ==============================================================

def loss_func(outputs, targets):
    """Loss function for multiple sets of training data."""
    loss = 0
    for i in range(len(outputs)):
        loss += (outputs[i] - targets[i]).abs().square().sum()
    loss /= len(outputs)
    return loss

def loss_func_single(output, target):
    """Loss function for one field of training data."""
    return (output - target).abs().square().sum()

