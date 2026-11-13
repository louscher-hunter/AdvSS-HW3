import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import eigh
from scipy.linalg import LinAlgError
from scipy.optimize import linear_sum_assignment

# lattice defined like so:
a=1
a1 = a*np.array([np.sqrt(3)/2,-1/2])
a2 = a*np.array([np.sqrt(3)/2,1/2])
t=-1
Rx = a1+a2
O = np.array([0,0])

# The ribbon is sliced along [1,0] direction, N cells wide.
# cross-symmetry cells making "barber pole" stripes:

N = 6
#%% Building the matrices

# T matrix, done Hunter's way
def T(k,n=N, site_energies = None):
    if site_energies is None:
        site_energies = np.zeros(n)

    Tk = np.diag(site_energies).astype(complex)
    neighbordexes = [1,2,1,-1,-2,-1]
    neighbordisps = [a2,-a1+a2,-a1,-a2,a1-a2,a1]

    for i in range(n):
        for dex,disp in zip(neighbordexes,neighbordisps):
            if i+dex in range(n):
                Tk[i+dex,i] += t*np.exp(1j*k@disp)
    return Tk

# T matrix, done Baishali's way
def T2(k,n=N, site_energies = None):
    if site_energies is None:
        site_energies = np.zeros(n)

    Tk = np.diag(site_energies).astype(complex)
    neighbordexes = [1,2,1,-1,-2,-1]
    neighbordisps = [O,-Rx,-Rx,O,Rx,Rx]

    for i in range(n):
        for dex,disp in zip(neighbordexes,neighbordisps):
            if i+dex in range(n):
                Tk[i+dex,i] += t*np.exp(1j*k@disp)
    return Tk



# reciprocal lattice. only needs one value because
# there is only one infinite axis

b = np.pi/np.linalg.norm(a1+a2)

#%% The juicy bit


def lattice_eigensolution(ribbon_width: int = 11, Kpath_resolution: int = 500,Tk = T):
    """
    
    args:
        ribbon_width: number of atoms across the ribbon
        Kpath_resolution: number of sample points from Gamma to X.
    
    returns:
        ET: listed by energy level, lists the energies at that level for each K-point from -X to X.
            These energies start in ascending order at -X, but the energies are tracked through crossings.

            E_n(K_j)

            Index order: [n,j]
        
        CT: listed by energy level n, for each K-point K_j, the probability amplitude at the ith lattice site C_ni.

            C_ni(K_j).

            recall that in energy state n, under wavevector K_j, the probability of measuring an excitation
            at the ith lattice site is P=|C_ni(K_j)|^2
        
            Index order: [n, j, i]


    """
    N = ribbon_width

    Xpath = np.linspace(-b,b,2*Kpath_resolution)
    Kpath = np.array([[x,0] for x in Xpath])

    num_k = len(Kpath)
    E = np.full((num_k, N), np.nan)
    C = np.full((num_k, N, N), np.nan, dtype=complex)
    
    # 1. Start at the center (Gamma, k = 0) where states are non-degenerate
    mid = num_k // 2
    w_mid, v_mid = eigh(Tk(Kpath[mid], N))
    E[mid], C[mid] = w_mid, v_mid
    
    # 2. Track forward from Gamma to +X
    v_old = v_mid
    for i in range(mid + 1, num_k):
        w, v = eigh(Tk(Kpath[i], N))
        overlap = np.abs(v_old.T.conj() @ v)
        _, mapping = linear_sum_assignment(1.0 - overlap)
        E[i], C[i] = w[mapping], v[:, mapping]
        v_old = C[i]
        
    # 3. Track backward from Gamma to -X
    v_old = v_mid
    for i in range(mid - 1, -1, -1):
        w, v = eigh(Tk(Kpath[i], N))
        overlap = np.abs(v_old.T.conj() @ v)
        _, mapping = linear_sum_assignment(1.0 - overlap)
        E[i], C[i] = w[mapping], v[:, mapping]
        v_old = C[i]
        
    # Transpose to [N_energy, L_kpoint] and [N_energy, L_kpoint, M_site]
    return E.T, np.transpose(C, axes=(2, 0, 1))