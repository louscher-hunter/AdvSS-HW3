import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import eigh
from scipy.linalg import LinAlgError

# lattice defined like so:
a=1
a1 = a*np.array([np.sqrt(3)/2,-1/2])
a2 = a*np.array([np.sqrt(3)/2,1/2])

# The ribbon is sliced along [1,0] direction, N cells wide.
# cross-symmetry cells making "barber pole" stripes:

N = 6

# n.n. spatial directions for each atom

def Nvlist(n=N):
    Nvlist_inner = np.array([a1,a2,a2-a1,-a1,-a2,a1-a2])
    Nvlist_bottom_edge_1 = np.array([a2,a2-a1,-a1])
    Nvlist_bottom_edge_2 = np.array([a1,a2,a2-a1,-a1,-a2])
    Nvlist_top_edge_1 = np.array([a1,a2,-a1,-a2,a1-a2])
    Nvlist_top_edge_2 = np.array([a1,-a2,a1-a2])

    Nvlist = [Nvlist_inner for _ in range(n)]
    Nvlist[0]=Nvlist_bottom_edge_1
    Nvlist[1]=Nvlist_bottom_edge_2
    Nvlist[-2]=Nvlist_top_edge_1
    Nvlist[-1]=Nvlist_top_edge_2
    return Nvlist

# n.n. relative indices corresponding to atom location

def offsetlist(n=N):
    offsets_inner = np.array([-1,1,2,1,-1,-2])
    offsets_bottom_1 = np.array([1,2,1])
    offsets_bottom_2 = np.array([-1,1,2,1,-1])
    offsets_top_1 = np.array([-1,1,1,-1,-2])
    offsets_top_2 = np.array([-1,-1,-2])

    offsetlist = [offsets_inner for _ in range(n)]
    offsetlist[0]=offsets_bottom_1
    offsetlist[1]=offsets_bottom_2
    offsetlist[-2] = offsets_top_1
    offsetlist[-1] = offsets_top_2
    return offsetlist

#%% Building the matrices

def T(k,n=N):
    R = [a2*i for i in range(n)]
    offsets = offsetlist(n)
    Nv = Nvlist(n)
    Tk = np.array([[0 for i in range(n)] for i in range(n)], dtype = complex)
    for i in range(n):
        Ri = R[i]
        offsetrow = offsets[i]
        Nvrow = Nv[i]
        for offset,dr in zip(offsetrow,Nvrow):
            Tk[i,i+offset] += -1*np.exp(1j*k@(dr))
    return Tk


# reciprocal lattice. only needs one value because
# there is only one infinite axis

b = np.pi/np.linalg.norm(a1+a2)

#%% The juicy bit


def lattice_eigensolution(ribbon_width: int = 11, Kpath_resolution: int = 500):
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

    # Solving for E for the Odd N
    E = [[] for k in Kpath]
    C = [[] for k in Kpath]
    v_old = None

    for i,k in enumerate(Kpath):
        try:
            w,v = eigh(T(k,N))

            if i==0:
                E[i],C[i] = w,v
                v_old = v
            else:
                overlap = np.abs(v_old.T.conj()@v)
                mapping = np.argmax(overlap, axis = 1)
                E[i]=w[mapping]
                C[i]=v[:,mapping]
                v_old = C[i]

        except LinAlgError:
            E[i] = np.array([float('nan') for _ in range(N)])
            C[i] = np.full((N,N),float('nan'))
            v_old = None
        
    ET = np.transpose(np.array(E))
    CT = np.transpose(np.array(C), axes = (2,0,1))
    return ET,CT