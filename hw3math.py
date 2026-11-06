# %% importing packages
import numpy as np
from matplotlib import pyplot as plt

# %% geometry

# lattice defined like so:
a=1
a1 = a*np.array([np.sqrt(3)/2,-1/2])
a2 = a*np.array([np.sqrt(3)/2,1/2])

# The ribbon is sliced along [1,0] direction, N cells wide.
# breaking symmetry cells making "barber pole" stripes:

N = 6

R = np.array([i*a2 for i in range(N)])

# n. neighbor directions


Nvlist_inner = np.array([a1,a2,a2-a1,-a1,-a2,a1-a2])
Nvlist_bottom_edge_1 = np.array([a2,a2-a1,-a1])
Nvlist_bottom_edge_2 = np.array([a1,a2,a2-a1,-a1,-a2])
Nvlist_top_edge_1 = np.array([a1,a2,-a1,-a2,a1-a2])
Nvlist_top_edge_2 = np.array([a1,-a2,a1-a2])

Nvlist = [Nvlist_inner for _ in range(N)]
Nvlist[0]=Nvlist_bottom_edge_1
Nvlist[1]=Nvlist_bottom_edge_2
Nvlist[-2]=Nvlist_top_edge_1
Nvlist[-1]=Nvlist_top_edge_1

# n. neighbor indices

offsets_inner = np.array([-1,1,2,1,-1,-2])
offsets_bottom_1 = np.array([1,2,1])
offsets_bottom_2 = np.array([-1,1,2,1,-1])
offsets_top_1 = np.array([-1,1,1,-1,-2])
offsets_top_2 = np.array([-1,-1,-2])

offsetlist = [offsets_inner for _ in range(N)]
offsetlist[0]=offsets_bottom_1
offsetlist[1]=offsets_bottom_2
offsetlist[-2] = offsets_top_1
offsetlist[-1] = offsets_top_2

def T(k,N=N):

    Tk = np.array([[0 for i in range(N)] for i in range(N)], dtype = complex)
    for i in range(N):
        Ri = R[i]
        offsetrow = offsetlist[i]
        Nvrow = Nvlist[i]
        for offset,dr in zip(offsetrow,Nvrow):
            Tk[i,i+offset] += -1*np.exp(1j*k@(Ri+dr))
    return(Tk)

def Eta(k,N=N)
    Etak = 
# %%
A = T(np.array([0,1]))
# %%
print(abs(A))

# %%
