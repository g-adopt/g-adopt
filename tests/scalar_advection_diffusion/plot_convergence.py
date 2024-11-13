import numpy as np
import matplotlib.pyplot as plt

nx = [50 / (2**i) for i in range(8)]  # really this is the grid peclet number

# running with a grid peclet number of 50 and initial mesh resolution of 5 cells gives these results
SU = np.loadtxt("errors-Pe50.0_SUTrue.dat")[:, 0]
SU = np.append(SU, [0.03732994, 0.0180603, 0.00614359, 0.00170367])
noSU = np.loadtxt("errors-Pe50.0_SUFalse.dat")[:, 0]
noSU = np.append(noSU, [0.04086379, 0.01557628, 0.00484643, 0.00130424])
SU_upwind = [
    0.16548618,
    0.11845977,
    0.08582658,
    0.06341912,
    0.0456253,
    0.02835158,
    0.01551193,
    0.00813498,
]  # i.e set Beta(Pe) = 1.

# running with a grid peclet number of 0.25 and initial mesh resolution of 5 cells gives these results
nx_fromPe0pt25 = [
    0.25 / (2**i) for i in range(4)
]  # really this is the grid peclet number
SU_fromPe0pt25 = np.loadtxt("errors-Pe0.25_SUTrue.dat")[:, 0]
noSU_fromPe0pt25 = np.loadtxt("errors-Pe0.25_SUFalse.dat")[:, 0]
SU_upwind_fromPe0pt25 = [
    0.04179666,
    0.02128963,
    0.01080482,
    0.00545314,
]  # i.e set Beta(Pe) = 1.

halforder = 1e-2 * np.array(nx) ** 0.5
firstorder = 0.5 * np.array(nx_fromPe0pt25)
secondorder = 5e-2 * np.array(nx_fromPe0pt25) ** 2

plt.loglog(nx, noSU, "bx", label="Galerkin")
plt.loglog(nx, SU_upwind, "rx", label="Pure upwinding")
plt.loglog(nx, SU, "yx", label="SU")
plt.loglog(nx_fromPe0pt25, noSU_fromPe0pt25, "b+", label="Galerkin from Pe = 0.25")
plt.loglog(
    nx_fromPe0pt25, SU_upwind_fromPe0pt25, "r+", label="Pure upwinding from Pe = 0.25"
)
plt.loglog(nx_fromPe0pt25, SU_fromPe0pt25, "y+", label="SU from Pe = 0.25")
plt.loglog(nx[:4], halforder[:4], "k--")
plt.loglog(nx_fromPe0pt25, firstorder, "k--")
plt.loglog(nx_fromPe0pt25, secondorder, "k--")
plt.text(3.5e-2, 7e-2, "O(1)")
plt.text(1e-1, 2e-4, "O(2)")
plt.text(15, 2e-2, "O(1/2)")
plt.xlabel("grid Peclet number")
plt.ylabel("L2 error")
plt.grid()
plt.legend()
plt.savefig("scalaradvdif_donea_analytical.png")
