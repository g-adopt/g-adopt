#!/usr/bin/env python3

import os
import numpy
import pylab
from matplotlib import rcParams

# Matplotlib defaults:
rcParams['axes.titlesize'] = 14
rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14
rcParams['legend.fontsize'] = 14


def log_params(f, str):
    f.write(str + "\n")
    f.flush()


# Range:
ns = numpy.array([2, 4, 8])
resolutions = ["A", "B", "C", "D"]
levels = [2, 3, 4]
expected_velocity_convergence = 3.5
expected_pressure_convergence = 2.0
symbols = ["o", "*", "s"]

# Loop over output and store convergence order:
theoretical_velocity_convergence = numpy.zeros((len(ns), 2))
theoretical_pressure_convergence = numpy.zeros((len(ns), 2))
# Arrays for storing errors:
errors_u = numpy.zeros((len(ns), len(resolutions)))
errors_p = numpy.zeros((len(ns), len(resolutions)))
for nindex, n in enumerate(ns):
    n_dir_name = f"n{n}"
    print("Working on directory: ", n_dir_name)
    retvallm = os.getcwd()
    os.chdir(n_dir_name)
    # First grab l2 norm of analytical solutions at finest level (for relative error):
    anal_vel = numpy.loadtxt("%s/errors.log" % (resolutions[-1]))[2]
    anal_pres = numpy.loadtxt("%s/errors.log" % (resolutions[-1]))[3]
    f = open(f'convergence_n{n}.log', 'w')
    for resindex, resolution in enumerate(resolutions):
        retvalres = os.getcwd()
        os.chdir(resolution)
        errors = numpy.loadtxt("errors.log")
        errors_u[nindex, resindex] = errors[0] / anal_vel  # relative u error
        errors_p[nindex, resindex] = errors[1] / anal_pres  # relative p error
        if resindex == 0:
            print(errors_u[nindex, resindex], errors_p[nindex, resindex])
        else:
            conv_u = numpy.log2(errors_u[nindex, resindex-1]/errors_u[nindex, resindex])
            conv_p = numpy.log2(errors_p[nindex, resindex-1]/errors_p[nindex, resindex])
            print(errors_u[nindex, resindex], errors_p[nindex, resindex], '|', conv_u, conv_p)
            log_params(f, f"{errors_u[nindex,resindex]} {errors_p[nindex,resindex]} {conv_u} {conv_p}")
        os.chdir(retvalres)
    theoretical_velocity_convergence[nindex, 0] = errors_u[nindex, 0]
    theoretical_velocity_convergence[nindex, 1] = errors_u[nindex, 0] * (2.**(-expected_velocity_convergence))**(len(resolutions)-1)
    theoretical_pressure_convergence[nindex, 0] = errors_p[nindex, 0]
    theoretical_pressure_convergence[nindex, 1] = errors_p[nindex, 0] * (2.**(-expected_pressure_convergence))**(len(resolutions)-1)
    f.close()
    os.chdir(retvallm)

# Now make a plot for all ns:
# First velocity:
for nindex, n in enumerate(ns):
    pylab.semilogy(levels, errors_u[nindex, :], 'k', ls="none", marker=symbols[nindex], markersize=10, alpha=0.7, label=rf"n = {n}")
pylab.semilogy((levels[0], levels[-1]), theoretical_velocity_convergence[-2, :], "k--", label=r"$\mathcal{O} (\Delta x^{4.0}$)")
pylab.xlabel("Level of Refinement")
pylab.xlim(levels[0]-0.5, levels[-1]+0.5)
pylab.xticks(levels)
pylab.ylabel(r"Error, $|| \mathbf{u} - \mathbf{u}^{*} ||_{2} \; / \;  || \mathbf{u}^{*} ||_{2}$")
pylab.grid()
pylab.legend(numpoints=1)
pylab.savefig("FS_Velocity_Convergence.png", bbox_inches="tight")
pylab.savefig("FS_Velocity_Convergence.pdf", bbox_inches="tight")
pylab.close()

# Now pressure:
for nindex, n in enumerate(ns):
    pylab.semilogy(levels, errors_p[nindex, :], 'k', ls="none", marker=symbols[nindex], markersize=10, alpha=0.7, label=rf"n = {n}")
pylab.semilogy((levels[0], levels[-1]), theoretical_pressure_convergence[-2, :], "k--", label=r"$\mathcal{O} (\Delta x^{2.0}$)")
pylab.xlabel("Level of Refinement")
pylab.xlim(levels[0]-0.5, levels[-1]+0.5)
pylab.xticks(levels)
pylab.ylabel(r"Error, $|| p - p^{*} ||_{2} \; / \; || p^{*} ||_{2}$")
pylab.grid()
pylab.legend(numpoints=1)
pylab.savefig("FS_Pressure_Convergence.png", bbox_inches="tight")
pylab.savefig("FS_Pressure_Convergence.pdf", bbox_inches="tight")
pylab.close()
