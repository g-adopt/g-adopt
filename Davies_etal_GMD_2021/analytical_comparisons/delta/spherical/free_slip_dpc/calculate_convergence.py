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
rcParams['legend.fontsize'] = 11

# Range:
ls = numpy.array([2, 4, 8])
resolutions = ["A", "B", "C", "D"]
levels = [3, 4, 5, 6]
expected_velocity_convergence = 3.5
expected_pressure_convergence = 2.0
symbols = ["o", ".", "s", "*", "^", ">"]
colours = ["r", "b", "r", "b", "r", "b"]

# Arrays for storing errors:
errors_u = numpy.zeros((len(ls), 2, len(resolutions)))
errors_p = numpy.zeros((len(ls), 2, len(resolutions)))
theoretical_velocity_convergence = numpy.zeros((len(ls), 2, 2))
theoretical_pressure_convergence = numpy.zeros((len(ls), 2, 2))

# Loop over cases:
for lindex, l in enumerate(ls):
    ms = numpy.array([l/2, l], dtype=int)
    for mindex, m in enumerate(ms):
        if m <= l:
            lm_dir_name = f"l{l}_m{m}"
            print("Working on directory: ", lm_dir_name)
            retvallm = os.getcwd()
            os.chdir(lm_dir_name)
            # First grab l2 norm of analytical solutions at finest level (for relative error):
            anal_vel = numpy.loadtxt(f"{resolutions[-1]}/errors.log")[2]
            anal_pres = numpy.loadtxt(f"{resolutions[-1]}/errors.log")[3]
            f = open(f"convergence_l{l}_m{m}.log", "w")
            for resindex, resolution in enumerate(resolutions):
                retvalres = os.getcwd()
                os.chdir(resolution)
                errors = numpy.loadtxt("errors.log")
                errors_u[lindex, mindex, resindex] = errors[0] / anal_vel  # relative u error
                errors_p[lindex, mindex, resindex] = errors[1] / anal_pres  # relative p error
                if resindex == 0:
                    print(errors_u[lindex, mindex, resindex], errors_p[lindex, mindex, resindex])
                else:
                    conv_u = numpy.log2(errors_u[lindex, mindex, resindex-1]/errors_u[lindex, mindex, resindex])
                    conv_p = numpy.log2(errors_p[lindex, mindex, resindex-1]/errors_p[lindex, mindex, resindex])
                    print(errors_u[lindex, mindex, resindex], errors_p[lindex, mindex, resindex], "|", conv_u, conv_p)
                os.chdir(retvalres)
            theoretical_velocity_convergence[lindex, mindex, 0] = errors_u[lindex, mindex, 0]
            theoretical_velocity_convergence[lindex, mindex, 1] = errors_u[lindex, mindex, 0] * (2.**(-expected_velocity_convergence))**(len(resolutions)-1)
            theoretical_pressure_convergence[lindex, mindex, 0] = errors_p[lindex, mindex, 0]
            theoretical_pressure_convergence[lindex, mindex, 1] = errors_p[lindex, mindex, 0] * (2.**(-expected_pressure_convergence))**(len(resolutions)-1)
            f.close()
            os.chdir(retvallm)

# Now plot:
# First velocity:
increment = 0
for lindex, l in enumerate(ls):
    ms = numpy.array([l/2, l], dtype=int)
    for mindex, m in enumerate(ms):
        pylab.semilogy(levels, errors_u[lindex, mindex, :], ls="none", marker=symbols[increment], color=colours[increment], markersize=10, alpha=0.7, label=rf"l{l}_m{m}")
        increment += 1
pylab.semilogy((levels[0], levels[-1]), theoretical_velocity_convergence[1, -1, :], "k--", label=r"$\mathcal{O} (\Delta x^{4.0}$)")
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
increment = 0
for lindex, l in enumerate(ls):
    ms = numpy.array([l/2, l], dtype=int)
    for mindex, m in enumerate(ms):
        pylab.semilogy(levels, errors_p[lindex, mindex, :], ls="none", marker=symbols[increment], color=colours[increment], markersize=10, alpha=0.7, label=rf"l{l}_m{m}")
        increment += 1
pylab.semilogy((levels[0], levels[-1]), theoretical_pressure_convergence[1, -1, :], "k--", label=r"$\mathcal{O} (\Delta x^{2.0}$)")
pylab.xlabel("Level of Refinement")
pylab.xlim(levels[0]-0.5, levels[-1]+0.5)
pylab.xticks(levels)
pylab.ylabel(r"Error, $|| p - p^{*} ||_{2} \; / \; || p^{*} ||_{2}$")
pylab.grid()
pylab.legend(numpoints=1)
pylab.savefig("FS_Pressure_Convergence.png", bbox_inches="tight")
pylab.savefig("FS_Pressure_Convergence.pdf", bbox_inches="tight")
pylab.close()
