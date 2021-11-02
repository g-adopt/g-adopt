#! /usr/bin/env python3

import numpy
import pylab
import sys
from matplotlib import rcParams

preamble = sys.argv[1]

# Matplotlib defaults:
rcParams['axes.titlesize'] = 15
rcParams['axes.labelsize'] = 15
rcParams['xtick.labelsize'] = 15
rcParams['ytick.labelsize'] = 15
rcParams['legend.fontsize'] = 15

# First, gather data

# dirs  = ["Level_4", "Level_5","Level_6", "Level_7"]
# cores = [24, 192, 1536, 12288]

dirs = ["Level_5", "Level_6", "Level_7", "Level_8"]
cores = [24, 192, 1536, 12288]

# Set up arrays:
schur_solve_time = numpy.zeros((len(dirs), 1))
energy_solve_time = numpy.zeros((len(dirs), 1))
SNES_function_time = numpy.zeros((len(dirs), 1))
SNES_jacobian_time = numpy.zeros((len(dirs), 1))
assembly_time = numpy.zeros((len(dirs), 1))
PC_setup_time = numpy.zeros((len(dirs), 1))
mean_velocity_its = numpy.zeros((len(dirs), 1))
mean_pressure_its = numpy.zeros((len(dirs), 1))
mean_energy_its = numpy.zeros((len(dirs), 1))

# Loop over directiories and store data:
for dir in range(len(dirs)):
    data = numpy.loadtxt(dirs[dir]+"/Timings_Iterations.dat")
    schur_solve_time[dir, :] = data[0]
    energy_solve_time[dir, :] = data[1]
    SNES_function_time[dir, :] = data[2]
    SNES_jacobian_time[dir, :] = data[3]
    PC_setup_time[dir, :] = data[4]
    mean_velocity_its[dir, :] = data[5]
    mean_pressure_its[dir, :] = data[6]
    mean_energy_its[dir, :] = data[7]

assembly_time = SNES_function_time + SNES_jacobian_time

# Now plot:
# Schur solve time
pylab.plot(cores, schur_solve_time, 'ro', linestyle="none", markersize=8)
pylab.xlabel("# Cores")
pylab.ylabel("Stokes Solve (s)")
pylab.xlim(-500, 12500)
pylab.xticks(numpy.linspace(0, 12500, 6))
pylab.ylim(7000, 13000)
pylab.yticks(numpy.linspace(7000, 13000, 4))
pylab.grid()
pylab.savefig(preamble+"_Schur_Solve_time.png", bbox_inches="tight")
pylab.savefig(preamble+"_Schur_Solve_time.pdf", bbox_inches="tight")
pylab.close()

# Energy solve time
pylab.plot(cores, energy_solve_time, 'ro', linestyle="none", markersize=8)
pylab.xlabel("# Cores")
pylab.ylabel("Energy Solve (s)")
pylab.xlim(-500, 12500)
pylab.xticks(numpy.linspace(0, 12500, 6))
pylab.ylim(8, 14)
pylab.yticks(numpy.linspace(8, 14, 4))
pylab.grid()
pylab.savefig(preamble+"_Energy_Solve_time.png", bbox_inches="tight")
pylab.savefig(preamble+"_Energy_Solve_time.pdf", bbox_inches="tight")
pylab.close()

# SNES Function time
pylab.plot(cores, SNES_function_time, 'ro', linestyle="none", markersize=8)
pylab.xlabel("# Cores")
pylab.ylabel("SNES Function (s)")
pylab.xlim(-500, 12500)
pylab.xticks(numpy.linspace(0, 12500, 6))
pylab.grid()
pylab.savefig(preamble+"_SNES_function_time.png", bbox_inches="tight")
pylab.savefig(preamble+"_SNES_function_time.pdf", bbox_inches="tight")
pylab.close()

# SNES Jacobian time
pylab.plot(cores, SNES_jacobian_time, 'ro', linestyle="none", markersize=8)
pylab.xlabel("# Cores")
pylab.ylabel("SNES Jacobian (s)")
pylab.xlim(-500, 12500)
pylab.xticks(numpy.linspace(0, 12500, 6))
pylab.grid()
pylab.savefig(preamble+"_SNES_jacobian_time.png", bbox_inches="tight")
pylab.savefig(preamble+"_SNES_jacobian_time.pdf", bbox_inches="tight")
pylab.close()

# Assembly time
pylab.plot(cores, assembly_time, 'ro', linestyle="none", markersize=8)
pylab.xlabel("# Cores")
pylab.ylabel("Assembly (s)")
pylab.xlim(-500, 12500)
pylab.xticks(numpy.linspace(0, 12500, 6))
pylab.ylim(4, 10)
pylab.yticks(numpy.linspace(4, 10, 4))
pylab.grid()
pylab.savefig(preamble+"_assembly_time.png", bbox_inches="tight")
pylab.savefig(preamble+"_assembly_time.pdf", bbox_inches="tight")
pylab.close()

# PC Setup Time
pylab.plot(cores, PC_setup_time, 'ro', linestyle="none", markersize=8)
pylab.xlabel("# Cores")
pylab.ylabel("PC Setup (s)")
pylab.xlim(-500, 12500)
pylab.xticks(numpy.linspace(0, 12500, 6))
pylab.ylim(200, 500)
pylab.yticks(numpy.linspace(200, 500, 4))
pylab.grid()
pylab.savefig(preamble+"_PC_Setup_time.png", bbox_inches="tight")
pylab.savefig(preamble+"_PC_Setup_time.pdf", bbox_inches="tight")
pylab.close()

# Pressure iterations
pylab.plot(cores, mean_pressure_its, 'ro', linestyle="none", markersize=8)
pylab.xlabel("# Cores")
pylab.ylabel("# Pressure Solve Iterations")
pylab.xlim(-500, 12500)
pylab.xticks(numpy.linspace(0, 12500, 6))
pylab.ylim(9, 11)
pylab.yticks(numpy.linspace(9, 11, 5))
pylab.grid()
pylab.savefig(preamble+"_Pressure_Iterations.png", bbox_inches="tight")
pylab.savefig(preamble+"_Pressure_Iterations.pdf", bbox_inches="tight")
pylab.close()

# Velocity iterations
pylab.plot(cores, mean_velocity_its, 'ro', linestyle="none", markersize=8)
pylab.xlabel("# Cores")
pylab.ylabel("# Velocity Solve Iterations")
pylab.xlim(-500, 12500)
pylab.xticks(numpy.linspace(0, 12500, 6))
pylab.ylim(40, 56)
pylab.yticks(numpy.linspace(40, 56, 5))
pylab.grid()
pylab.savefig(preamble+"_Velocity_Iterations.png", bbox_inches="tight")
pylab.savefig(preamble+"_Velocity_Iterations.pdf", bbox_inches="tight")
pylab.close()

# Energy iterations
pylab.plot(cores, mean_energy_its, 'ro', linestyle="none", markersize=8)
pylab.xlabel("# Cores")
pylab.ylabel("# Energy Solve Iterations")
pylab.xlim(-500, 12500)
pylab.xticks(numpy.linspace(0, 12500, 6))
pylab.ylim(11, 13)
pylab.yticks(numpy.linspace(11, 13, 5))
pylab.grid()
pylab.savefig(preamble+"_Energy_Iterations.png", bbox_inches="tight")
pylab.savefig(preamble+"_Energy_Iterations.pdf", bbox_inches="tight")
pylab.close()

# Make some combination plots:
# Iterations
pylab.plot(cores, mean_energy_its, 'mo', linestyle="none", markersize=8, label="Energy")
pylab.plot(cores, mean_velocity_its, 'g*', linestyle="none", markersize=8, label="Velocity")
pylab.plot(cores, mean_pressure_its, 'bs', linestyle="none", markersize=8, label="Pressure")
pylab.xlabel("# Cores")
pylab.ylabel("# Solve Iterations")
pylab.xlim(-500, 12500)
pylab.xticks(numpy.linspace(0, 12500, 6))
pylab.ylim(0, 60)
pylab.yticks(numpy.linspace(0, 60, 4))
pylab.grid()
pylab.legend(numpoints=1)
pylab.savefig(preamble+"_Combined_Iterations.png", bbox_inches="tight")
pylab.savefig(preamble+"_Combined_Iterations.pdf", bbox_inches="tight")
pylab.close()


# Energy
# fig, ax1 = pylab.subplots()
# color = 'tab:red'
# ax1.set_xlabel('# Cores (s)')
# ax1.set_ylabel('# Energy Solve Iterations', color=color)
# ax1.plot(cores, mean_energy_its, 'o', color=color, linestyle="none", markersize=8)
# ax1.tick_params(axis='y', labelcolor=color)
# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
# color = 'tab:blue'
# ax2.set_ylabel('Energy Solve Time (s)', color=color)  # we already handled the x-label with ax1
# ax2.plot(cores, energy_solve_time, 'o', color=color, linestyle="none", markersize=8)
# ax2.tick_params(axis='y', labelcolor=color)
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# pylab.savefig(preamble+"_Combined_Energy.png",bbox_inches="tight")
# pylab.savefig(preamble+"_Combined_Energy.pdf",bbox_inches="tight")
# pylab.close()
