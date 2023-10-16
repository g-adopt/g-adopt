#! /usr/bin/env python3

import numpy

# Read file:
f = open("output_2.dat", "r").readlines()

# First extract timing information

# Filter for relevant solves:
solves = [l for l in f if "_solve:" in l]
SNESFunction = [l for l in f if "SNESFunctionEval" in l]
SNESJacobian = [l for l in f if "SNESJacobianEval" in l]
PC_setup = [l for l in f if "PCSetUp" in l]
SNESSolve = [l for l in f if "SNESSolve" in l]
TotalTime = [l for l in f if "Time (sec):" in l]

# Extract relevant columns:
solve_timings = numpy.array([l.split()[2] for l in solves], dtype=float)
SNESFunction_timings = numpy.array([l.split()[3] for l in SNESFunction], dtype=float)
SNESJacobian_timings = numpy.array([l.split()[3] for l in SNESJacobian], dtype=float)
PC_setup_timings = numpy.array([l.split()[3] for l in PC_setup], dtype=float)
SNESSolve_timings = numpy.array([l.split()[3] for l in SNESSolve], dtype=float)
Total_timings = numpy.array([l.split()[2] for l in TotalTime], dtype=float)

# Now extract iterations
velocity_solve_iterations = [l for l in f if "Linear firedrake_0_fieldsplit_0_ solve converged due to" in l]
pressure_solve_iterations = [l for l in f if "Linear firedrake_0_fieldsplit_1_ solve converged due to" in l]
energy_solve_iterations = [l for l in f if "Linear firedrake_1_ solve converged due to" in l]
average_velocity_iterations = numpy.mean(numpy.array([l.split()[-1] for l in velocity_solve_iterations], dtype=float))
average_pressure_iterations = numpy.mean(numpy.array([l.split()[-1] for l in pressure_solve_iterations], dtype=float))
average_energy_iterations = numpy.mean(numpy.array([l.split()[-1] for l in energy_solve_iterations], dtype=float))

# Output data:
timings_iterations = numpy.zeros(9)
timings_iterations[0] = solve_timings[0]
timings_iterations[1] = solve_timings[1]
timings_iterations[2] = SNESFunction_timings[0]
timings_iterations[3] = SNESJacobian_timings[0]
timings_iterations[4] = PC_setup_timings[0]
timings_iterations[5] = average_velocity_iterations
timings_iterations[6] = average_pressure_iterations
timings_iterations[7] = average_energy_iterations
timings_iterations[8] = Total_timings
numpy.savetxt("Timings_Iterations.dat", timings_iterations)
