#!/usr/bin/env python3

import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
import pandas as pd
import sys

from scaling import cases, get_data

preamble = sys.argv[1]

rcParams['axes.titlesize'] = 16
rcParams['axes.labelsize'] = 16
rcParams['xtick.labelsize'] = 16
rcParams['ytick.labelsize'] = 16
rcParams['legend.fontsize'] = 15

cores = []
frames = []
for level, conf in cases.items():
    try:
        data = get_data(level)
    except FileNotFoundError:
        continue

    cores.append(conf["cores"])
    df = pd.DataFrame(data, index=[level])
    frames.append(df)

df = pd.concat(frames)
df["assembly_time"] = df["snes_function"] + df["snes_jacobian"]
df["velocity_cost"] = (df["stokes_solve"] - df["pc_setup"]) / df["velocity_iterations"]


def plot_quantity(quantity, label, suffix, xlim=(10, 3e4), ylim=None, yticks=None):
    plt.semilogx(cores, quantity, "ko", linestyle="none", markersize=10)

    plt.xlabel("# Cores")
    plt.ylabel(label)

    plt.xlim(*xlim)

    if ylim is not None:
        plt.ylim(*ylim)
    if yticks is not None:
        plt.yticks(yticks)
    plt.grid()

    for ext in ("png", "pdf"):
        plt.savefig(f"{preamble}_{suffix}.{ext}", bbox_inches="tight")

    plt.close()


plot_quantity(df["stokes_solve"], "Stokes Solve (s)", "Schur_Solve_time",
              ylim=(0, 16000), yticks=np.linspace(0, 16000, 5))
plot_quantity(df["energy_solve"], "Energy Solve (s)", "Energy_Solve_time",
              ylim=(8, 14), yticks=np.linspace(8, 14, 4))
plot_quantity(df["snes_function"], "SNES Function (s)", "SNES_function_time")
plot_quantity(df["snes_jacobian"], "SNES Jacobian (s)", "SNES_jacobian_time")
plot_quantity(df["assembly_time"], "Assembly (s)", "assembly_time",
              ylim=(0, 12), yticks=np.linspace(0, 12, 4))
plot_quantity(df["pc_setup"], "PC Setup (s)", "PC_Setup_time",
              ylim=(0, 600), yticks=np.linspace(0, 600, 7))
plot_quantity(df["pressure_iterations"], "# Pressure Solve Iterations", "Pressure_Iterations",
              ylim=(9, 11), yticks=np.linspace(9, 11, 5))
plot_quantity(df["velocity_iterations"], "# Velocity Solve Iterations", "Velocity_Iterations",
              ylim=(20, 40), yticks=np.linspace(20, 40, 5))
plot_quantity(df["energy_iterations"], "# Energy Solve Iterations", "Energy_Iterations",
              ylim=(11, 13), yticks=np.linspace(11, 13, 5))
plot_quantity(df["velocity_cost"], "Per Velocity Iteration (s)", "Velocity_Cost_Per_Iteration",
              ylim=(0, 300), yticks=np.linspace(0, 300, 4))
plot_quantity(df["total_time"], "Total Time (s)", "Total_Time",
              ylim=(0, 16000), yticks=np.linspace(0, 16000, 5))

# Combined plot of iterations per phase
plt.semilogx(cores, df["velocity_iterations"], 'go', linestyle="none", markersize=10, label="Velocity")
plt.semilogx(cores, df["pressure_iterations"], 'rs', linestyle="none", markersize=10, label="Pressure")
plt.semilogx(cores, df["energy_iterations"], 'b*', linestyle="none", markersize=10, label="Energy")
plt.xlabel("# Cores")
plt.ylabel("# Solve Iterations")
plt.xlim(10, 3e4)
plt.ylim(0, 80)
plt.yticks(np.linspace(0, 80, 5))
plt.grid()
plt.legend(numpoints=1)
for ext in ("png", "pdf"):
    plt.savefig(f"{preamble}_Combined_Iterations.{ext}", bbox_inches="tight")
plt.close()
