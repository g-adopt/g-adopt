import adjoint_small as adjoint
import numpy as np
import pyadjoint
import sys

from checkpoint_schedules import SingleDiskStorageSchedule
from checkpoint_schedules import SingleMemoryStorageSchedule
from firedrake import Function
from firedrake.adjoint_utils.blocks import FunctionAssignBlock

match sys.argv[1]:
    case "memory":
        scheduler = SingleMemoryStorageSchedule()

    case "disk":
        scheduler = SingleDiskStorageSchedule()

    case "none":
        scheduler = None

    case _:
        raise ValueError

rng = np.random.default_rng(42)

Tic, reduced_functional = adjoint.forward_problem(scheduler=scheduler, age0=10.0, timesteps=int(sys.argv[2]) + 1)

tape = pyadjoint.get_working_tape()
blocks = tape.get_blocks()

with open(f"blocks_{sys.argv[1]}.txt", "w") as f:
    for block in blocks:
        f.write(repr(block) + "\n")

dtemp = Function(Tic.function_space())
dtemp.dat.data_wo[:] = rng.random(dtemp.dat.data_ro.shape)

# Jm = reduced_functional(Tic)
# print(f"reduced functional: {Jm}")

"""
for block in blocks:
    if isinstance(block, FunctionAssignBlock):
        print(block)
"""

dJdm = reduced_functional.derivative()
dJdm = dtemp._ad_dot(dJdm)
print(f"dJdm 1: {dJdm}")

"""
for block in blocks:
    if isinstance(block, FunctionAssignBlock):
        print(block)
"""
Tic.assign(Tic + 0.1)
Jm = reduced_functional(Tic)
print(f"reduced functional: {Jm}")
"""
for block in blocks:
    if isinstance(block, FunctionAssignBlock):
        print(block)
"""

dJdm = reduced_functional.derivative()
dJdm = dtemp._ad_dot(dJdm)
print(f"dJdm 2: {dJdm}")

"""
for block in blocks:
    if isinstance(block, FunctionAssignBlock):
        print(block)
"""

Jm = reduced_functional(Tic)
print(f"reduced functional: {Jm}")
dJdm = reduced_functional.derivative()
dJdm = dtemp._ad_dot(dJdm)
print(f"dJdm 3: {dJdm}")

#sys.exit(0)

def perturbe(eps):
    return Tic._ad_add(dtemp._ad_mul(eps))


epsilons = [0.01 / 2 ** i for i in range(4)]
residuals = []
for eps in epsilons:
    Jp = reduced_functional(perturbe(eps))
    residuals.append(abs(Jp - Jm - eps * dJdm))

for eps, res in zip(epsilons, residuals):
    print(f"eps: {eps}, res: {res}")
