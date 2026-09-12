"""Compare both temperature variables through the standard G-ADOPT solvers."""

import argparse
import json
from pathlib import Path

import firedrake as fd

from gadopt import (
    EnergySolver, FullTemperatureTruncatedAnelasticLiquidApproximation,
    ImplicitMidpoint, StokesSolver, TruncatedAnelasticLiquidApproximation,
    create_stokes_nullspace,
)


def run(args):
    with fd.CheckpointFile(args.initial, "r") as checkpoint:
        mesh = checkpoint.load_mesh()
        velocity = checkpoint.load_function(mesh, "Velocity")
        pressure = checkpoint.load_function(mesh, "Pressure")
        absolute = checkpoint.load_function(mesh, "AbsoluteTemperature")
    mesh.cartesian = True
    V, P, Q = velocity.function_space(), pressure.function_space(), absolute.function_space()
    Z = V * P
    x, y = fd.SpatialCoordinate(mesh)
    surface, di, ra = 0.091, 0.5, 1e4
    reference = surface * fd.exp(di*(1-y))
    rho = fd.exp(di*(1-y))
    initial = fd.Function(Q).interpolate(
        absolute + 0.005 * fd.cos(fd.pi*x) * fd.sin(fd.pi*y))
    models = []
    for full in (True, False):
        T = fd.Function(Q).interpolate(initial if full else initial-reference)
        state = fd.Function(Z)
        state.subfunctions[0].assign(velocity)
        state.subfunctions[1].assign(pressure)
        u, _ = fd.split(state)
        if full:
            approximation = FullTemperatureTruncatedAnelasticLiquidApproximation(
                ra, di, rho=rho, reference_temperature=reference)
            bottom, top = surface+1, surface
        else:
            approximation = TruncatedAnelasticLiquidApproximation(
                ra, di, rho=rho, Tbar=reference-surface)
            bottom, top = 1 - surface*(fd.exp(di)-1), 0
        nullspace = create_stokes_nullspace(Z, closed=True, rotational=False)
        parameters = {"snes_type": "ksponly", "ksp_type": "preonly", "pc_type": "lu",
                      "pc_factor_mat_solver_type": "mumps", "mat_type": "aij"}
        stokes = StokesSolver(
            state, approximation, T,
            bcs={1: {"ux": 0}, 2: {"ux": 0}, 3: {"uy": 0}, 4: {"uy": 0}},
            nullspace=nullspace, transpose_nullspace=nullspace,
            solver_parameters=parameters | {"mat_mumps_icntl_24": 1},
            constant_jacobian=True, quad_degree=8)
        energy = EnergySolver(
            T, u, approximation, fd.Constant(args.dt), ImplicitMidpoint,
            bcs={3: {"T": bottom}, 4: {"T": top}}, solver_parameters=parameters)
        models.append((T, state, stokes, energy))

    def l2(expression):
        return fd.sqrt(fd.assemble(fd.inner(expression, expression) * fd.dx))

    rows = []
    for step in range(args.steps):
        for _, _, stokes, energy in models:
            stokes.solve()
            energy.solve()
        q_full = models[0][0]
        q_perturbation = models[1][0]+reference
        u_full = models[0][1].subfunctions[0]
        u_perturbation = models[1][1].subfunctions[0]
        rows.append({"step": step+1, "time": (step+1)*args.dt,
                     "temperature_relative_l2": float(l2(q_full-q_perturbation)/l2(q_full)),
                     "velocity_relative_l2": float(l2(u_full-u_perturbation)/l2(u_full)),
                     "temperature_change_l2": float(l2(q_full-initial))})
    assert rows[-1]["temperature_change_l2"] > 1e-5
    assert max(row["temperature_relative_l2"] for row in rows) < 1e-5
    assert max(row["velocity_relative_l2"] for row in rows) < 1e-5
    if mesh.comm.rank == 0:
        output = {"dt": args.dt, "steps": args.steps,
                  "mpi_size": mesh.comm.size, "rows": rows}
        Path(args.output).write_text(json.dumps(output, indent=2)+"\n")
        print(json.dumps(rows[-1]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", required=True)
    parser.add_argument("--dt", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--output", default="transient_check.json")
    run(parser.parse_args())
