"""Cartesian King TALA steady benchmark using G-ADOPT equation terms.

Run this explicitly as a numerical experiment, not a stored-reference update.
The three-field steady solve retains the constant-pressure nullspace.
All physical momentum, continuity and energy terms come from G-ADOPT.
"""

import argparse
import json
from pathlib import Path
import time

import firedrake as fd
import numpy as np

import gadopt
from gadopt import scalar_equation
from gadopt.equations import Equation
from gadopt.momentum_equation import momentum_terms, divergence_term


def run(args):
    start = time.monotonic()
    mesh = fd.UnitSquareMesh(args.n, args.n, quadrilateral=True)
    mesh.cartesian = True
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    P = fd.FunctionSpace(mesh, "CG", 1)
    Q = fd.FunctionSpace(mesh, "CG", 2)
    Z = V * P * Q
    state = fd.Function(Z, name="State")
    u, p, temperature = fd.split(state)
    v, pressure_test, temperature_test = fd.TestFunctions(Z)
    x, y = fd.SpatialCoordinate(mesh)
    density = fd.exp(args.di * (1 - y))
    surface = 273.0 / 3000.0
    reference = surface * fd.exp(args.di * (1 - y))
    if args.formulation in ("full", "surface-relative"):
        approximation = gadopt.FullTemperatureTruncatedAnelasticLiquidApproximation(
            args.ra, args.di, reference_temperature=reference, rho=density,
            temperature_offset=surface if args.formulation == "surface-relative" else 0)
        q = approximation.absolute_temperature(temperature)
        theta = approximation.temperature_anomaly(temperature)
        offset = surface if args.formulation == "surface-relative" else 0
        top, bottom = surface-offset, surface+1-offset
    else:
        approximation = gadopt.TruncatedAnelasticLiquidApproximation(
            args.ra, args.di, Tbar=reference - surface, rho=density)
        q = temperature + reference
        theta = temperature
        top, bottom = 0, 1 - surface * (np.exp(args.di) - 1)

    momentum = Equation(
        v, V, momentum_terms,
        eq_attrs={"p": p, "stress": approximation.stress(u),
                  "source": approximation.buoyancy(p, temperature)
                  * fd.as_vector((0, 1))},
        approximation=approximation, quad_degree=args.quadrature)
    continuity = Equation(
        pressure_test, P, divergence_term,
        eq_attrs={"u": u, "rho_continuity": approximation.rho_continuity()},
        approximation=approximation, quad_degree=args.quadrature)
    energy = Equation(
        temperature_test, Q,
        [scalar_equation.advection_term, scalar_equation.diffusion_term,
         scalar_equation.sink_term, scalar_equation.source_term],
        eq_attrs={"u": u, "advective_velocity_scaling": approximation.rhocp(),
                  "diffusivity": approximation.kappa(),
                  "reference_for_diffusion": approximation.Tbar,
                  "sink_coeff": approximation.linearized_energy_sink(u),
                  "source": approximation.energy_source(u)},
        approximation=approximation, quad_degree=args.quadrature)
    dx = fd.dx(domain=mesh, degree=args.quadrature)
    ds = fd.ds(domain=mesh, degree=args.quadrature)
    residual = ((1 / args.ra) * momentum.residual(u) + continuity.residual(p)
                + energy.residual(temperature))
    bcs = [fd.DirichletBC(Z.sub(0).sub(0), 0, (1, 2)),
           fd.DirichletBC(Z.sub(0).sub(1), 0, (3, 4)),
           fd.DirichletBC(Z.sub(2), bottom, 3),
           fd.DirichletBC(Z.sub(2), top, 4)]
    initial_q = surface + 1 - y + 0.2 * fd.cos(fd.pi*x) * fd.sin(fd.pi*y)
    if args.initial:
        # Interpolate checkpoint fields between meshes with Firedrake, not NumPy.
        with fd.CheckpointFile(args.initial, "r") as checkpoint:
            old_mesh = checkpoint.load_mesh()
            old_u = checkpoint.load_function(old_mesh, "Velocity")
            old_p = checkpoint.load_function(old_mesh, "Pressure")
            old_q = checkpoint.load_function(old_mesh, "AbsoluteTemperature")
        state.subfunctions[0].interpolate(old_u)
        state.subfunctions[1].interpolate(old_p)
        state.subfunctions[2].interpolate(old_q)
        if args.formulation == "perturbation":
            state.subfunctions[2].interpolate(state.subfunctions[2] - reference)
        elif args.formulation == "surface-relative":
            state.subfunctions[2].interpolate(state.subfunctions[2] - surface)
    else:
        state.subfunctions[0].interpolate(fd.as_vector((
            -25 * fd.pi * fd.sin(fd.pi*x) * fd.cos(fd.pi*y) / density,
            25 * fd.pi * fd.cos(fd.pi*x) * fd.sin(fd.pi*y) / density)))
        state.subfunctions[2].interpolate(
            initial_q - (reference if args.formulation == "perturbation" else offset))
    for bc in bcs:
        bc.apply(state)

    nullspace = fd.MixedVectorSpaceBasis(
        Z, [Z.sub(0), fd.VectorSpaceBasis(constant=True, comm=mesh.comm), Z.sub(2)])
    solver = fd.NonlinearVariationalSolver(
        fd.NonlinearVariationalProblem(residual, state, bcs=bcs),
        solver_parameters={"snes_type": "newtonls", "snes_linesearch_type": "bt",
                           "snes_rtol": 1e-10, "snes_atol": 1e-10,
                           "snes_max_it": 40, "snes_monitor": None,
                           "ksp_type": "preonly", "pc_type": "lu",
                           "pc_factor_mat_solver_type": "mumps",
                           "mat_mumps_icntl_24": 1,
                           "mat_type": "aij"},
        nullspace=nullspace, transpose_nullspace=nullspace,
        options_prefix="king_steady")
    solver.solve()

    # Compare/save pressure in one gauge; TALA is invariant to a constant shift.
    state.subfunctions[1].assign(state.subfunctions[1] - fd.assemble(p * dx))

    normal = fd.FacetNormal(mesh)
    work_full = approximation.linearized_energy_sink(u) * q
    work_perturbation = approximation.linearized_energy_sink(u) * theta
    phi = approximation.viscous_dissipation(u)
    # Boundary reactions from the steady energy residual with affine lifts.
    # They converge faster than directly differentiating the boundary FE field.

    def reaction(lift):
        return fd.assemble((
            approximation.rhocp() * lift * fd.dot(u, fd.grad(temperature))
            + fd.dot(fd.grad(lift), fd.grad(q))
            + lift * approximation.linearized_energy_sink(u) * temperature
            - lift * approximation.energy_source(u)) * dx)

    volume = fd.assemble(fd.Constant(1) * dx)
    nu_top = -fd.assemble(fd.dot(fd.grad(q), normal) * ds(4))
    nu_bottom = fd.assemble(fd.dot(fd.grad(q), normal) * ds(3))
    phi_integral = fd.assemble(phi * dx)
    work_integral = fd.assemble(work_full * dx)
    advection_integral = fd.assemble(
        approximation.rhocp() * fd.dot(u, fd.grad(temperature)) * dx)
    nu_top_reaction, nu_bottom_reaction = -reaction(y), reaction(1-y)
    result = {
        "ra": args.ra, "di": args.di, "n": args.n,
        "formulation": args.formulation, "quadrature": args.quadrature,
        "mpi_size": mesh.comm.size, "gadopt_path": gadopt.__file__,
        "initial_checkpoint": args.initial,
        "temperature_surface": surface, "temperature_bottom": surface + 1,
        "evolved_temperature_surface": top, "evolved_temperature_bottom": bottom,
        "temperature_offset": None if args.formulation == "perturbation" else offset,
        "nu_top_gradient": nu_top, "nu_bottom_gradient": nu_bottom,
        "nu_top_reaction": nu_top_reaction, "nu_bottom_reaction": nu_bottom_reaction,
        "vrms": np.sqrt(fd.assemble(fd.dot(u, u) * dx) / volume),
        "surface_vrms": np.sqrt(fd.assemble(fd.dot(u, u) * ds(4))),
        "mean_surface_relative_temperature": fd.assemble((q-surface) * dx) / volume,
        "mean_absolute_temperature": fd.assemble(q * dx) / volume,
        "viscous_heating": phi_integral,
        "adiabatic_work_full": work_integral,
        "adiabatic_work_perturbation": fd.assemble(work_perturbation * dx),
        "advective_heat_integral_evolved": advection_integral,
        "reaction_energy_defect": (nu_top_reaction - nu_bottom_reaction
                                   - fd.assemble(approximation.energy_source(u) * dx)
                                   + fd.assemble(approximation.linearized_energy_sink(u)
                                                 * temperature * dx) + advection_integral),
        "gradient_energy_defect": nu_top - nu_bottom - phi_integral + work_integral,
        "mass_divergence_l2": np.sqrt(fd.assemble(fd.div(density*u)**2 * dx)),
        "mean_pressure": fd.assemble(p * dx) / volume,
        "snes_iterations": solver.snes.getIterationNumber(),
        "snes_residual": solver.snes.getFunctionNorm(),
        "snes_reason": int(solver.snes.getConvergedReason()),
        "seconds": time.monotonic() - start,
    }
    if result["vrms"] < 1:
        raise RuntimeError("Newton converged to conduction, not the convective branch")
    prefix = Path(args.output)
    if mesh.comm.rank == 0:
        prefix.parent.mkdir(parents=True, exist_ok=True)
    mesh.comm.barrier()
    absolute = fd.Function(Q, name="AbsoluteTemperature").interpolate(q)
    velocity, pressure = state.subfunctions[:2]
    with fd.CheckpointFile(str(prefix)+".h5", "w") as checkpoint:
        checkpoint.save_mesh(mesh)
        checkpoint.save_function(absolute, name="AbsoluteTemperature")
        checkpoint.save_function(velocity, name="Velocity")
        checkpoint.save_function(pressure, name="Pressure")
    # Owned nodal data are gathered only for portable plotting/comparison output.
    coords = fd.Function(fd.VectorFunctionSpace(mesh, "CG", 2)).interpolate(
        fd.SpatialCoordinate(mesh))
    parts = mesh.comm.gather((coords.dat.data_ro.copy(), absolute.dat.data_ro.copy(),
                             velocity.dat.data_ro.copy()), root=0)
    if mesh.comm.rank == 0:
        xy = np.concatenate([part[0] for part in parts])
        temperatures = np.concatenate([part[1] for part in parts])
        velocities = np.concatenate([part[2] for part in parts])
        order = np.lexsort((xy[:, 1], xy[:, 0]))
        np.savez(str(prefix)+".npz", xy=xy[order], temperature=temperatures[order],
                 velocity=velocities[order])
        Path(str(prefix)+".json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ra", type=float, default=1e4)
    parser.add_argument("--di", type=float, default=0.5)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--quadrature", type=int, default=8)
    parser.add_argument("--formulation", choices=["full", "surface-relative", "perturbation"],
                        default="surface-relative")
    parser.add_argument("--initial")
    parser.add_argument("--output", default="king_result")
    run(parser.parse_args())
