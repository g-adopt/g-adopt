"""Reference forward run for the box optimisation benchmark.

This script produces the synthetic observations for a twin experiment. It runs the
same isoviscous Boussinesq convection problem as the adjoint tutorial in
``demos/mantle_convection/adjoint``: a unit square, Rayleigh number 1e6, free-slip on
every side, temperature fixed to 1 at the bottom and 0 at the top, Q2-Q1 Stokes and a
DQ2 temperature with an implicit midpoint step.

Two things differ from the tutorial, both for the optimiser comparison:

- The resolution and the number of timesteps are command-line arguments, so a
  cheap version of the problem exists for interface checks and a full one for the
  benchmark.
- The temperature field is written at every timestep. The inverse problem inverts
  the last ``window`` steps of this trajectory, so the true initial temperature of
  that window is the field at timestep ``steps - window``. The twin-experiment error
  is measured against it, and it has to be on disk for that.

Usage:
    python forward.py [--steps 80] [--resolution 150] [--output FILE] [--vtk]

The checkpoint holds the mesh, ``Temperature`` at every step, ``Velocity`` and
``Average_Temperature`` at every step with timestepping information attached.
"""

import argparse

import numpy as np

from gadopt import *


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--steps", type=int, default=80,
                        help="number of timesteps of the reference run (default 80)")
    parser.add_argument("--resolution", type=int, default=150,
                        help="cells per side of the unit square (default 150)")
    parser.add_argument("--delta-t", type=float, default=4e-6,
                        help="timestep length (default 4e-6)")
    parser.add_argument("--output", default="reference-state.h5",
                        help="checkpoint file to write (default reference-state.h5)")
    parser.add_argument("--vtk", action="store_true",
                        help="also write output.pvd for visualisation")
    return parser.parse_args(argv)


def run_forward(steps, resolution, delta_t, output, vtk=False):
    """Run the reference forward model and write the checkpoint.

    Args:
        steps: number of timesteps.
        resolution: cells per side; the mesh is an extruded interval mesh so that
            it matches the tutorial and the layer averaging works.
        delta_t: timestep length in non-dimensional units.
        output: path of the checkpoint file.
        vtk: write a VTK series alongside the checkpoint.
    """
    # Unit square as an extruded mesh. Layer averaging and the ``ds_t`` surface
    # measure both rely on the extruded structure.
    mesh1d = IntervalMesh(resolution, length_or_left=0.0, right=1.0)
    mesh = ExtrudedMesh(mesh1d, layers=resolution, layer_height=1.0 / resolution,
                        extrusion_type="uniform")
    mesh.cartesian = True
    boundary = get_boundary_ids(mesh)

    V = VectorFunctionSpace(mesh, "CG", 2)  # velocity
    W = FunctionSpace(mesh, "CG", 1)  # pressure
    Q = FunctionSpace(mesh, "DQ", 2)  # temperature
    Q1 = FunctionSpace(mesh, "CG", 1)  # layer average, and the control space later
    Z = MixedFunctionSpace([V, W])

    z = Function(Z)
    u, p = z.subfunctions
    u.rename("Velocity")
    p.rename("Pressure")
    T = Function(Q, name="Temperature")

    # Initial temperature: a conductive profile with a boundary layer at each side,
    # plus a Gaussian anomaly that convection will advect. The same expression as
    # the tutorial, so the two demos are comparable.
    X = SpatialCoordinate(mesh)
    T.interpolate(
        0.5 * (erf((1 - X[1]) * 3.0) + erf(-X[1] * 3.0) + 1)
        + 0.1 * exp(-0.5 * ((X - as_vector((0.5, 0.2))) / Constant(0.1)) ** 2)
    )

    # Horizontal layer average of the temperature. The inverse problem damps the
    # control towards this profile, so it is written at every step.
    Taverage = Function(Q1, name="Average Temperature")
    averager = LayerAveraging(mesh, np.linspace(0, 1.0, resolution * 2), quad_degree=6)
    averager.extrapolate_layer_average(Taverage, averager.get_layer_average(T))

    checkpoint_file = CheckpointFile(output, "w")
    checkpoint_file.save_mesh(mesh)

    Ra = Constant(1e6)
    approximation = BoussinesqApproximation(Ra)
    delta_t = Constant(delta_t)

    # Free slip on all four sides, fixed temperature top and bottom.
    stokes_bcs = {
        boundary.bottom: {"uy": 0},
        boundary.top: {"uy": 0},
        boundary.left: {"ux": 0},
        boundary.right: {"ux": 0},
    }
    temp_bcs = {
        boundary.bottom: {"T": 1.0},
        boundary.top: {"T": 0.0},
    }

    # Closed box with no rotation: the pressure nullspace is the only one.
    Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=False)

    energy_solver = EnergySolver(T, u, approximation, delta_t, ImplicitMidpoint, bcs=temp_bcs)
    stokes_solver = StokesSolver(z, approximation, T, bcs=stokes_bcs,
                                 nullspace=Z_nullspace, transpose_nullspace=Z_nullspace)

    output_file = VTKFile("output.pvd") if vtk else None

    # Every field is written with the same index convention as the tutorial:
    # ``Temperature`` at index k is the state at the START of step k, and
    # ``Velocity`` at index k is the velocity computed FROM that state during step k.
    # The final temperature is written at index ``steps - 1`` after the last step,
    # so the last stored temperature is the observation.
    for timestep in range(steps):
        checkpoint_file.save_function(T, name="Temperature", idx=timestep,
                                      timestepping_info={"index": float(timestep),
                                                         "delta_t": float(delta_t)})
        if output_file is not None:
            output_file.write(*z.subfunctions, T, Taverage)

        stokes_solver.solve()
        energy_solver.solve()
        averager.extrapolate_layer_average(Taverage, averager.get_layer_average(T))

        checkpoint_file.save_function(u, name="Velocity", idx=timestep,
                                      timestepping_info={"index": float(timestep),
                                                         "delta_t": float(delta_t)})
        checkpoint_file.save_function(Taverage, name="Average_Temperature", idx=timestep,
                                      timestepping_info={"index": float(timestep),
                                                         "delta_t": float(delta_t)})

    # The state after the last step goes to index ``steps``. It is the observation
    # that the temperature misfit compares against. ``Temperature`` therefore has
    # ``steps + 1`` entries, indices 0 to ``steps``, while ``Velocity`` and
    # ``Average_Temperature`` have ``steps`` entries, indices 0 to ``steps - 1``.
    checkpoint_file.save_function(T, name="Temperature", idx=steps,
                                  timestepping_info={"index": float(steps),
                                                     "delta_t": float(delta_t)})
    checkpoint_file.close()
    log(f"Reference state written to {output}: {steps} steps at resolution {resolution}")


if __name__ == "__main__":
    args = parse_args()
    run_forward(args.steps, args.resolution, args.delta_t, args.output, vtk=args.vtk)
