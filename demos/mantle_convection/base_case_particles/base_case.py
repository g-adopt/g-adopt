from gadopt import *
import numpy as np

from pathlib import Path

# particles submodule isn't yet included in Firedrake
import sys
sys.path.insert(0, "/opt/firedrake/particles")

from particle_time_stepper import ForwardEulerStepper
from particle_crossing_solver import BisectionSolver, BisectionSolverParams
from particle_traj_solver import ParticleTrajectorySolver, ParticleTrajectorySolverParams

def write_particle_vtk(fname, vom, function, coordinates=None):
    if coordinates is None:
        coordinates = vom.coordinates.dat.data_ro
    coordinates = np.pad(coordinates, (0,1))
    print("writing with coordinates:", coordinates)
    dat = function.dat.data_ro
    dat = np.pad(dat, (0,1))
    print("writing with data:", dat)
    num_points = coordinates.shape[0]

    def write_array_descriptor(f, name, arr, offset):
        nbytes = 0
        ncmp = {0: "", 1: "3"}[len(arr.shape[1:])]
        typ = {np.dtype("float64"): "Float64",}[arr.dtype]
        offset += nbytes
        nbytes += (4 + arr.nbytes)
        f.write(f"""<DataArray Name="{name}" type="{typ}" NumberOfComponents="{ncmp}" format="appended" offset="{offset}" />""".encode())
        return nbytes

    def write_array(f, arr):
        np.uint32(arr.nbytes).tofile(f)
        arr.tofile(f)

    with open(fname, "wb") as f:
        offset = 0
        f.write(f"""<?xml version="1.0" ?>
<VTKFile type="UnstructuredGrid" version="2.2" byte_order="LittleEndian" header_type="UInt32">
<UnstructuredGrid>
<Piece NumberOfPoints="{num_points}" NumberOfCells="0">
<Cells>
<DataArray type="Int32" Name="connectivity" format="ascii"></DataArray>
<DataArray type="Int32" Name="offsets" format="ascii"></DataArray>
<DataArray type="UInt8" Name="types" format="ascii"></DataArray>
</Cells>
<Points>
""".encode())
        offset += write_array_descriptor(f, "Coordinates", coordinates, offset)
        f.write(b"</Points>")
        f.write(b"""<PointData Vectors="Particles">""")
        offset += write_array_descriptor(f, "Particles", dat, offset)
        f.write(b"""</PointData>
</Piece>
</UnstructuredGrid>
<AppendedData encoding="raw">
_""")
        write_array(f, coordinates)
        write_array(f, dat)
        f.write(b"""</AppendedData>
</VTKFile>""")


def write_pvd(fname, tmpl, count):
    with open(fname, "w") as f:
        f.write("""<?xml version="1.0" ?>
<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">
<Collection>
""")
        for i in range(count):
            f.write(f"""<DataSet timestep="{i}" file="{tmpl.format(i)}" />\n""")

        f.write("</Collection></VTKFile>")


# We will set up the problem using a bilinear quadrilateral element
# pair (Q2-Q1) for velocity and pressure, with Q2 elements for
# temperature.
#
# We first need a mesh: for simple domains such as the unit square,
# Firedrake provides built-in meshing functions. As such, the
# following code defines the mesh, with 40 quadrilateral elements in x
# and y directions. We also tag boundary IDs.  Boundaries are
# automatically tagged by the built-in meshes supported by
# Firedrake. For the `UnitSquareMesh` being used here, tag 1
# corresponds to the plane $x=0$; 2 to the $x=1$ plane; 3 to the $y=0$ plane;
# and 4 to the $y=1$ plane. We name these `left`, `right`, `bottom` and `top`,
# respectively.
#
# On the mesh, we also denote that our geometry is Cartesian, i.e. gravity points
# in the negative z-direction. This attribute is used by gadopt specifically, not
# Firedrake. By contrast, a non-Cartesian geometry is assumed to have gravity
# pointing in the radially inward direction.

nx, ny = 40, 40  # Number of cells in x and y directions.
#mesh = UnitSquareMesh(nx, ny, quadrilateral=True)  # Square mesh generated via firedrake
mesh = UnitSquareMesh(nx, ny, quadrilateral=False)  # Square mesh generated via firedrake
mesh.cartesian = True
boundary = get_boundary_ids(mesh)

num_particles = 10
r = 0.25
c = np.array([0.5, 0.5])
theta = np.linspace(0, 2*np.pi, num_particles, endpoint=False)

x0 = c[0] + r*np.cos(theta)
y0 = c[1] + r*np.sin(theta)
q0 = np.column_stack([x0, y0])

particle_vom = VertexOnlyMesh(mesh, q0)
P0DG = VectorFunctionSpace(particle_vom, "DG", 0)
print("Initial particle positions: ", particle_vom.coordinates.dat.data_ro)

# We also need function spaces, which is achieved by associating the
# mesh with the relevant finite element: V , W and Q are symbolic
# variables representing function spaces. They also contain the
# function space’s computational implementation, recording the
# association of degrees of freedom with the mesh and pointing to the
# finite element basis.

V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)

# Function spaces can be combined in the natural way to create mixed
# function spaces, combining the velocity and pressure spaces to form
# a function space for the mixed Stokes problem, Z.

Z = MixedFunctionSpace([V, W])  # Mixed function space.

# We also specify functions to hold our solutions: z in the mixed
# function space, noting that a symbolic representation of the two
# parts – velocity and pressure – is obtained with `split`. For later
# visualisation, we rename the subfunctions of z Velocity and Pressure.

z = Function(Z)  # A field over the mixed function space Z.
u, p = split(z)  # Returns symbolic UFL expression for u and p
z.subfunctions[0].rename("Velocity")
z.subfunctions[1].rename("Pressure")

# +
Ra = Constant(1e4)  # Rayleigh number
approximation = BoussinesqApproximation(Ra)

time = 0.0  # Initial time
delta_t = Constant(1e-6)  # Initial time-step
timesteps = 1000  # Maximum number of timesteps
t_adapt = TimestepAdaptor(delta_t, u, V, maximum_timestep=0.1, increase_tolerance=1.5)

steady_state_tolerance = 1e-9  # Used to determine if solution has reached a steady state.
# -

X = SpatialCoordinate(mesh)
T = Function(Q, name="Temperature")
T.interpolate((1.0-X[1]) + (0.05*cos(pi*X[0])*sin(pi*X[1])))

# We can visualise the initial temperature field using Firedrake's
# built-in plotting functionality.

# + tags=["active-ipynb"]
# import matplotlib.pyplot as plt
# fig, axes = plt.subplots()
# collection = tripcolor(T, axes=axes, cmap='coolwarm')
# fig.colorbar(collection);
# -

Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=False)

stokes_bcs = {
    boundary.bottom: {'uy': 0},
    boundary.top: {'uy': 0},
    boundary.left: {'ux': 0},
    boundary.right: {'ux': 0},
}

temp_bcs = {
    boundary.bottom: {'T': 1.0},
    boundary.top: {'T': 0.0},
}
# -

# We next set up our output, in VTK format. To do so, we create the output file
# and specify the output_frequency in timesteps.

output_file = VTKFile("output.pvd")
output_frequency = 10

# Given that this model is isoviscous, we can speed up the simulation by specifying a
# constant Jacobian (preventing uneccesary matrix re-assembly).
# We note that solution of the two variational problems is undertaken by PETSc.

# +
energy_solver = EnergySolver(T, u, approximation, delta_t, ImplicitMidpoint, bcs=temp_bcs)

stokes_solver = StokesSolver(
    z,
    approximation,
    T,
    bcs=stokes_bcs,
    nullspace=Z_nullspace,
    transpose_nullspace=Z_nullspace,
)

# XXX this doesn't re-interpolate u when the VOM moves (or time changes...?)
stepper = ForwardEulerStepper(particle_vom, 0.0, u)
cell_crossing_solver = BisectionSolver()
particle_traj_solver_params = ParticleTrajectorySolverParams(
  bary_tol=1e-9,
  abs_time_tol=1e-9,
  rel_time_tol=0,
  max_iters=50,
  plot=False,
)
particle_traj_solver = ParticleTrajectorySolver(
  stepper,
  cell_crossing_solver,
  particle_traj_solver_params,
)

# set up vtu output directory
Path("particles").mkdir(exist_ok=True)
particle_file_base = "particles/particles_{}.vtu"
particle_file_count = 0

u_at_points = Function(P0DG)
u_at_points.interpolate(z.subfunctions[0])
# -

for timestep in range(0, timesteps):
    # Write output:
    if timestep % output_frequency == 0:
        output_file.write(*z.subfunctions, T)
        write_particle_vtk(particle_file_base.format(particle_file_count), particle_vom, u_at_points)
        particle_file_count += 1

    dt = t_adapt.update_timestep()
    time_start = time
    stepper._dt = dt
    time += dt

    # Solve Stokes sytem:
    stokes_solver.solve()

    # Temperature system:
    energy_solver.solve()

    particle_traj_solver.solve(time_start, time)

    u_at_points.interpolate(z.subfunctions[0])

    # Calculate L2-norm of change in temperature:
    maxchange = sqrt(assemble((T - energy_solver.T_old)**2 * dx))

    # Leave if steady-state has been achieved:
    if maxchange < steady_state_tolerance:
        log("Steady-state achieved -- exiting time-step loop")
        break

# We can visualise the final temperature field using Firedrake's
# built-in plotting functionality.

# + tags=["active-ipynb"]
# fig, axes = plt.subplots()
# collection = tripcolor(T, axes=axes, cmap='coolwarm')
# fig.colorbar(collection);
