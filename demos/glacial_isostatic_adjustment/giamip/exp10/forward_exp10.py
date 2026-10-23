r"""GIAMIP Exp10: a forward run from the inverted bed, with the surface fields.

WHAT THIS SCRIPT DOES
    It runs the coupled self-gravitating sea-level solver of
    `invert_exp10.py` forward, once, from the bed that an inversion found,
    and writes what the run predicts at the surface Re:

    1. Load the state file of an inversion (`common.read_inputs(path,
       last_control=True)`): the meshes, the earth fields, the ice of the
       march steps (2 kyr steps from the start to year 0: steps 0 .. 61
       from 122 ka, 0 .. 40 from 80 ka) and the bed of
       the last ROL iteration (CG2 on the mantle, only the nodes on Re carry
       it). The run must use the number of ranks of that file.
    2. Build the solver exactly as `invert_exp10.py` does
       (`common.build_solver`, the same options and defaults), with
       `SL_init = -bed / D` on the mantle nodes on Re (`SurfaceNodeMap`, an
       exact copy of nodal values) and the ice of step 0 as `I_init`.
    3. March the steps of 2 kyr to year 0 (61 from 122 ka, 40 from 80 ka),
       then one step of 1000 yr to
       year +1000 (3000 CE), the end of the GIAMIP protocol. The input file
       has no ice for +1000 yr; this script reads it from giamip with the
       transfer of the prepare script (`prepare_exp10.SurfaceForcing`).
    4. At year 0, compute the misfit J of the inversion and compare it with
       the last J in the state file.
    5. At every step, step 0 included, write the surface fields to one
       `CheckpointFile` and one row of scalars to a `ParameterLog`.

    Steps 3 and 4 describe the default schedule, `--schedule inversion`.
    `--schedule protocol` replaces the march by the steps of the GIAMIP
    protocol (next section).

    Nothing is taped: `firedrake.adjoint` and `gadopt.inverse` are not
    imported, and the annotation of pyadjoint is paused at the start.

THE TWO SCHEDULES
    `--schedule inversion` (the default) marches the steps of the input
    file: 2 kyr from the start to year 0, then one step of 1000 yr. At year
    0 this is the march of the inversion's tape, so the misfit J must equal
    the last J of the state file.

    `--schedule protocol` marches every model year of the GIAMIP protocol
    (`GiamipExperiment.model_years()`, protocol v4.0): 500 yr steps from the
    start (122 ka or 80 ka) to year -26000, then 100 yr steps to year +1000.
    That is 462 steps from 122 ka (year 0 is step 452) and 378 steps from
    80 ka (year 0 is step 368). The ice of every step is read from giamip
    in the job (`prepare_exp10.SurfaceForcing`, the transfer of the prepare
    script), one step at a time: the input file holds the ice of the 2 kyr
    steps only, and the ice of every protocol year in an input file is too
    large at the finer levels. `SurfaceForcing` is made with
    `step_years = 1`, so `coarse_step_indices` keeps every protocol year;
    the script checks that the years and the forcing steps equal those of
    giamip. Before the march, the ice of every year that the input file also
    holds (the 2 kyr years) is made again here and compared with the input
    file, so the ice on the mesh is the ice of the inversion at those years.

    The bed was inverted with 2 kyr steps. The protocol steps have a smaller
    time-discretisation error (backward Euler, first order in dt), so the
    bed at year 0 is not the observed bed to the 0.3 mm of the inversion.
    The misfit at year 0 against the observed bed measures that difference:
    the log and the summary give J, the RMS misfit on the solver's measure,
    and the nodal largest and RMS difference of the bed (`bed_misfit_year0`
    in the checks). The surface file holds the observed bed and the misfit
    field at year 0 (saved once, without an index).

    The protocol run uses `--gia_rebuild_every 20` unless the option is
    given (decision of 2026-10-20: the steps are small, and every 20 solves
    gave the same J to 7 digits as every 5 at L0). The time step changes
    once, from 500 to 100 yr, at the step after year -26000; the rebuild
    rule rebuilds at the next solve (next section).

THE FINE SCHEDULES (Exp09, Exp10, Exp12; design handout 8.3 and 12.7)
    `--schedule fine` (schedule A, decision S19): 50 yr steps from the start
    to year -26000, then 10 yr steps to year +1000; 1920 + 2700 = 4620
    steps from 122 ka, 1080 + 2700 = 3780 from 80 ka. `--schedule fine_b`
    (schedule B, the fallback of S25): 100 yr steps to year -26000, then
    25 yr steps, and 10 yr steps in the fast melt windows of the ice
    history (`FINE_B_WINDOWS`). The step lists are built in code from these
    rules (`fine_years`), and every protocol year is a year of each list.

    The ice at a year between two protocol years is linear in time between
    the two protocol slices (`LinearIce`, P9); at a protocol year it is the
    protocol slice bit for bit. The run checks this at the first
    `--ice_check_years` protocol years against a fresh read from giamip.

    The surface fields and the row of the parameter log are written at the
    protocol years only (S12), with the protocol position as the index
    (0, 1, 2, ...), so `paraview_exp10.py` reads a contiguous series, as for
    the protocol schedule. In a row, `dt_yr`, `newton` and `outer` are those
    of the last step before the output, and `solve_s` is the solve time of
    all steps since the previous output. Every step goes to the text log
    (`fine step ...`) and to `fine_steps` in the summary.

    The preconditioner is rebuilt every 100 solves (FINE_REBUILD_EVERY), at
    a change of dt, and when a solve needed more than FINE_REBUILD_MAX_OUTER
    outer iterations per Newton iteration, unless the options say otherwise.

SCHEDULE C (`--schedule windows`; Exp09, Exp10, Exp12; S33, S34)
    The protocol steps (500 yr to year -26000, then 100 yr to +1000) plus
    the fast melt windows of the ice history at 25 yr and 10 yr
    (`exp10_common.SCHEDULE_C_WINDOWS`, design handout 8.3): 732 steps from
    122 ka for ICE-6G_D, 762 for GLAC3c profile 1, 792 for profile 2. The
    years come from `exp10_common.schedule_c_years`, which the prepare
    script also uses (`prepare_expXX.py --schedule windows`), so the
    inversion on schedule C and this run march the same years. Output,
    preconditioner rule, checkpoints and restart are those of the fine
    schedules.

    The ice. When the state file holds the schedule-C march (its record
    says `"schedule": "windows"` and its years are the years of this
    schedule up to year 0), the steps up to year 0 take their ice from the
    file, as the inversion does, so this run and the inversion see the same
    ice at every step, bit for bit. Before the march the run checks the ice
    of the file: at each non-protocol year it must equal the linear
    combination of the two protocol slices of the file to 1e-12 of the
    largest ice (`exp10_common.check_linear_ice`; the prepare job checks it
    bit for bit before the write, and the read of the file interpolates
    each field, exact only to round-off), and at every year it must equal the
    ice made here from giamip (`LinearIce`) within the tolerance of the
    transfer test (the transfer of the prepare job and of this run differ
    by about 1e-10 m). The steps after year 0 read the ice from giamip
    (`LinearIce`, at protocol years only, so a copy of the slice). With a
    state file of another march (for example the 2 kyr inversion), every
    step takes the ice of `LinearIce`, as in the fine schedules.

THE RESTART
    With checkpoints (the fine schedules by default, or `--checkpoint_every`
    or `--wall_limit_hours`) the run copies the state file into the output
    directory (`forward_state.h5`) and loads its meshes and fields from the
    copy. A checkpoint (`Checkpoints`) is appended to that copy: the
    displacement, the internal variables and the reference of `delta_g`,
    copied from the mantle onto parent functions of the same element
    (exact nodal copies), the potential, and the `Real` unknowns, the step,
    the year, the output index and the leg as timestepping values. A
    restart (`--restart`, same output directory, same rank count) loads the
    meshes from the copy, so the checkpoint fields load onto the same mesh,
    restores the solution and `solution_old`, reads the two bracketing ice
    slices again, rebuilds the preconditioner at its first solve (the
    iteration counts change, the solution does not), and appends to the
    surface file and the parameter log. Periodic checkpoints are written at
    the first protocol year at or after every `--checkpoint_every` steps.
    `--wall_limit_hours` writes a checkpoint and stops when the remaining
    time would not cover three of the slowest recent steps and the save of
    a checkpoint; the run then leaves the file RESTART_NEEDED, and
    `run_forward_fine.slurm` submits the next leg.

THE INITIAL SEA LEVEL
    The inversion wrote the control of each ROL iteration into the state
    file as `extend(b)`: the surface bed `b` copied onto the mantle CG2
    nodes on Re, zero at every other node (`exp10_common.append_control`).
    `read_inputs` loads that field onto the mantle with `Function.assign`.
    Here `restrict` copies the nodes on Re back onto the surface (`b` bit
    for bit), and `extend(b, scale=-1/D)` writes `SL_init`. That is the
    operation of the first block of the inversion tape
    (`SurfaceExtendBlock`), applied to the same `b`, so `SL_init` is the
    field of the replay at the last iterate bit for bit. The script checks
    that `extend(restrict(bed))` equals the stored field and that
    `restrict(extend(b))` equals `b`; both differences must be zero. With
    `--controls` (the default is `controls.h5` beside the state file) it
    also compares the min, max and integral of `b` with the last control of
    `controls.h5`, which the inversion wrote from the ROL vector itself.

THE TIME STEP
    `dt` is the time step of the solver, a `Function` on the Real space of
    the mantle (`exp10_common.time_step`), in units of t_bar. It changes once:
    from 2000 yr to 1000 yr before the last step (inversion schedule), or
    from 500 yr to 100 yr after year -26000 (protocol schedule). The script
    assigns `dt` only when the step length changes. The preconditioner of the
    solver has one rebuild rule (`gadopt.preconditioners._RebuildRule`). The
    moving coastline makes the Jacobian depend on the state, so the solver
    publishes `operator_version = None` and the rule takes the lagged
    branch (`gia_rebuild = lagged`, every 5 solves by default, every 20 in
    the protocol schedule). On that branch `_lag_expired` compares
    `appctx["dt"]`, which is this `Function`, with the value of the last
    build, and rebuilds at the first solve after a change, independently of
    the count of solves. The first step after a change of dt therefore
    starts from a new preconditioner. The log shows the build counts of the
    preconditioner blocks after every step (`rebuild_counts`).

THE MISFIT CHECK
    `J = 1/2 int_Re ((SL - SL_obs) D)^2 dS` on the sea-level measure of the
    solver, the form of `invert_exp10.BedInversion.misfit`. The value at
    year 0 must equal the last J of the state file (`J` of the last
    `bed_start_parent_cg2_m` entry) up to the reproducibility of the
    iterative solves. The J of the inversion comes from a replay of the tape
    whose preconditioners carried the build history of the earlier replays
    and adjoints; this run builds them from new. J is small at the end of
    an inversion (RMS misfit 0.3 mm against a sea-level change of tens of
    metres), so the relative difference of J is sensitive to the solver
    tolerance.

THE SURFACE FIELDS (`forward_exp10_surface.h5`)
    One `CheckpointFile` on the surface submesh `giamip_surface`
    (`common.surface_mesh`, P1 geometry). Every entry has `idx` = the output
    index (the step in the inversion and protocol schedules, the protocol
    position in the fine schedules) and
    the timestepping values `step`, `year`, `shift_m`, `m1`, `m2`, `m3`.
    All fields are in SI units. Saved once: `coordinates_m`, the positions
    of the CG2 nodes of the curved mantle at Re (the surface mesh itself is
    flat; `paraview_exp10.py` builds the curved mesh from this field). Also
    saved once, when the march reaches year 0: `bed_observed_m`, the
    observed present-day bed `-SL_obs D` (CG2, m), and
    `bed_misfit_year0_m`, the bed of year 0 minus the observed bed (CG2,
    m; positive where the run ends above the observed bed).

    CG2 nodal fields. The mantle field is made at the CG2 nodes of the
    mantle (the curved positions), and `SurfaceNodeMap.restrict` copies the
    nodes on Re onto the surface. The displacement components are copied
    from the vector CG2 data without any evaluation. The other fields are
    interpolated into mantle CG2 first: the value at each node is the
    expression at that node, to round-off.

      displacement_m        u, vector, m
      delta_bed_m           u . rhat, the radial displacement, m (GIAMIP
                            `delta_bed`, the bedrock elevation change in the
                            centre-of-mass frame)
      displacement_east_m   u . e_east, m (GIAMIP `delta_bed_east`)
      displacement_north_m  u . e_north, m (GIAMIP `delta_bed_north`)
      geoid_m               N = (psi + psi_rot) / g_s, m, with the
                            rotational potential, as the solver's sea level
                            uses it
      sea_surface_m         N + Shift, m
      delta_g_m             (N + Shift) - (N + Shift at step 0), m
      sea_level_m           SL = SL_init + N - u . rhat + Shift, m
      bed_m                 -SL, m (the bed elevation against the sea
                            surface; at step 0 the bed of the start)

    The solver's sea level uses `u . n` with the facet normal of the curved
    mantle facet; the nodal field uses the radial unit vector at the node.
    On the P2 facets the two directions differ by O(h^2), which with
    horizontal displacements of tens of metres is below a millimetre. The
    log gives the RMS difference of `N + Shift` between the nodal field and
    the solver's expression on the sea-level measure, every step.

    The ice thickness `ice_thickness_m` (CG1, m). The CG1 ice of the mantle
    is interpolated into mantle CG2 (exact: CG1 is contained in CG2 on every
    cell), restricted to the surface, and interpolated into surface CG1
    (the vertex values). Each step costs one CG2 field of work memory.

    DG0 facet fractions. The masks of the solver contain the facet area
    (`gadopt.sea_level_masks.mask_steepness`), so they have a value only
    inside a facet integral. Each surface cell is a mantle facet on Re, and
    the field holds the mean of the mask over that facet:

        value(f) = int_f mask dS / int_f 1 dS

    on the mantle facet measure, with the quadrature degree of the solver,
    assembled against a DG0 test function on the surface (a mixed-dimension
    form, mantle `ds` intersected with surface `dx`). The integral of the
    field times the facet area is then the integral of the mask that the
    solver sees. The sea level in the masks is built on the mantle from the
    potential copied onto the mantle, which on Re equals the parent's trace.

      ocean_function            C = H_k(SL), 1 where the sea level is
                                positive (ocean, also under grounded ice)
      ocean_without_grounded_ice  B, 1 over open water and floating ice
                                (the ocean of the sea-level load; GIAMIP
                                `ocean_area_fraction`)
      grounded_ice_fraction     (1 - B) * [I > 1 mm], the grounded ice
      land_ice_fraction         [I > 1 mm], grounded and floating ice
                                (GIAMIP `land_ice_area_fraction`: "covered
                                by grounded and floating land ice")

THE SCALARS (`params_forward_exp10.log`)
    One row per output, step 0 included, from UFL `assemble` on the solver's
    sea-level measure `dS` (curved mantle facets on Re, the solver's
    quadrature degree). `I` is the ice thickness, `SL` the sea level of the
    solver, `B` and `C` the solver's masks (`gadopt.sea_level_masks`),
    `rho_i` and `rho_w` the densities of the experiment (917 and 1025
    kg/m^3).

      step, year, dt_yr      the step, its model year, its length (0 at
                             step 0)
      newton, outer          Newton and outer FGMRES iterations of the solve
      shift_m                Shift, the uniform sea-level change, m
      m1, m2, m3             the rotation vector perturbation
      mean_delta_g_m         int delta_g B dS / int B dS, m: GIAMIP "spatial
                             mean of delta_g over the ocean area", over the
                             ocean of the current step (B, open water and
                             floating ice), with delta_g = (N + Shift) -
                             (N + Shift)_0 and N + Shift = SL - SL_init + u.n
      grd_ice_mass_kg        rho_i int (1 - B) I dS, kg (as `state_row` of
                             `tests/giamip/exp01.py`): GIAMIP "grounded ice
                             volume times ice density"
      total_ice_mass_kg      rho_i int I dS, kg: GIAMIP "grounded and
                             floating ice volume times ice density"
      ocean_area_m2          int B dS, m^2: GIAMIP "ocean area excluding
                             marine regions covered by grounded ice". `int
                             B` and not `int B C`, see
                             `grounded_ice_function`
      ocean_area_grdice_m2   int C dS, m^2: GIAMIP "ocean area including
                             marine regions covered by grounded ice" (the
                             sea level is positive there)
      maf_kg                 rho_i int max(0, I - (rho_w / rho_i) max(SL,
                             0)) dS, kg (as `state_row`): the ice mass above
                             flotation
      solve_s                the wall time of the solve, s

    Areas are in units of D^2 and volumes in D^3 on the mesh; the factors
    D^2 and D^3 convert them.

RUN
    srun python3 forward_exp10.py --state <dir>/state.h5 --output_path <dir>
    [--schedule protocol] [--steps 3] [--plus_step always]

    `--steps n` stops the march after n steps (of 2 kyr, or of the
    protocol). In the inversion schedule the step to +1000 yr follows the
    march when it reaches year 0 (`--plus_step auto`); `--plus_step always`
    runs it after a shorter march too, which tests the giamip read and the
    change of dt in a short job (the step then has no physical meaning).
    The protocol schedule ignores `--plus_step`: year +1000 is its last
    protocol step.

    At L0 on 3 nodes the protocol run takes about 3-7 h (462 steps of
    20-45 s), well inside the 24 h limit of the work partition; it runs
    without checkpoints unless asked. The fine schedules need the restart
    (THE RESTART).

OUTPUT (in --output_path)
    forward_state.h5             with checkpoints: the run's copy of the
                                 state file, with the checkpoints
    RESTART_NEEDED               after a stop at the wall limit
    forward_exp10.log            the log (rank 0 also prints it)
    params_forward_exp10.log     the scalars, one row per output
    forward_exp10_surface.h5     the surface fields of every output
    summary_forward.json         the settings, the checks and the rows
"""

import argparse
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# `gadopt` imports `firedrake.pyplot`, which imports matplotlib's font
# manager. Without a font cache every rank scans the fonts; the job
# environment names a cache built once on a login node.
if os.environ.get("EXP01_MPLCONFIGDIR"):
    os.environ["MPLCONFIGDIR"] = os.environ["EXP01_MPLCONFIGDIR"]
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

# The shared module imports gadopt before firedrake.
import exp10_common as common  # noqa: E402
from gadopt import (COMM_WORLD, CheckpointFile, Constant,  # noqa: E402
                    FacetNormal, Function, FunctionSpace, Measure,
                    SpatialCoordinate, TestFunction,
                    VectorFunctionSpace, as_vector, assemble, conditional,
                    dot, ds, gt, max_value, sqrt)
from gadopt.momentum_equation import rotational_potential  # noqa: E402
from gadopt.sea_level_masks import (grounded_ice_function,  # noqa: E402
                                    ocean_function)
from mpi4py import MPI  # noqa: E402
from pyadjoint.tape import annotate_tape, pause_annotation  # noqa: E402

# The prepare script holds the transfer from the GIAMIP grid to the mesh;
# the ice of +1000 yr is made with it, so the rule is the rule of the input
# file. Importing it imports giamip.
import prepare_exp10 as prepare  # noqa: E402
from giamip.mip.inputs import GiamipExperiment  # noqa: E402

# No tape in this run. The annotation of pyadjoint is off unless something
# switched it on; this makes sure of it.
pause_annotation()

D_SCALE = common.D_SCALE

#: The file names of the outputs.
SURFACE_FILE = "forward_exp10_surface.h5"
PARAMS_FILE = "params_forward_exp10.log"
LOG_FILE = "forward_exp10.log"
SUMMARY_FILE = "summary_forward.json"

#: The ice thickness above which a point counts as covered by ice in the
#: land-ice and grounded-ice fractions, m. The CG1 ice of the input file is
#: zero outside the ice sheets up to round-off (about 1e-19 in units of D),
#: and the bilinear transfer of the grid gives thin ice next to a margin;
#: 1 mm separates the two.
ICE_COVER_MIN_M = 1.0e-3

#: The rebuild interval of the preconditioner (solves) of the protocol
#: schedule when `--gia_rebuild_every` is not given (Sia, 2026-10-20). The
#: protocol steps are 4 to 20 times shorter than the 2 kyr steps, so the
#: operator changes less per solve; at L0 every 20 solves gave the J of
#: every 5 to 7 digits and saved 12-14 % of the time (handout L1-L2,
#: section 13).
PROTOCOL_REBUILD_EVERY = 20

#: The columns of `params_forward_exp10.log`, in order (see the module
#: docstring for each definition).
PARAM_COLUMNS = ("step", "year", "dt_yr", "newton", "outer", "shift_m", "m1",
                 "m2", "m3", "mean_delta_g_m", "grd_ice_mass_kg",
                 "total_ice_mass_kg", "ocean_area_m2", "ocean_area_grdice_m2",
                 "maf_kg", "solve_s")


# --------------------------------------------------------------------------
# Helpers that move fields to the surface
# --------------------------------------------------------------------------


def copy_vector_component(source, component, out):
    """Copy one component of a vector CG2 field into a scalar CG2 field.

    A vector Lagrange space and the scalar space of the same element on the
    same mesh share the node numbering; the vector data holds one row of
    components per node. The copy is therefore a copy of nodal values, with
    no evaluation of basis functions. Only the owned rows are written; the
    halo of `out` is marked out of date and is updated when it is read.

    Args:
      source: a Function in a vector CG2 space.
      component: the index of the component.
      out: a Function in the scalar CG2 space of the same mesh.

    Returns:
      `out`.
    """
    out.dat.data_wo[:] = source.dat.data_ro[:, component]
    return out


def set_vector_component(out, component, source, scale=1.0):
    """Write a scalar CG2 field, times `scale`, into one component of a vector.

    The inverse of `copy_vector_component`, on owned rows.

    Args:
      out: a Function in a vector CG2 space.
      component: the index of the component.
      source: a Function in the scalar CG2 space of the same mesh.
      scale: a float factor (for example D to go to metres).
    """
    # Read-write access: the other components keep their values.
    out.dat.data[:, component] = scale * source.dat.data_ro


def vec_max_abs(f):
    """The largest absolute owned value of a Function, over all ranks."""
    data = f.dat.data_ro
    local = float(np.abs(data).max()) if data.size else 0.0
    return COMM_WORLD.allreduce(local, op=MPI.MAX)


def max_abs_diff(a, b):
    """The largest absolute difference of the owned values of two Functions.

    Both must be in the same space. A plain NumPy difference on the owned
    data, reduced over the ranks, so that an exact copy gives exactly 0.
    """
    da, db = a.dat.data_ro, b.dat.data_ro
    local = float(np.abs(da - db).max()) if da.size else 0.0
    return COMM_WORLD.allreduce(local, op=MPI.MAX)


def facet_frame(mesh):
    """The radial, east and north unit vectors as UFL on `mesh`.

    rhat    = x / |x|
    e_east  = (-y, x, 0) / rho
    e_north = (-x z, -y z, rho^2) / (|x| rho)

    with rho = sqrt(x^2 + y^2) the distance from the rotation axis. East and
    north are undefined on the axis; `rho` is bounded below by 1e-12 (in
    units of D) there, so a node exactly on a pole gets a finite value
    without meaning.

    Returns:
      `(rhat, e_east, e_north)`.
    """
    X = SpatialCoordinate(mesh)
    r = sqrt(dot(X, X))
    rho = max_value(sqrt(X[0] ** 2 + X[1] ** 2), 1e-12)
    rhat = X / r
    e_east = as_vector([-X[1] / rho, X[0] / rho, 0.0])
    e_north = as_vector([-X[0] * X[2] / (r * rho), -X[1] * X[2] / (r * rho),
                         rho / r])
    return rhat, e_east, e_north


def last_state_J(path):
    """The J of the last control entry of a state file, read with h5py.

    `exp10_common.append_control` saves each iterate with the timestepping
    values `J` and `rms_misfit_m`. `CheckpointFile` stores them as the
    attributes `firedrake_timestepping_history_J` and
    `firedrake_timestepping_history_rms_misfit_m` on the timestepping group
    of the function, one array entry per save, in the order of the saves,
    which is the order of the iterations. The last entry belongs to the
    control that `read_inputs(..., last_control=True)` loads (the largest
    index). Read on rank 0 and broadcast; no mesh is loaded.

    Args:
      path: the state file of an inversion.

    Returns:
      `(J, rms_misfit_m)` of the last iterate, or `(None, None)` if the file
      holds no such attribute.
    """
    found = (None, None)
    if COMM_WORLD.rank == 0:
        import h5py
        hits = []
        key_J = "firedrake_timestepping_history_J"
        key_rms = "firedrake_timestepping_history_rms_misfit_m"

        def visit(name, obj):
            # The group of the control function carries its name in the path.
            if common.STATE_CONTROL_NAME in name and key_J in obj.attrs:
                hits.append((name, np.asarray(obj.attrs[key_J]),
                             np.asarray(obj.attrs.get(key_rms, [np.nan]))))

        with h5py.File(path, "r") as handle:
            handle.visititems(visit)
        if hits:
            _, J, rms = hits[0]
            found = (float(J[-1]), float(rms[-1]))
    return COMM_WORLD.bcast(found, root=0)


def rebuild_counts(solver):
    """The build counts of the cached preconditioner pieces, by class.

    Walks the PC tree of the outer solve: a Python PC gives its context, the
    context may hold an inner PC (`.pc`), and a field split gives the PCs of
    its sub-solves. Every context that has `build_count`, `assembly_count`
    or `column_builds` is reported. The type of every PC is checked before
    `getPythonContext`: asking a PC that is not of type python for its
    context reads a pointer that is not a Python object.

    Diagnostics only; any failure gives an empty dict.

    Args:
      solver: the `SelfGravitatingGIASolver`.

    Returns:
      `{class name: count}`.
    """
    counts = {}

    def walk(pc, depth):
        if pc is None or depth > 6:
            return
        kind = pc.getType()
        if kind == "python":
            ctx = pc.getPythonContext()
            for attribute in ("build_count", "assembly_count",
                              "column_builds"):
                value = getattr(ctx, attribute, None)
                if isinstance(value, (int, np.integer)):
                    counts[f"{type(ctx).__name__}.{attribute}"] = int(value)
            inner = getattr(ctx, "pc", None)
            if inner is not None and inner is not pc:
                walk(inner, depth + 1)
        elif kind == "fieldsplit":
            try:
                subs = pc.getFieldSplitSubKSP()
            except Exception:  # noqa: BLE001  (not set up)
                subs = []
            for ksp in subs:
                walk(ksp.getPC(), depth + 1)
        elif kind == "ksp":
            walk(pc.getKSP().getPC(), depth + 1)

    try:
        walk(solver.solver.snes.ksp.pc, 0)
    except Exception:  # noqa: BLE001  (diagnostics only)
        return {}
    return counts


# --------------------------------------------------------------------------
# The surface output
# --------------------------------------------------------------------------


class SurfaceOutput:
    """The surface fields of every step, and the scalars, of the forward run.

    Built once after the solver. Holds the mantle work fields, the surface
    fields of the file and the UFL forms of the scalars, so that each step
    only interpolates, copies and assembles. See the module docstring for
    every field and every scalar.

    Args:
      solver: the `SelfGravitatingGIASolver`.
      layout: its `GIASpaceLayout`.
      mantle: the mantle mesh.
      surface: the surface submesh (`common.surface_mesh`).
      node_map: the `common.SurfaceNodeMap` between CG2 on the surface and
        CG2 on the mantle.
      forcing: `{"SL_init": .., "I": .., "SL_obs": ..}`, the Functions of
        the run on the mantle.
      scales: the `common.Scales`.
      path: the surface file; it is created here with the mesh and the
        coordinates.
      new_file: False for a restarted run, which appends to the surface
        file of the earlier legs (the mesh and the coordinates are there).
    """

    def __init__(self, solver, layout, mantle, surface, node_map, forcing,
                 scales, path, new_file=True):
        self.solver, self.layout = solver, layout
        self.mantle, self.surface = mantle, surface
        self.node_map = node_map
        self.scales = scales
        self.path = path
        self.SL_init = forcing["SL_init"]
        self.I = forcing["I"]
        self.SL_obs = forcing["SL_obs"]
        p = solver.sea_level_parameters
        self.params = p
        # The densities in kg/m^3 and the scale factors of area and volume.
        self.rho_i_kg = float(p.rho_i) * scales.rho_bar
        self.rho_w_kg = float(p.rho_w) * scales.rho_bar
        D2, D3 = D_SCALE ** 2, D_SCALE ** 3
        self.D2, self.D3 = D2, D3

        # ---- spaces -----------------------------------------------------
        V_m2 = node_map.V_mantle
        V_m2v = solver.displacement.function_space()
        if (V_m2v.ufl_element().degree() != 2
                or V_m2.ufl_element().degree() != 2):
            raise RuntimeError("the surface output assumes CG2 on the mantle")
        self.V_s2 = node_map.V_surface
        self.V_s2v = VectorFunctionSpace(surface, "CG", 2)
        self.V_s1 = FunctionSpace(surface, "CG", 1)
        self.V_s0 = FunctionSpace(surface, "DG", 0)

        # ---- mantle work fields ------------------------------------------
        # The potential copied from the parent onto the mantle. Each mantle
        # node is a parent node, and `assign` across the submesh map copies
        # nodal values.
        self.psi_m = Function(V_m2, name="potential_on_mantle")
        # The rotation scalars and Shift as Constants, set at every step from
        # the solved `Real` fields (replicated on every rank).
        self.m_const = [Constant(0.0), Constant(0.0), Constant(0.0)]
        self.shift_const = Constant(0.0)
        # One scalar CG2 work field on the mantle, reused for every field.
        self.work_m = Function(V_m2, name="mantle_work")
        # The geoid N on the mantle CG2 nodes (non-dimensional), kept for
        # the sea-surface check.
        self.N_m = Function(V_m2, name="geoid_mantle")
        # N + Shift of step 0 on the mantle, the reference of delta_g.
        self.S0_m = Function(V_m2, name="sea_surface_step0_mantle")

        g_s = float(p.g_surface)
        self.g_s = g_s
        # psi_rot = -Omega^2 [m1 x z + m2 y z - m3 (x^2 + y^2)], the negated
        # centrifugal perturbation, in units of g_bar D: the polynomial of
        # the solver (`rotational_potential`), with the solver's Omega_sq.
        psi_rot_m = rotational_potential(self.m_const, mantle,
                                         Omega_sq=solver.Omega_sq)
        # N = (psi + psi_rot) / g_s in units of D.
        self.N_expr = (self.psi_m + psi_rot_m) / g_s
        rhat, e_east, e_north = facet_frame(mantle)
        u = solver.displacement
        self.ur_expr = dot(u, rhat)
        self.ue_expr = dot(u, e_east)
        self.un_expr = dot(u, e_north)

        # ---- surface fields of the file -----------------------------------
        S = self.V_s2
        self.s_comp = Function(S, name="surface_work")
        self.s_u = Function(self.V_s2v, name="displacement_m")
        self.s_ur = Function(S, name="delta_bed_m")
        self.s_ue = Function(S, name="displacement_east_m")
        self.s_un = Function(S, name="displacement_north_m")
        self.s_N = Function(S, name="geoid_m")
        self.s_S = Function(S, name="sea_surface_m")
        self.s_S0 = Function(S, name="sea_surface_step0_m")
        self.s_dg = Function(S, name="delta_g_m")
        self.s_SL = Function(S, name="sea_level_m")
        self.s_bed = Function(S, name="bed_m")
        self.s_SL_init = Function(S, name="sea_level_init_nondim")
        self.s_ice = Function(self.V_s1, name="ice_thickness_m")
        self.s_C = Function(self.V_s0, name="ocean_function")
        self.s_B = Function(self.V_s0, name="ocean_without_grounded_ice")
        self.s_grd = Function(self.V_s0, name="grounded_ice_fraction")
        self.s_land = Function(self.V_s0, name="land_ice_fraction")
        # The initial sea level on the surface (non-dimensional), an exact
        # copy of the nodes on Re.
        node_map.restrict(self.SL_init, out=self.s_SL_init)
        self.fields = (self.s_u, self.s_ur, self.s_ue, self.s_un, self.s_N,
                       self.s_S, self.s_dg, self.s_SL, self.s_bed, self.s_ice,
                       self.s_C, self.s_B, self.s_grd, self.s_land)

        # ---- DG0 facet means of the masks ----------------------------------
        # The sea level of the solver, written on the mantle only: the
        # potential copied onto the mantle, the rotation and Shift as
        # Constants, `u . n` with the facet normal of the mantle. On Re this
        # is the solver's `sea_level()` (which reads the parent's `avg(psi)`,
        # equal to the trace of the copy).
        n_m = FacetNormal(mantle)
        SL_m = (self.SL_init + (self.N_expr - p.N_init)
                - (dot(u, n_m) - p.ur_init) + self.shift_const)
        # The mask steepness of the solver (facet area, frozen slope).
        k = solver._sea_level_steepness()
        C_m = ocean_function(SL_m, k)
        B_m = grounded_ice_function(self.I, SL_m, k, p.rho_w, p.rho_i)
        cover = conditional(gt(self.I, ICE_COVER_MIN_M / D_SCALE), 1.0, 0.0)
        # The mantle facet measure on Re, intersected with the cells of the
        # surface submesh, at the solver's quadrature degree.
        quad = solver.form.quad_degree
        # The construction of `sea_level_measure` (the `ds` object called
        # with keywords), with the surface cells in place of the parent's
        # interior facets.
        ds_sub = ds(domain=mantle, degree=quad,
                    intersect_measures=(Measure("dx", domain=surface),))
        self.ds_sub = ds_sub(common.SURF_RE)
        v = TestFunction(self.V_s0)
        self.facet_area = assemble(v * self.ds_sub)
        self.mask_forms = {self.s_C: C_m * v * self.ds_sub,
                           self.s_B: B_m * v * self.ds_sub,
                           self.s_grd: (1 - B_m) * cover * v * self.ds_sub,
                           self.s_land: cover * v * self.ds_sub}

        # ---- the scalars on the solver's measure ---------------------------
        dss = solver.sea_level_measure()(common.SURF_RE)
        self.dss = dss
        SL = solver.sea_level()
        k_s = solver._sea_level_steepness()
        C = ocean_function(SL, k_s)
        B = grounded_ice_function(self.I, SL, k_s, p.rho_w, p.rho_i)
        u_split = solver.solution_split[layout.displacement]
        # N + Shift of the solver: SL - SL_init + (u . n - ur_init) + N_init.
        sea_surface = (SL - self.SL_init + (dot(u_split, n_m) - p.ur_init)
                       + p.N_init)
        self.forms = {
            "area": Constant(1.0) * dss,
            "ocean_B": B * dss,
            "ocean_C": C * dss,
            "dg_B": (sea_surface - self.S0_m) * B * dss,
            "grd_ice": (1 - B) * self.I * dss,
            "total_ice": self.I * dss,
            "maf": max_value(self.I - (p.rho_w / p.rho_i) * max_value(SL, 0.0),
                             0.0) * dss,
            # The RMS difference of N + Shift between the solver's expression
            # and the nodal CG2 field (a check of the surface field).
            "S_check": (sea_surface - (self.N_m + self.shift_const)) ** 2 * dss,
            # The misfit of `invert_exp10.BedInversion.misfit`.
            "misfit": Constant(0.5 * D_SCALE ** 2)
                      * (SL - self.SL_obs) ** 2 * dss,
        }
        self.area = float(assemble(self.forms["area"]))

        # ---- the file: mesh and coordinates --------------------------------
        coordinates = Function(self.V_s2v, name="coordinates_m")
        X = SpatialCoordinate(mantle)
        for i in range(3):
            # The curved positions of the mantle CG2 nodes on Re.
            self.work_m.interpolate(X[i])
            node_map.restrict(self.work_m, out=self.s_comp)
            set_vector_component(coordinates, i, self.s_comp, D_SCALE)
        if new_file:
            with CheckpointFile(path, "w") as chk:
                chk.save_mesh(surface)
                chk.save_function(coordinates)

    def _to_surface(self, expr, out, scale):
        """`out = scale * restrict(interpolate(expr, mantle CG2))`.

        The expression is evaluated at the CG2 nodes of the mantle, the nodes
        on Re are copied onto the surface, and the result is scaled (to SI).
        """
        self.work_m.interpolate(expr)
        self.node_map.restrict(self.work_m, out=out)
        if scale != 1.0:
            out.dat.data[:] *= scale
        return out

    def update_constants(self):
        """Set the rotation and Shift Constants from the solved `Real` fields.

        Returns:
          `(shift, rotation)`: Shift (non-dimensional) and the dict of the
          rotation scalars.
        """
        rotation = self.solver.rotation_values()
        for i, name in enumerate(("m1", "m2", "m3")):
            self.m_const[i].assign(rotation.get(name, 0.0))
        shift = float(self.solver.solution.subfunctions[self.layout.sea_level])
        self.shift_const.assign(shift)
        return shift, rotation

    def scalars(self):
        """The scalars of the current state (collective).

        Returns:
          A dict with the scalar columns of the module docstring (without
          step, year, dt_yr, newton, outer, solve_s), plus `misfit_J`,
          `rms_misfit_m` and `sea_surface_check_rms_m`.
        """
        a = {name: float(assemble(form)) for name, form in self.forms.items()}
        D = D_SCALE
        return {
            "mean_delta_g_m": D * a["dg_B"] / a["ocean_B"] if a["ocean_B"] else 0.0,
            "grd_ice_mass_kg": self.rho_i_kg * a["grd_ice"] * self.D3,
            "total_ice_mass_kg": self.rho_i_kg * a["total_ice"] * self.D3,
            "ocean_area_m2": a["ocean_B"] * self.D2,
            "ocean_area_grdice_m2": a["ocean_C"] * self.D2,
            "maf_kg": self.rho_i_kg * a["maf"] * self.D3,
            "misfit_J": a["misfit"],
            "rms_misfit_m": float(np.sqrt(2.0 * a["misfit"] / self.area)),
            "sea_surface_check_rms_m": D * float(np.sqrt(a["S_check"]
                                                         / self.area)),
            "ocean_C_nondim": a["ocean_C"],
        }

    def write(self, step, year, first=False):
        """Make the surface fields of the current state and append them.

        Collective. At the first call (`first`), N + Shift of this state
        becomes the reference of `delta_g`.

        Args:
          step: the step index (the `idx` of the entries).
          year: the model year.
          first: whether this is the reference state.

        Returns:
          `(info, extra)`: the timestepping values of the entries, and the
          sums of the facet means, for the consistency check against the
          scalars.
        """
        D = D_SCALE
        shift, rotation = self.update_constants()
        # The potential on the mantle, the geoid at the mantle nodes.
        self.psi_m.assign(self.solver.potential)
        self.N_m.interpolate(self.N_expr)
        if first:
            # N + Shift of the reference state, on the mantle and on Re.
            self.S0_m.assign(self.N_m + shift)
        # Displacement: three components, copied without evaluation.
        u = self.solver.displacement
        for i in range(3):
            copy_vector_component(u, i, self.work_m)
            self.node_map.restrict(self.work_m, out=self.s_comp)
            set_vector_component(self.s_u, i, self.s_comp, D)
        # Radial, east and north displacement at the nodes, m.
        self._to_surface(self.ur_expr, self.s_ur, D)
        self._to_surface(self.ue_expr, self.s_ue, D)
        self._to_surface(self.un_expr, self.s_un, D)
        # The geoid, m.
        self.node_map.restrict(self.N_m, out=self.s_N)
        self.s_N.dat.data[:] *= D
        # Sea surface N + Shift, and its change from the reference, m.
        self.s_S.dat.data_wo[:] = self.s_N.dat.data_ro + shift * D
        if first:
            self.s_S0.dat.data_wo[:] = self.s_S.dat.data_ro
        self.s_dg.dat.data_wo[:] = self.s_S.dat.data_ro - self.s_S0.dat.data_ro
        # Sea level SL = SL_init + N - u_r + Shift and bed = -SL, m.
        self.s_SL.dat.data_wo[:] = (D * self.s_SL_init.dat.data_ro
                                    + self.s_S.dat.data_ro
                                    - self.s_ur.dat.data_ro)
        self.s_bed.dat.data_wo[:] = -self.s_SL.dat.data_ro
        # Ice: mantle CG1 -> mantle CG2 (exact) -> surface CG2 -> surface CG1
        # (the vertex values), m.
        self.work_m.interpolate(self.I)
        self.node_map.restrict(self.work_m, out=self.s_comp)
        self.s_ice.interpolate(self.s_comp)
        self.s_ice.dat.data[:] *= D
        # The facet means of the masks.
        area = self.facet_area.dat.data_ro
        sums = {}
        for target, form in self.mask_forms.items():
            integral = assemble(form)
            values = integral.dat.data_ro
            target.dat.data_wo[:] = values / area
            sums[target.name()] = COMM_WORLD.allreduce(
                float(values.sum()), op=MPI.SUM)

        info = {"step": float(step), "year": float(year),
                "shift_m": shift * D,
                "m1": float(rotation.get("m1", 0.0)),
                "m2": float(rotation.get("m2", 0.0)),
                "m3": float(rotation.get("m3", 0.0))}
        with CheckpointFile(self.path, "a") as chk:
            for f in self.fields:
                chk.save_function(f, idx=int(step), timestepping_info=info)
        return info, sums

    def set_reference_from_mantle(self):
        """`s_S0` from `S0_m`: the reference of `delta_g` on the surface, m.

        A restarted run does not write step 0, where `write` sets `s_S0`;
        the checkpoint restores `S0_m` on the mantle, and this copies its
        nodes on Re onto the surface. Step 0 builds `s_S0` as
        restrict(N) D + Shift D and this as restrict(N + Shift) D, which
        agree to round-off.
        """
        self.node_map.restrict(self.S0_m, out=self.s_S0)
        self.s_S0.dat.data[:] *= D_SCALE

    def write_info(self, step, year):
        """The timestepping values of `write` for the current state, without a save.

        A restarted run uses it for an output that the surface file already
        holds. The reference of `delta_g` (`S0_m`) comes from the checkpoint,
        and the geoid field `N_m` that the scalars read is made here as in
        `write`.
        """
        shift, rotation = self.update_constants()
        self.psi_m.assign(self.solver.potential)
        self.N_m.interpolate(self.N_expr)
        return {"step": float(step), "year": float(year),
                "shift_m": shift * D_SCALE,
                "m1": float(rotation.get("m1", 0.0)),
                "m2": float(rotation.get("m2", 0.0)),
                "m3": float(rotation.get("m3", 0.0))}

    def bed_misfit(self):
        """The nodal misfit of the current bed against the observed bed.

        Call it right after `write` of the step at year 0: it reads the
        surface field `bed_m` of that step. The observed bed on the surface
        is `-SL_obs D`, with `SL_obs` copied from the mantle CG2 nodes on Re
        (`restrict`, exact). The misfit field is `bed_m - bed_observed_m`,
        positive where the run ends above the observed bed. Both fields are
        saved once (no index) to the surface file.

        The RMS here is the plain mean over the owned surface CG2 nodes, not
        an area integral: the nodes are not equally spaced, so it differs
        from `rms_misfit_m` of `scalars` (the integral on the solver's
        measure), which is the number to compare with the inversion. The
        largest absolute value is the number the integral cannot show.

        Returns:
          A dict with `max_abs_m`, `rms_nodal_m`, `min_m`, `max_m` and the
          number of nodes.
        """
        D = D_SCALE
        observed = Function(self.V_s2, name="bed_observed_m")
        self.node_map.restrict(self.SL_obs, out=observed)
        observed.dat.data[:] *= -D
        misfit = Function(self.V_s2, name="bed_misfit_year0_m")
        misfit.dat.data_wo[:] = (self.s_bed.dat.data_ro
                                 - observed.dat.data_ro)
        values = misfit.dat.data_ro
        n = COMM_WORLD.allreduce(int(values.size), op=MPI.SUM)
        sum_sq = COMM_WORLD.allreduce(float(np.sum(values ** 2)), op=MPI.SUM)
        lo = COMM_WORLD.allreduce(float(values.min()) if values.size
                                  else np.inf, op=MPI.MIN)
        hi = COMM_WORLD.allreduce(float(values.max()) if values.size
                                  else -np.inf, op=MPI.MAX)
        with CheckpointFile(self.path, "a") as chk:
            chk.save_function(observed)
            chk.save_function(misfit)
        return {"max_abs_m": max(abs(lo), abs(hi)),
                "rms_nodal_m": float(np.sqrt(sum_sq / n)) if n else 0.0,
                "min_m": lo, "max_m": hi, "nodes": n}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# The fine schedules, the ice between protocol years, the log, the restart
# --------------------------------------------------------------------------

#: The year at which the protocol changes from 500 yr to 100 yr steps, and
#: at which the fine schedules change their step.
PROTOCOL_SWITCH_YEAR = -26000
#: The last year of every schedule (the end of the GIAMIP protocol).
LAST_YEAR = 1000

#: The 10 yr windows of schedule B (`--schedule fine_b`, design handout
#: section 8.3), by the file of the ice history, as `(first year, end year)`:
#: the fast melt of each history after Lau (2023), Sect. 3.1. Steps that
#: start inside a window are 10 yr long; the other steps after 26 ka are
#: 25 yr long.
FINE_B_WINDOWS = {
    # ICE-6G_D (Exp10, Exp12): MWP-1A, 15 to 13 ka.
    "iceHistory-ICE-6G_D.nc": ((-15000, -13000),),
    # GLAC3c profile 1 (Exp03, Exp09): 15 to 14 ka and 12 to 11 ka.
    "iceHistory-GLAC3c_profile1.nc": ((-15000, -14000), (-12000, -11000)),
    # GLAC3c profile 2 (Exp04, Exp10): 15 to 14 ka and 11 to 10 ka.
    "iceHistory-GLAC3c_profile2.nc": ((-15000, -14000), (-11000, -10000)),
}

#: The rebuild interval of the preconditioner (solves) in the fine
#: schedules when `--gia_rebuild_every` is not given (design handout 8.4,
#: P17): 1 kyr of model time at 10 yr steps.
FINE_REBUILD_EVERY = 100
#: The rebuild guard of the fine schedules when `--gia_rebuild_max_outer` is
#: not given: rebuild when the previous solve took more than this many outer
#: iterations per Newton iteration. The Maxwell protocol runs of Exp10,
#: Exp03, Exp04 and Exp11 (rebuild every 20 solves, 462 steps each) had a
#: median of 2, a 90th percentile of 4 and a largest value of 5, so 6 means
#: "worse than any step of those runs".
FINE_REBUILD_MAX_OUTER = 6.0
#: The default interval of the periodic checkpoints of the fine schedules,
#: in steps. A checkpoint is written at the first protocol year at or after
#: each multiple of this interval (see `--checkpoint_every`).
FINE_CHECKPOINT_EVERY = 200

#: The per-run copy of the input file (or state file) in the output
#: directory. The run loads its meshes and fields from this copy and writes
#: its checkpoints into it, so the meshes, the input fields and the
#: checkpoints are in one file (see THE RESTART in the module docstring).
RUN_STATE_FILE = "forward_state.h5"
#: The file that a run writes when it stops at the wall limit; the job
#: script resubmits the run when it finds it.
RESTART_MARKER = "RESTART_NEEDED"


def fine_dt(name, year, windows=()):
    """The length (yr) of the step of schedule `name` that starts at `year`.

    `fine` (schedule A, decision S19): 50 yr before year -26000, 10 yr from
    there to +1000. `fine_b` (schedule B, the fallback of S25): 100 yr
    before year -26000, then 25 yr, and 10 yr inside the `windows`.
    """
    if name == "fine":
        return 50 if year < PROTOCOL_SWITCH_YEAR else 10
    if name == "fine_b":
        if year < PROTOCOL_SWITCH_YEAR:
            return 100
        if any(first <= year < end for first, end in windows):
            return 10
        return 25
    raise ValueError(f"unknown fine schedule {name!r}")


def fine_years(name, start, protocol_years, windows=()):
    """Every year of a fine schedule, from `start` to `LAST_YEAR`.

    The years are integers, built by adding the step length of `fine_dt`,
    so no rounding error accumulates. Every protocol year must be a year of
    the schedule (the output is written there): the step lengths 50, 100,
    25 and 10 divide the protocol steps of 500 and 100 yr, and the window
    ends are whole kyr.

    Args:
      name: `fine` or `fine_b`.
      start: the first year (the start of the experiment), an integer.
      protocol_years: the protocol years of giamip.
      windows: the 10 yr windows of `fine_b`.

    Returns:
      A NumPy integer array of the years, `start` first.

    Raises:
      RuntimeError: if the schedule does not end at `LAST_YEAR` or misses a
        protocol year.
    """
    years = [int(start)]
    while years[-1] < LAST_YEAR:
        years.append(years[-1] + fine_dt(name, years[-1], windows))
    years = np.asarray(years, dtype=np.int64)
    missing = sorted(set(int(y) for y in protocol_years) - set(years.tolist()))
    if years[-1] != LAST_YEAR or missing:
        raise RuntimeError(f"schedule {name} from {start} ends at "
                           f"{years[-1]} or misses protocol years "
                           f"{missing[:5]}")
    return years


class LinearIce:
    """The ice at any year, linear in time between two protocol slices.

    Design handout P9: ice(t) = (1 - w) ice(k) + w ice(k + 1), with
    w = (t - t_k) / (t_{k+1} - t_k) and t_k <= t < t_{k+1} two neighbouring
    protocol years. The two bracketing slices are kept as CG1 Functions on
    the mantle; a new slice is read from giamip (`SurfaceForcing.ice`) only
    when the year passes t_{k+1}, so the march reads every protocol slice
    once. At a protocol year the ice is a copy of that slice, bit for bit,
    so the load of the fine schedules equals the load of the protocol
    schedule at every protocol year.

    The combination is computed on the nodal values with NumPy, owned and
    halo nodes alike; the halo values equal those of the owners, and the
    next read exchanges them with the same numbers. A UFL expression with
    the weight as a literal would compile a new kernel for every new
    weight.

    Args:
      forcing: the `prepare.SurfaceForcing` of the protocol years.
      years: the protocol years of `forcing`.
    """

    def __init__(self, forcing, years):
        self.forcing = forcing
        self.years = np.asarray(years, dtype=float)
        # Protocol index -> CG1 Function of that slice; at most two entries.
        self.slices = {}
        self.reads = 0

    def _slice(self, k):
        """The slice of protocol index `k`, read once while it is needed."""
        if k not in self.slices:
            self.slices[k] = self.forcing.ice(int(k))
            self.reads += 1
        return self.slices[k]

    def bracket(self, year):
        """The protocol index k with t_k <= year < t_{k+1} (the last k at the end)."""
        return common.ice_bracket(self.years, year)[0]

    def set(self, year, out):
        """Write the ice of `year` into `out` (a CG1 Function on the mantle).

        The bracket, the weight and the combination come from
        `exp10_common.ice_bracket` and `exp10_common.linear_ice_into`, the
        functions with which the prepare script writes the ice of a
        schedule-C input file, so the same year gives the same nodal values
        from the same two slices.

        Returns:
          `(k, w)`: the bracket and the weight of slice k + 1.
        """
        k, w = common.ice_bracket(self.years, year)
        # Drop the slices that the march has passed.
        for old in [j for j in self.slices if j < k]:
            del self.slices[old]
        a = self._slice(k)
        # At a protocol year (w = 0): an exact copy of the slice.
        b = None if w == 0.0 else self._slice(k + 1)
        common.linear_ice_into(out, a, w, b)
        return k, w


class ParamsLog:
    """The scalar log of the run: one header line, one row per output.

    The format of `gadopt.ParameterLog` (columns separated by spaces, written
    on rank 0 and flushed at every row). A restarted run appends to the log
    of the earlier legs: the rows whose output index is above the last
    output of the checkpoint are removed first, because the restarted run
    writes them again.

    Args:
      path: the log file.
      columns: the column names (the header).
      append: continue an existing log (a restart).
      keep_upto: with `append`, the largest output index to keep.
    """

    def __init__(self, path, columns, append=False, keep_upto=None):
        self.handle = None
        if COMM_WORLD.rank != 0:
            return
        if append and os.path.exists(path):
            with open(path) as handle:
                lines = handle.read().splitlines()
            kept = lines[:1] + [line for line in lines[1:]
                                if line.split() and keep_upto is not None
                                and int(line.split()[0]) <= keep_upto]
            with open(path, "w") as handle:
                handle.write("\n".join(kept) + "\n")
            self.handle = open(path, "a")
        else:
            self.handle = open(path, "w")
            self.log_str(" ".join(columns))

    def log_str(self, text):
        """Write one line (rank 0)."""
        if self.handle is not None:
            self.handle.write(text + "\n")
            self.handle.flush()

    def close(self):
        if self.handle is not None:
            self.handle.close()


#: The names of the checkpoint fields in the run's state file.
CHECKPOINT_DISPLACEMENT = "checkpoint_displacement"
CHECKPOINT_INTERNAL = "checkpoint_internal_variables"
CHECKPOINT_POTENTIAL = "checkpoint_potential"
CHECKPOINT_SEA_SURFACE0 = "checkpoint_sea_surface_step0"


class Checkpoints:
    """The checkpoints of the forward run, in the run's copy of the input file.

    A function saved on the mantle submesh would load back onto a submesh
    loaded from the file, a different mesh object from the submesh that
    `read_inputs` cuts from the loaded parent, so the restored fields and
    the solver's fields would live on two meshes. The checkpoint therefore
    stores parent functions. The run loaded its parent mesh from the run's state file
    (`RUN_STATE_FILE`), so a function on that parent mesh can be saved into
    the same file and loaded back onto a parent mesh loaded from it later,
    with the same rank count. The solution lives on two meshes: the
    displacement and the internal variables on the mantle submesh, the
    potential on the parent. Each mantle sub-field is copied onto a parent
    function of the same element with `Function.assign` across the submesh
    map, which copies nodal values exactly (as `exp10_common.append_control`
    does for the bed); the parent cells outside the mantle stay zero. The
    potential is saved as it is. The `Real` sub-fields (core pressure,
    rotation, centre of mass, Shift) are a few numbers, replicated on every
    rank; they are stored exactly as timestepping values of the entry, with
    the step, the year, the output index and the leg.

    The reference of `delta_g` (N + Shift of step 0 on the mantle,
    `SurfaceOutput.S0_m`) is saved too, so the surface fields of a restarted
    run are those of an uninterrupted one.

    Args:
      path: the run's state file.
      parent: the parent mesh, loaded from `path`.
      solver: the `SelfGravitatingGIASolver`.
      layout: its layout.
      output: the `SurfaceOutput` (for `S0_m`).
    """

    def __init__(self, path, parent, solver, layout, output):
        self.path, self.parent = path, parent
        self.solver, self.layout, self.output = solver, layout, output
        if layout.internal_variable_field is None:
            raise RuntimeError("the checkpoint needs the uncondensed layout "
                               "(the internal variables in the solution)")
        z = solver.solution.subfunctions

        def parent_copy(f, name):
            return Function(FunctionSpace(parent, f.function_space()
                                          .ufl_element()), name=name)

        # (mantle or parent Function of the solution, parent copy) pairs.
        self.mantle_pairs = [
            (z[layout.displacement],
             parent_copy(z[layout.displacement], CHECKPOINT_DISPLACEMENT)),
            (z[layout.internal_variable_field],
             parent_copy(z[layout.internal_variable_field],
                         CHECKPOINT_INTERNAL)),
            (output.S0_m, parent_copy(output.S0_m, CHECKPOINT_SEA_SURFACE0)),
        ]
        self.potential = z[layout.potential]
        # A plain Function on the parent (not a Function on the indexed
        # subspace of the mixed space, which `CheckpointFile` does not take).
        self.potential_copy = parent_copy(self.potential, CHECKPOINT_POTENTIAL)
        # Work fields of the round-trip check in `save`.
        self.scratch = [Function(source.function_space())
                        for source, _ in self.mantle_pairs]
        self.real_indices = [i for i, f in enumerate(z)
                             if f.function_space().ufl_element().family()
                             == "Real"]

    def save(self, index, info):
        """Append checkpoint `index` with the timestepping values `info`.

        Collective. Returns the wall time of the save, s.
        """
        tic = time.time()
        z = self.solver.solution.subfunctions
        # `CheckpointFile` stores timestepping values as floats only; the
        # integers (step, output index, leg) are exact as floats.
        values = {key: float(value) for key, value in info.items()}
        for i in self.real_indices:
            values[f"real_{i}"] = float(z[i].dat.data_ro[0])
        for (source, copy), scratch in zip(self.mantle_pairs, self.scratch):
            copy.assign(source, allow_missing_dofs=True)
            # The copy must be exact in both directions: assign it back and
            # compare, so a restart restores exactly what was saved.
            scratch.assign(copy)
            if max_abs_diff(scratch, source) != 0.0:
                raise RuntimeError(f"the parent copy of {copy.name()} is "
                                   "not exact")
        self.potential_copy.assign(self.potential)
        with CheckpointFile(self.path, "a") as chk:
            for _, copy in self.mantle_pairs:
                chk.save_function(copy, idx=int(index),
                                  timestepping_info=values)
            chk.save_function(self.potential_copy, idx=int(index),
                              timestepping_info=values)
        return common.elapsed(tic)

    def latest(self):
        """The timestepping values of the last checkpoint, or None.

        Read with h5py on rank 0 from the attributes that `CheckpointFile`
        writes for the displacement copy, and broadcast.
        """
        found = None
        if COMM_WORLD.rank == 0:
            import h5py
            hits = {}

            # `CheckpointFile` keeps two groups that end with this path: the
            # group of the vector data holds the index, the group of the
            # function holds the user values (step, year, ...). Both are
            # merged into one dict here.
            def visit(name, obj):
                if (name.endswith(f"{CHECKPOINT_DISPLACEMENT}/"
                                  "firedrake_timestepping")):
                    hits.update({key[len("firedrake_timestepping_history_"):]:
                                 np.asarray(value)
                                 for key, value in obj.attrs.items()})

            with h5py.File(self.path, "r") as handle:
                handle.visititems(visit)
            if hits:
                last = int(np.argmax(hits["index"]))
                found = {key: float(value[last]) for key, value in hits.items()}
        return COMM_WORLD.bcast(found, root=0)

    def load(self, info):
        """Restore the solution and `S0_m` from the checkpoint of `info`.

        Collective. After the load `solution_old` equals the solution, which
        is the state at the end of the checkpointed step.
        """
        index = int(info["index"])
        z = self.solver.solution.subfunctions
        with CheckpointFile(self.path, "r") as chk:
            for target, copy in self.mantle_pairs:
                stored = chk.load_function(self.parent, copy.name(), idx=index)
                target.assign(stored)
            stored = chk.load_function(self.parent, CHECKPOINT_POTENTIAL,
                                       idx=index)
            self.potential.assign(stored)
        for i in self.real_indices:
            z[i].dat.data_wo[...] = info[f"real_{i}"]
        self.solver.solution_old.assign(self.solver.solution)
        # The reference of `delta_g` on the surface (`s_S0`), from the
        # restored `S0_m`.
        self.output.set_reference_from_mantle()


def surface_last_index(path):
    """The largest output index in the surface file, or -1 (rank 0, broadcast).

    Read with h5py from the timestepping history that `CheckpointFile` keeps
    for `delta_bed_m`, one of the fields of every output.
    """
    found = -1
    if COMM_WORLD.rank == 0 and os.path.exists(path):
        import h5py
        hits = []

        def visit(name, obj):
            # Two groups end with this path: the group of the vector data
            # holds the index, the group of the function the user values.
            if name.endswith("delta_bed_m/firedrake_timestepping"):
                value = obj.attrs.get("firedrake_timestepping_history_index")
                if value is not None:
                    hits.append(np.asarray(value))

        with h5py.File(path, "r") as handle:
            handle.visititems(visit)
        if hits and hits[0].size:
            found = int(np.max(hits[0]))
    return COMM_WORLD.bcast(found, root=0)


def parse_args():
    """The command line. The solver options and defaults are those of
    `invert_exp10.py`."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state", required=True,
                   help="the state.h5 of an inversion (the input file with "
                        "the bed of every ROL iteration)")
    p.add_argument("--controls", default=None,
                   help="the controls.h5 of the same inversion, for a check "
                        "of the bed (default: controls.h5 beside --state; "
                        "'none' to skip)")
    p.add_argument("--output_path", default="./")
    p.add_argument("--schedule",
                   choices=["inversion", "protocol", "fine", "fine_b",
                            "windows"],
                   default="inversion",
                   help="the steps of the march: those of the input file "
                        "(2 kyr, then +1000 yr), every model year of the "
                        "GIAMIP protocol (500 yr to 26 ka, then 100 yr, to "
                        "+1000 yr), with the ice read from giamip, or the "
                        "fine schedules A (fine: 50 yr, then 10 yr) and B "
                        "(fine_b: 100 yr, then 25 yr and 10 yr in the fast "
                        "melt windows), with the ice linear in time between "
                        "the protocol years and the output at the protocol "
                        "years only, or schedule C (windows: the protocol "
                        "steps plus 25 yr and 10 yr steps in the fast melt "
                        "windows, the march of the inversion on schedule "
                        "C)")
    p.add_argument("--steps", type=int, default=None,
                   help="stop the march after this many steps")
    p.add_argument("--plus_step", choices=["auto", "always", "never"],
                   default="auto",
                   help="the step to +1000 yr: after a march to year 0 "
                        "(auto), after any march (always, a test), or not")
    p.add_argument("--expected_J", type=float, default=None,
                   help="the J to compare with at year 0 (default: the last "
                        "J of the state file)")
    # The solver options of invert_exp10.py, with the same defaults.
    p.add_argument("--k_s", type=float, default=0.9422,
                   help="the fluid Love number of the bulge")
    p.add_argument("--gia_rebuild", choices=["per_solve", "lagged", "always"],
                   default=None)
    p.add_argument("--gia_rebuild_every", type=int, default=None,
                   help="solves between preconditioner rebuilds on the "
                        "lagged branch (default: the solver default, 5, for "
                        "the inversion schedule; PROTOCOL_REBUILD_EVERY for "
                        "the protocol schedule)")
    p.add_argument("--gia_rebuild_max_outer", type=float, default=None,
                   help="rebuild the preconditioner when the previous solve "
                        "took more outer iterations per Newton iteration "
                        "(default: off; FINE_REBUILD_MAX_OUTER for the fine "
                        "schedules)")
    p.add_argument("--checkpoint_every", type=int, default=None,
                   help="write a checkpoint at the first output at or after "
                        "every this many steps (default: "
                        "FINE_CHECKPOINT_EVERY for the fine schedules, off "
                        "otherwise)")
    p.add_argument("--wall_limit_hours", type=float, default=None,
                   help="write a checkpoint and stop before this wall time "
                        "(h); the job script resubmits with --restart")
    p.add_argument("--stop_after_step", type=int, default=None,
                   help="a test of the restart: write a checkpoint after "
                        "this step and stop as at the wall limit")
    p.add_argument("--restart", action="store_true",
                   help="continue the run in --output_path from its last "
                        "checkpoint")
    p.add_argument("--ice_check_years", type=int, default=3,
                   help="fine schedules: compare the ice of this many "
                        "protocol years after the start with a fresh read "
                        "from giamip, bit for bit")
    p.add_argument("--block0_max_it", type=int, default=50)
    p.add_argument("--outer_atol", type=float, default=0.0)
    p.add_argument("--outer_max_it", type=int, default=30)
    return p.parse_args()


def check_controls(path, bed_s, say):
    """Compare the bed with the last control of `controls.h5`.

    `controls.h5` holds the ROL vector of each iteration on the surface mesh
    of the inversion, saved directly (not through the state file). The
    surface mesh loads from it as a new mesh, so the two fields cannot be
    compared node by node here; the nodal min and max are compared instead,
    and must be equal.

    The integral is not compared. This run has the rank count of the
    inversion, and on that count `CheckpointFile` loads the surface submesh
    with its saved distribution, a load that counts cells more than once
    (job 51182555: area 248.4 instead of 61.0 on 384 ranks; the file is
    right, 1 and 16 ranks give the integral of the state file to 1e-11).
    The broken load is not even reproducible: the forward run on 384 ranks
    gave a third value, -480390.14, for the same file.
    The nodal values load correctly, so the min and max are a valid check.

    Returns:
      A dict with both sets of numbers and the differences, or None if the
      file does not exist.
    """
    if not os.path.exists(path):
        say(f"  controls check: {path} not found, skipped")
        return None
    with CheckpointFile(path, "r") as chk:
        mesh = chk.load_mesh(common.SURFACE_NAME)
        history = chk.get_timestepping_history(mesh, common.CONTROL_NAME)
        index = int(max(history["index"]))
        stored = chk.load_function(mesh, common.CONTROL_NAME, idx=index)

    def numbers(f):
        return common.field_statistics(f, Measure("dx", domain=f.function_space().mesh()))

    ours, theirs = numbers(bed_s), numbers(stored)
    # Only the min and max are valid here (see the docstring).
    for numbers_ in (ours, theirs):
        numbers_.pop("integral")
    out = {"file": path, "iteration": index, "state_bed": ours,
           "controls_bed": theirs,
           "difference": {key: ours[key] - theirs[key] for key in ours}}
    say(f"  controls check: iteration {index} of {path}: min {theirs['min']:.6f}"
        f" / {ours['min']:.6f} m, max {theirs['max']:.6f} / {ours['max']:.6f} "
        f"m (controls.h5 / state.h5)")
    return out


def main():
    """Run the forward march and write the outputs."""
    args = parse_args()
    protocol = args.schedule == "protocol"
    # Schedule C (`windows`) is a fine schedule: output at the protocol
    # years, the same preconditioner rule, checkpoints and restart.
    fine = args.schedule in ("fine", "fine_b", "windows")
    # The protocol schedule rebuilds the preconditioner every 20 solves
    # unless the command line says otherwise; the inversion schedule keeps
    # the solver default, so that it repeats the inversion's own march. The
    # fine schedules rebuild every 100 solves and when a solve needed more
    # than FINE_REBUILD_MAX_OUTER outer iterations per Newton iteration
    # (design handout 8.4).
    if protocol and args.gia_rebuild_every is None:
        args.gia_rebuild_every = PROTOCOL_REBUILD_EVERY
    if fine and args.gia_rebuild_every is None:
        args.gia_rebuild_every = FINE_REBUILD_EVERY
    if fine and args.gia_rebuild_max_outer is None:
        args.gia_rebuild_max_outer = FINE_REBUILD_MAX_OUTER
    # Checkpoints: periodic in the fine schedules, and whenever a wall limit
    # is set. They need the run's own copy of the state file (THE RESTART).
    if args.checkpoint_every is None:
        args.checkpoint_every = FINE_CHECKPOINT_EVERY if fine else 0
    checkpointing = (args.checkpoint_every > 0
                     or args.wall_limit_hours is not None
                     or args.stop_after_step is not None)
    if args.restart and not checkpointing:
        raise SystemExit("--restart needs checkpoints: a fine schedule, "
                         "--checkpoint_every or --wall_limit_hours")
    if COMM_WORLD.rank == 0:
        os.makedirs(args.output_path, exist_ok=True)
    COMM_WORLD.barrier()
    say = common.RunLog(os.path.join(args.output_path, LOG_FILE))
    tic_run = time.time()
    say("=" * 78)
    say(f"GIAMIP Exp10 forward run from an inverted bed, {COMM_WORLD.size} "
        f"ranks, schedule {args.schedule}"
        f"{', RESTART' if args.restart else ''}")
    say(f"  annotation on: {annotate_tape()} (must be False)")
    say("=" * 78)
    if annotate_tape():
        raise RuntimeError("the tape is annotating; this run must not tape")
    checks = {}

    # ---- the state file ---------------------------------------------------
    # With checkpoints the run works on its own copy of the state file in
    # the output directory: the meshes and fields are loaded from the copy,
    # and the checkpoints are written into it, so a restart loads the meshes,
    # the fields and the checkpoint from one file (THE RESTART). A restart
    # uses the copy of the first leg.
    tic = time.time()
    state_path = args.state
    if checkpointing:
        state_path = os.path.join(args.output_path, RUN_STATE_FILE)
        if not args.restart:
            if COMM_WORLD.rank == 0:
                shutil.copyfile(args.state, state_path)
            COMM_WORLD.barrier()
        elif not os.path.exists(state_path):
            raise SystemExit(f"--restart: {state_path} does not exist")
        say(f"  run state file: {state_path} ({'existing' if args.restart else 'copied from ' + args.state})")
    parent, mantle, fields, ice, record, (bed_mantle, bed_iteration) = \
        common.read_inputs(state_path, last_control=True)
    say(f"  state: {args.state}, experiment {record['experiment']}, mesh "
        f"{record['mesh']['md5']} refine {record['mesh']['refine_levels']}, "
        f"{record['mesh']['cells_mantle']} mantle cells, bed of ROL iteration "
        f"{bed_iteration} ({time.time() - tic:.1f} s)")
    # The experiment of the run comes from the record of the state file, and
    # the ice transfer comes from the prepare script of this directory. A
    # state file of another experiment would run that other experiment and
    # write its results under the file names of this one, with no error.
    # Stop here, before the solver build and the march.
    if record["experiment"] != prepare.EXPERIMENT:
        raise RuntimeError(
            f"the state file {args.state} is of experiment "
            f"{record['experiment']}, but this script is for "
            f"{prepare.EXPERIMENT}; set STATE to the state file of "
            f"{prepare.EXPERIMENT}")
    found = common.input_statistics(mantle, fields, ice)
    bad = common.compare_statistics(record["statistics"], found)
    if bad:
        raise RuntimeError(f"the state file did not load as written: {bad[:5]}")
    say("  inputs: min, max and integral of every field agree with the "
        "prepare record")
    scales = common.Scales.from_dict(record["scales"])
    J_state, rms_state = last_state_J(state_path)
    if args.expected_J is not None:
        J_state = args.expected_J
    say(f"  last J of the state file: {J_state} (RMS misfit {rms_state} m)")

    # ---- the forcing Functions and the solver (as invert_exp10.py) ---------
    # `SL_init` starts as the observed sea level, as in the inversion, and
    # is overwritten from the bed after the solver is built; the inversion
    # builds its solver in the same order.
    SL_obs = fields["SL_obs"]
    SL_init = Function(SL_obs.function_space(), name="SL_init").assign(SL_obs)
    I_init = Function(ice[0].function_space(), name="ice_init").assign(ice[0])
    I = Function(ice[0].function_space(), name="ice").assign(ice[0])
    # The time step, a Real Function (`common.time_step`).
    dt = common.time_step(mantle)
    tic = time.time()
    solver, layout, solver_record = common.build_solver(
        parent, mantle, fields, scales, record["solver_constants"],
        {"SL_init": SL_init, "I": I, "I_init": I_init}, dt, args,
        record=record)
    say(f"  solver: {solver_record['unknowns']} unknowns, "
        f"{solver_record['real_rows']} Real rows, k_s {args.k_s}, rebuild "
        f"{solver_record['gia_rebuild']} every "
        f"{solver_record['gia_rebuild_every']} ({time.time() - tic:.1f} s)")

    # ---- the bed and SL_init through the exact node map --------------------
    surface = common.surface_mesh(mantle)
    V_s2 = FunctionSpace(surface, "CG", 2)
    node_map = common.SurfaceNodeMap(V_s2, SL_init.function_space(),
                                     common.SURF_RE)
    bed_s = Function(V_s2, name=common.CONTROL_NAME)
    node_map.restrict(bed_mantle, out=bed_s)
    # SL_init = -bed / D on the mantle nodes on Re, zero elsewhere: the
    # first block of the inversion tape, on the same bed.
    node_map.extend(bed_s, SL_init, scale=-1.0 / D_SCALE)
    # Check 1: extend(restrict(stored)) is the stored field (the stored
    # field is zero off Re and carries the bed on Re).
    back = Function(SL_init.function_space())
    node_map.extend(bed_s, back)
    d1 = max_abs_diff(back, bed_mantle)
    # Check 2: restrict(extend(b)) is b.
    again = Function(V_s2)
    node_map.restrict(back, out=again)
    d2 = max_abs_diff(again, bed_s)
    # Check 3: SL_init on Re is -b / D.
    sl_s = Function(V_s2)
    node_map.restrict(SL_init, out=sl_s)
    expected = Function(V_s2).assign(-bed_s / D_SCALE)
    d3 = max_abs_diff(sl_s, expected)
    say(f"  bed: |extend(restrict(stored)) - stored| = {d1:.3e} m, "
        f"|restrict(extend(b)) - b| = {d2:.3e} m, |SL_init - (-b/D)| on Re = "
        f"{d3:.3e}; bed range {-vec_max_abs(bed_s):.1f} .. "
        f"{vec_max_abs(bed_s):.1f} m (abs)")
    checks["bed_node_map"] = {"extend_restrict_stored_m": d1,
                              "restrict_extend_b_m": d2,
                              "SL_init_minus_b_over_D": d3}
    if d1 != 0.0 or d2 != 0.0:
        raise RuntimeError("the bed did not come back exactly through the "
                           "node map")
    controls = args.controls
    if controls is None:
        controls = os.path.join(os.path.dirname(os.path.abspath(args.state)),
                                "controls.h5")
    if controls != "none":
        checks["controls"] = check_controls(controls, bed_s, say)

    # ---- the schedule of the march and its forcing -------------------------
    # The march of the input file (the inversion's steps): its years, the
    # step of year 0 and the number of ice fields in the file.
    years = np.asarray(record["march"]["years"], dtype=float)
    year0 = int(record["march"]["year0_step"])
    n_ice = int(record["march"]["ice_steps"])
    tic = time.time()
    experiment = GiamipExperiment(record["experiment"])
    # Every model year of the protocol and the slice of the ice history that
    # belongs to it, as giamip defines them.
    protocol_years = np.asarray(experiment.model_years(), dtype=float)
    protocol_steps = np.asarray(experiment.forcing_steps())
    # The steps whose ice comes from the input file in a fine schedule
    # (schedule C with a schedule-C state file); empty otherwise.
    file_steps = set()
    # Each entry of `schedule` is (step, year, dt_yr, ice): `ice` is a
    # callable that returns the ice of the end of the step (backward Euler)
    # as a CG1 Function on the mantle, so the protocol schedule reads one
    # step of ice at a time and never holds the whole history.
    if protocol or fine:
        # `step_years = 1` keeps every protocol year in `coarse_step_indices`
        # (every distance from the start is a whole number of years), so the
        # forcing steps are the protocol steps, read with the transfer of the
        # prepare script.
        forcing = prepare.SurfaceForcing(mantle, experiment, 1)
        f_years = np.asarray(forcing.years, dtype=float)
        f_steps = np.asarray(forcing.steps)
        n_forcing = len(f_years)
        is_protocol = (np.array_equal(f_years, protocol_years)
                       and np.array_equal(f_steps, protocol_steps))
        # A state file of the schedule-C march (S34): its years between two
        # protocol years hold the linear ice of `exp10_common.linear_ice_into`
        # and have no protocol slice of their own (protocol step -1 in the
        # record). With `--schedule windows` its steps up to year 0 take
        # their ice from the file (SCHEDULE C in the module docstring).
        c_file = (args.schedule == "windows"
                  and record["march"].get("schedule") == "windows")
        # The years of the input file that are protocol years: every year
        # of a 2 kyr march, the protocol years of a schedule-C march.
        f_year_set = set(f_years.tolist())
        p_index = np.array([k for k, y in enumerate(years)
                            if (not c_file) or float(y) in f_year_set],
                           dtype=int)
        # The protocol position of each of those years. Each of them is a
        # protocol year with the same ice slice.
        positions = np.searchsorted(f_years, years[p_index])
        inside = bool(np.all(positions < n_forcing))
        shared = (inside
                  and np.array_equal(f_years[np.minimum(positions,
                                                        n_forcing - 1)],
                                     years[p_index])
                  and np.array_equal(
                      f_steps[np.minimum(positions, n_forcing - 1)],
                      np.asarray(record["march"]["protocol_steps"])[p_index]))
        zero = np.flatnonzero(f_years == 0.0)
        dts = np.diff(f_years)
        say(f"  forcing: {n_forcing} protocol years from giamip "
            f"({f_years[0]:.0f} .. {f_years[-1]:.0f}), equal to the protocol: "
            f"{is_protocol}; the {n_ice} years of the input file are protocol "
            f"years with the same ice slices: {shared}; steps of "
            f"{sorted(set(dts.tolist()))} yr; year 0 is step "
            f"{int(zero[0]) if zero.size else None}")
        if (record["march"].get("schedule") == "windows"
                and args.schedule != "windows"):
            raise SystemExit("the state file holds the march of schedule C; "
                             "run it with --schedule windows")
        if not (is_protocol and shared and zero.size == 1):
            raise RuntimeError("the forcing steps of giamip are not the "
                               "protocol years, or do not contain the march "
                               "of the input file")
        year0_step = int(zero[0])
        # The transfer test at every shared year: the ice made here against
        # the ice of the input file (made by the prepare job with the same
        # transfer), and the observed sea level. The run stops before the
        # march if one differs, so the ice of this run is the ice of the
        # inversion at every year that both have.
        worst = {"ice_max_diff_m": 0.0, "year": None, "ice_max_m": 0.0}
        # A schedule-C file is compared in the loop below, at every year
        # with `LinearIce`, which reads each protocol slice once.
        for k, position in ([] if c_file else zip(p_index, positions)):
            here = forcing.ice(int(position))
            d_ice = max_abs_diff(here, ice[k]) * D_SCALE
            ice_max = vec_max_abs(ice[k]) * D_SCALE
            if d_ice > 1e-9 * max(ice_max, 1.0):
                raise RuntimeError(
                    f"the ice of year {years[k]:.0f} made here differs from "
                    f"the input file by {d_ice:.3e} m")
            if d_ice >= worst["ice_max_diff_m"]:
                worst = {"ice_max_diff_m": d_ice, "year": float(years[k]),
                         "ice_max_m": ice_max}
            del here
        if c_file:
            # Every year of the file against the ice made here from giamip
            # by `LinearIce`, with the same tolerance: at a protocol year a
            # copy of the slice transferred here, between protocol years
            # the combination of two such slices. `LinearIce` reads each
            # protocol slice once in this pass.
            here_linear = LinearIce(forcing, f_years)
            here = Function(forcing.cg1)
            for k, year in enumerate(years):
                here_linear.set(float(year), here)
                d_ice = max_abs_diff(here, ice[k]) * D_SCALE
                ice_max = vec_max_abs(ice[k]) * D_SCALE
                if d_ice > 1e-9 * max(ice_max, 1.0):
                    raise RuntimeError(
                        f"the ice of year {year:.0f} made here differs "
                        f"from the input file by {d_ice:.3e} m")
                if d_ice >= worst["ice_max_diff_m"]:
                    worst = {"ice_max_diff_m": d_ice, "year": float(year),
                             "ice_max_m": ice_max}
            del here, here_linear
            # Inside the file: the ice of every year between two protocol
            # years is the combination of the two protocol slices of the
            # file, so the inversion marched the linear ice. The prepare job
            # checked this bit for bit before the write; after the read
            # (`read_inputs` interpolates from the parent copy) it holds to
            # round-off, so the test is relative to the largest ice.
            linear_check = common.check_linear_ice(
                years, ice, f_years, Function(ice[0].function_space()))
            linear_tol = 1e-12 * max(linear_check["max_abs_ice"], 1e-30)
            linear_check["tolerance"] = linear_tol
            say(f"  input file of schedule C: the ice of the "
                f"{linear_check['years_checked']} years between protocol "
                f"years against the linear combination of its protocol "
                f"slices: largest difference "
                f"{linear_check['max_abs_diff']:.3e} (tolerance "
                f"{linear_tol:.3e}, units of D)")
            checks["linear_ice_of_file"] = linear_check
            if linear_check["max_abs_diff"] > linear_tol:
                raise RuntimeError("the ice of the input file is not linear "
                                   "between its protocol slices")
        d_sl = max_abs_diff(forcing.SL_obs, SL_obs) * D_SCALE
        say(f"  transfer test: ice of the {n_ice} shared years, giamip now "
            f"against the input file: largest difference "
            f"{worst['ice_max_diff_m']:.3e} m (year {worst['year']:.0f}); "
            f"observed bed: largest difference {d_sl:.3e} m")
        checks["transfer"] = {"shared_years": n_ice, **worst,
                              "SL_obs_max_diff_m": d_sl}
        if d_sl > 1e-6:
            raise RuntimeError("the observed bed of this run does not "
                               "reproduce the input file")
        # The ice of the 2 kyr steps is not used any more; only the ice of
        # step 0 (in `I` and `I_init`) is. Dropping the list frees one CG1
        # field per 2 kyr step. A schedule-C file keeps it: its steps up to
        # year 0 take their ice from it.
        if not c_file:
            ice = None
    if protocol:
        last = n_forcing - 1 if args.steps is None else min(n_forcing - 1,
                                                            args.steps)
        # Each entry: (step, year, dt_yr, set_ice, output index). The output
        # index of a protocol step is the step.
        schedule = [(index, float(f_years[index]),
                     float(f_years[index] - f_years[index - 1]),
                     (lambda index=index: I.assign(forcing.ice(index))),
                     index)
                    for index in range(1, last + 1)]
        dt_change = [float(f_years[i]) for i in range(1, len(dts))
                     if dts[i] != dts[i - 1]]
        say(f"  forcing read ({time.time() - tic:.1f} s); protocol steps "
            f"1..{last} to year {float(f_years[last]):.0f}, dt changes after "
            f"year {dt_change}; rebuild every {args.gia_rebuild_every} solves "
            f"and at a change of dt")
        checks["schedule"] = {"name": "protocol", "steps": last,
                              "year0_step": year0_step,
                              "dt_values_yr": sorted(set(dts.tolist())),
                              "dt_change_after_year": dt_change}
    elif fine:
        # The fine schedules (design handout 8.3 and 12.7): every year of
        # the schedule, the ice linear in time between the protocol slices
        # (`LinearIce`), and an output at the protocol years only, with the
        # protocol position as the output index, so the surface file holds
        # a contiguous series 0, 1, 2, ... like the protocol schedule.
        ice_file = os.path.basename(str(forcing.ice_history.path))
        if args.schedule == "windows":
            # Schedule C: the years of `exp10_common.schedule_c_years`, the
            # function of the prepare script of schedule C.
            windows = common.schedule_c_windows(ice_file)
            fine_y = common.schedule_c_years(int(f_years[0]), f_years,
                                             windows, LAST_YEAR)
        else:
            windows = FINE_B_WINDOWS.get(ice_file)
            if args.schedule == "fine_b" and windows is None:
                raise RuntimeError(f"no 10 yr windows of schedule B for the "
                                   f"ice history {ice_file}")
            fine_y = fine_years(args.schedule, int(f_years[0]), f_years,
                                windows or ())
        fine_dts = np.diff(fine_y)
        position = {int(y): k for k, y in enumerate(f_years)}
        linear = LinearIce(forcing, f_years)
        n_fine = len(fine_y) - 1
        last = n_fine if args.steps is None else min(n_fine, args.steps)
        if c_file:
            # The march of the file must be the beginning of this schedule,
            # to year 0, so its step j is step j here.
            if not (np.array_equal(years, fine_y[:n_ice].astype(float))
                    and float(fine_y[year0]) == 0.0
                    and year0 == n_ice - 1):
                raise RuntimeError("the march of the schedule-C input file "
                                   "is not schedule C of this run to year 0")
            # The steps 1 .. year 0 take the ice of the file, as the
            # inversion does (`I.assign(ice[step])`).
            file_steps = set(range(1, n_ice))
        schedule = [(index, float(fine_y[index]), float(fine_dts[index - 1]),
                     ((lambda index=index: I.assign(ice[index]))
                      if index in file_steps else
                      (lambda year=float(fine_y[index]): linear.set(year, I))),
                     position.get(int(fine_y[index])))
                    for index in range(1, last + 1)]
        year0_step = int(np.flatnonzero(fine_y == 0)[0])
        dt_change = [float(fine_y[i]) for i in range(1, len(fine_dts))
                     if fine_dts[i] != fine_dts[i - 1]]
        before = int(np.sum(fine_y[1:] <= PROTOCOL_SWITCH_YEAR))
        say(f"  forcing read ({time.time() - tic:.1f} s); schedule "
            f"{args.schedule}: {n_fine} steps from {fine_y[0]} to "
            f"{fine_y[-1]} ({before} to year {PROTOCOL_SWITCH_YEAR}, "
            f"{n_fine - before} after), steps of "
            f"{sorted(set(fine_dts.tolist()))} yr, year 0 is step "
            f"{year0_step}, dt changes after year {dt_change}; output at the "
            f"{n_forcing} protocol years; this run: steps 1..{last}; rebuild "
            f"every {args.gia_rebuild_every} solves, at a change of dt and "
            f"above {args.gia_rebuild_max_outer} outer iterations per Newton "
            f"iteration")
        checks["schedule"] = {"name": args.schedule, "steps": last,
                              "steps_total": n_fine,
                              "steps_before_switch": before,
                              "year0_step": year0_step,
                              "dt_values_yr": sorted(set(fine_dts.tolist())),
                              "dt_change_after_year": dt_change,
                              # The 10 yr windows of schedule B, or the
                              # (first, end, step) windows of schedule C.
                              "windows_10yr": (windows if args.schedule
                                               != "windows" else None),
                              "windows": (windows if args.schedule
                                          == "windows" else None),
                              "ice_history": ice_file,
                              "ice_from_input_file_steps": (
                                  [min(file_steps), max(file_steps)]
                                  if file_steps else None)}
    else:
        if record["march"].get("schedule") == "windows":
            raise SystemExit("the state file holds the march of schedule C; "
                             "run it with --schedule windows")
        last = year0 if args.steps is None else min(year0, args.steps)
        plus = (args.plus_step == "always"
                or (args.plus_step == "auto" and last == year0))
        forcing = prepare.SurfaceForcing(mantle, experiment,
                                         record["march"]["step_years"])
        # The march of the input file is the beginning of the forcing steps.
        same_years = np.array_equal(
            np.asarray(forcing.years[:n_ice], dtype=float), years)
        same_steps = np.array_equal(
            np.asarray(forcing.steps[:n_ice]),
            np.asarray(record["march"]["protocol_steps"]))
        # The step after year 0 must be the last forcing step of the protocol.
        n_forcing = len(forcing.years)
        plus_year = float(forcing.years[-1])
        plus_is_last = (n_forcing == n_ice + 1
                        and int(forcing.steps[-1]) == int(protocol_steps[-1])
                        and plus_year == float(protocol_years[-1]))
        say(f"  forcing: {n_forcing} steps from giamip, the input file has "
            f"{n_ice}; years and protocol steps of the march equal: "
            f"{same_years and same_steps}; step {n_forcing - 1} is year "
            f"{plus_year:.0f}, protocol step {int(forcing.steps[-1])} of "
            f"{len(protocol_steps)} (the last: {plus_is_last})")
        if not (same_years and same_steps and plus_is_last):
            raise RuntimeError("the forcing steps of giamip do not continue "
                               "the march of the input file with one last "
                               "step")
        # The transfer test: the ice of year 0 made here against the ice of
        # the input file, and the observed sea level likewise.
        ice_year0 = forcing.ice(year0)
        d_ice = max_abs_diff(ice_year0, ice[year0]) * D_SCALE
        ice_max = vec_max_abs(ice[year0]) * D_SCALE
        d_sl = max_abs_diff(forcing.SL_obs, SL_obs) * D_SCALE
        say(f"  transfer test: ice of year 0, giamip now against the input "
            f"file: largest difference {d_ice:.3e} m (largest ice "
            f"{ice_max:.1f} m); observed bed: largest difference "
            f"{d_sl:.3e} m")
        checks["transfer"] = {"ice_year0_max_diff_m": d_ice,
                              "ice_year0_max_m": ice_max,
                              "SL_obs_max_diff_m": d_sl}
        if d_ice > 1e-9 * max(ice_max, 1.0) or d_sl > 1e-6:
            raise RuntimeError("the transfer of this run does not reproduce "
                               "the input file")
        ice_plus = forcing.ice(n_forcing - 1) if plus else None
        dt_plus_yr = plus_year - float(years[year0])
        del forcing
        year0_step = year0
        schedule = [(index, float(years[index]),
                     float(years[index] - years[index - 1]),
                     (lambda index=index: I.assign(ice[index])), index)
                    for index in range(1, last + 1)]
        if plus:
            schedule.append((last + 1, plus_year, dt_plus_yr,
                             (lambda: I.assign(ice_plus)), last + 1))
        say(f"  forcing read ({time.time() - tic:.1f} s); march steps "
            f"1..{last}"
            f"{f', then +{dt_plus_yr:.0f} yr to year {plus_year:.0f}' if plus else ''}")
        checks["schedule"] = {"name": "inversion", "steps": len(schedule),
                              "year0_step": year0_step}

    # ---- outputs ------------------------------------------------------------
    tic = time.time()
    surface_path = os.path.join(args.output_path, SURFACE_FILE)
    output = SurfaceOutput(solver, layout, mantle, surface, node_map,
                           {"SL_init": SL_init, "I": I, "SL_obs": SL_obs},
                           scales, surface_path, new_file=not args.restart)
    checkpoints = (Checkpoints(state_path, parent, solver, layout, output)
                   if checkpointing else None)
    # The restart: the last checkpoint of the run's state file gives the
    # step, the year, the output index and the solution at the end of that
    # step. The surface file and the parameter log of the earlier legs are
    # continued; outputs that a crashed leg wrote after its last checkpoint
    # are not written again (the surface file keeps them, the parameter log
    # is cut back to the checkpoint and gets those rows again).
    restored = None
    last_written = -1
    if args.restart:
        restored = checkpoints.latest()
        if restored is None:
            raise SystemExit(f"--restart: {state_path} holds no checkpoint")
        checkpoints.load(restored)
        last_written = surface_last_index(surface_path)
        say(f"  restart from checkpoint {int(restored['index'])}: step "
            f"{int(restored['step'])}, year {restored['year']:.0f}, output "
            f"{int(restored['out_index'])}, leg {int(restored['leg']) + 1}; "
            f"the surface file holds outputs up to {last_written}")
    params = ParamsLog(os.path.join(args.output_path, PARAMS_FILE),
                       PARAM_COLUMNS, append=args.restart,
                       keep_upto=(int(restored["out_index"])
                                  if restored else None))
    say(f"  output set up ({time.time() - tic:.1f} s), surface area "
        f"{output.area:.6f} D^2")
    rows = []
    fine_rows = []

    def record_step(step, year, dt_yr, newton, outer, solve_s):
        """Write the fields and one row of scalars of the current state.

        `step` is the output index: the step in the inversion and protocol
        schedules, the protocol position in the fine schedules. An output
        that the surface file already holds (a restart after a crash) is
        not written again; its row goes to the parameter log only.
        """
        tic_w = time.time()
        if step > last_written:
            info, sums = output.write(step, year, first=(step == 0))
        else:
            info, sums = output.write_info(step, year), None
        values = output.scalars()
        row = {"step": step, "year": year, "dt_yr": dt_yr, "newton": newton,
               "outer": outer, "shift_m": info["shift_m"], "m1": info["m1"],
               "m2": info["m2"], "m3": info["m3"], **values,
               "solve_s": solve_s, "write_s": time.time() - tic_w}
        # The facet means sum to the solver's integral of C.
        row["ocean_C_facet_sum_rel"] = (
            (sums["ocean_function"] - values["ocean_C_nondim"])
            / values["ocean_C_nondim"]
            if sums and values["ocean_C_nondim"] else 0.0)
        row["rebuilds"] = rebuild_counts(solver)
        rows.append(row)
        params.log_str(" ".join(
            f"{row[c]:d}" if c in ("step", "newton", "outer")
            else f"{row[c]:.10e}" for c in PARAM_COLUMNS))
        say(f"  step {step:3d}  year {year:8.0f}  dt {dt_yr:5.0f} yr  newton "
            f"{newton}  outer {outer:3d}  shift {row['shift_m']:9.4f} m  "
            f"mean dg {row['mean_delta_g_m']:9.4f} m  solve {solve_s:6.1f} s"
            f"  write {row['write_s']:5.1f} s")
        say(f"           J {row['misfit_J']:.6e} (RMS {row['rms_misfit_m']:.4f}"
            f" m), N+Shift nodal vs solver RMS "
            f"{row['sea_surface_check_rms_m']:.2e} m, facet sum of C rel "
            f"{row['ocean_C_facet_sum_rel']:.1e}, builds {row['rebuilds']}, "
            f"peak memory {common.peak_memory_gb():.2f} GB")
        write_summary("running")
        return row

    def write_summary(state, **extra):
        """The summary file of the run, with its state."""
        common.write_json(os.path.join(args.output_path, SUMMARY_FILE),
                          {"args": vars(args), "state": state,
                           "solver": solver_record, "checks": checks,
                           "rows": rows, "fine_steps": fine_rows,
                           "restored": restored, **extra})

    # Step 0: the undeformed start with the ice of step 0 (not on a restart).
    if restored is None:
        record_step(0, float(years[0]), 0.0, 0, 0, 0.0)

    # ---- the march ----------------------------------------------------------
    previous_dt = None
    tic_march = time.time()
    first_step = 1 if restored is None else int(restored["step"]) + 1
    leg = 0 if restored is None else int(restored["leg"]) + 1
    n_checkpoints = 0 if restored is None else int(restored["index"]) + 1
    last_checkpoint_step = 0 if restored is None else int(restored["step"])
    checkpoint_s = 0.0
    recent = []
    # The solve time since the last output (carried across a restart) and
    # the last output index (for the checkpoint of a stop between outputs).
    solve_since_output = (0.0 if restored is None
                          else float(restored.get("solve_since_output", 0.0)))
    last_out_index = 0 if restored is None else int(restored["out_index"])
    last_builds = None
    ice_checks = []
    stopped = None
    for step, year, dt_yr, set_ice, out_index in schedule:
        if step < first_step:
            continue
        if dt_yr != previous_dt:
            # The time step in units of t_bar. The preconditioner rebuilds
            # at the next solve (see the module docstring).
            dt.assign(dt_yr / common.T_BAR_YR)
            previous_dt = dt_yr
        # Backward Euler: the load at the end of the step. In the protocol
        # schedule the callable reads and transfers the ice of this step
        # now; in the fine schedules it combines the two bracketing slices,
        # or (schedule C with a schedule-C file, steps to year 0) copies the
        # ice of the input file.
        set_ice()
        if (fine and out_index is not None and step not in file_steps
                and len(ice_checks) < args.ice_check_years):
            # The ice of a protocol year must be the protocol slice bit for
            # bit: compare with a fresh read and transfer from giamip. The
            # steps with the ice of the input file are not compared: that
            # ice is the prepare job's transfer, which differs from a fresh
            # transfer by round-off; the transfer test before the march
            # compared it.
            fresh = forcing.ice(out_index)
            d = max_abs_diff(fresh, I)
            ice_checks.append({"year": year, "max_abs_diff": d})
            checks["ice_at_protocol_years"] = ice_checks
            say(f"  ice check at protocol year {year:.0f}: largest "
                f"difference to a fresh read {d:.3e} (must be 0)")
            if d != 0.0:
                raise RuntimeError(f"the ice of protocol year {year:.0f} is "
                                   "not the protocol slice")
            del fresh
        tic = time.time()
        solver.solve()
        solve_s = common.elapsed(tic)
        newton, outer = common.iteration_counts(solver)
        solve_since_output += solve_s
        recent = (recent + [solve_s])[-20:]
        if fine:
            fine_rows.append([step, year, dt_yr, newton, outer,
                              round(solve_s, 2)])
            # The build counts of the preconditioner, in the line only when
            # they changed (a rebuild by the lag, a change of dt or the
            # outer-iteration guard).
            builds = rebuild_counts(solver)
            changed = builds != last_builds
            last_builds = builds
            say(f"  fine step {step:5d}  year {year:8.0f}  dt {dt_yr:4.0f} yr"
                f"  newton {newton}  outer {outer:3d}  solve {solve_s:6.1f} s"
                f"{'  (output ' + str(out_index) + ')' if out_index is not None else ''}"
                f"{'  builds ' + str(builds) if changed else ''}")
        row = None
        if out_index is not None:
            row = record_step(out_index, year, dt_yr, newton, outer,
                              solve_since_output)
            solve_since_output = 0.0
            last_out_index = out_index
        # On a restart after a crash, an output that the surface file holds
        # was not written again, so `s_bed` is not that of this year: the J
        # comparison below holds, the nodal misfit field is skipped (the
        # earlier leg wrote it).
        rewritten = row is not None and out_index <= last_written
        if row is not None and step == year0_step and year == 0.0:
            # The misfit against the observed bed at year 0. In the
            # inversion schedule J must equal the inversion's J; in the
            # protocol and fine schedules the difference measures the change
            # of the time discretisation (module docstring).
            rel = (abs(row["misfit_J"] - J_state) / abs(J_state)
                   if J_state else float("nan"))
            nodal = (output.bed_misfit() if not rewritten else
                     {"nodes": 0, "min_m": float("nan"), "max_m": float("nan"),
                      "rms_nodal_m": float("nan"),
                      "note": "written by an earlier leg"})
            checks["misfit_year0"] = {"schedule": args.schedule,
                                      "step": step,
                                      "J_forward": row["misfit_J"],
                                      "J_state": J_state,
                                      "relative_difference": rel,
                                      "rms_forward_m": row["rms_misfit_m"],
                                      "rms_state_m": rms_state}
            checks["bed_misfit_year0"] = nodal
            say(f"  MISFIT AT YEAR 0 (step {step}, schedule {args.schedule}): "
                f"J forward {row['misfit_J']:.10e}, J of the inversion "
                f"{J_state:.10e}, relative difference {rel:.3e}; RMS "
                f"{row['rms_misfit_m'] * 1e3:.4f} mm (inversion "
                f"{rms_state * 1e3 if rms_state is not None else float('nan'):.4f}"
                f" mm)")
            say(f"           bed - observed bed at the {nodal['nodes']} "
                f"surface nodes: min {nodal['min_m']:.4f} m, max "
                f"{nodal['max_m']:.4f} m, nodal RMS {nodal['rms_nodal_m']:.4f}"
                f" m")

        if not checkpointing or step == schedule[-1][0]:
            continue
        # The wall limit: stop with a checkpoint when the time left would
        # not cover three of the slowest recent steps and a checkpoint
        # (at least 2 min for the save of the checkpoint).
        elapsed_h = common.elapsed(tic_run) / 3600.0
        margin_h = (3.0 * max(recent) + max(120.0, 2.0 * checkpoint_s)) / 3600.0
        at_limit = ((args.wall_limit_hours is not None
                     and elapsed_h + margin_h > args.wall_limit_hours)
                    or step == args.stop_after_step)
        # Periodic checkpoints at the first output at or after every
        # `checkpoint_every` steps, so the checkpoint state is the state of
        # an output and a restart after a crash repeats few steps.
        # A checkpoint also at the output of year 0, so a crash after it
        # never repeats the year-0 output.
        periodic = (args.checkpoint_every > 0 and row is not None
                    and (step - last_checkpoint_step >= args.checkpoint_every
                         or year == 0.0))
        if at_limit or periodic:
            info = {"step": step, "year": year, "leg": leg,
                    "out_index": last_out_index,
                    "solve_since_output": solve_since_output,
                    "index": n_checkpoints}
            checkpoint_s = checkpoints.save(n_checkpoints, info)
            say(f"  checkpoint {n_checkpoints} after step {step} (year "
                f"{year:.0f}) in {checkpoint_s:.1f} s"
                f"{' at the wall limit' if at_limit else ''}")
            n_checkpoints += 1
            last_checkpoint_step = step
            if at_limit:
                stopped = {"step": step, "year": year,
                           "elapsed_h": elapsed_h}
                break
    march_s = time.time() - tic_march
    params.close()
    if stopped is not None:
        # The job script finds the marker and submits the next leg with
        # --restart in the same output directory.
        if COMM_WORLD.rank == 0:
            with open(os.path.join(args.output_path, RESTART_MARKER),
                      "w") as handle:
                handle.write(f"{stopped['step']} {stopped['year']}\n")
        write_summary("checkpointed", stopped=stopped, march_s=march_s)
        say(f"  STOPPED at the wall limit after step {stopped['step']} "
            f"(year {stopped['year']:.0f}, {stopped['elapsed_h']:.2f} h); "
            f"resubmit with --restart")
        return
    if COMM_WORLD.rank == 0:
        marker = os.path.join(args.output_path, RESTART_MARKER)
        if os.path.exists(marker):
            os.remove(marker)
    summary = {"args": vars(args), "state": "finished",
               "experiment": record["experiment"],
               "bed_iteration": bed_iteration, "solver": solver_record,
               "checks": checks, "rows": rows, "fine_steps": fine_rows,
               "restored": restored, "march_s": march_s,
               "total_s": time.time() - tic_run,
               "peak_memory_gb": common.peak_memory_gb(),
               "provenance": (common.provenance(HERE)
                              if COMM_WORLD.rank == 0 else None)}
    common.write_json(os.path.join(args.output_path, SUMMARY_FILE), summary)
    say(f"  march {march_s:.1f} s, total {summary['total_s']:.1f} s, peak "
        f"memory {summary['peak_memory_gb']:.2f} GB")


if __name__ == "__main__":
    main()
