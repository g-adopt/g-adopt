r"""Shared parts of the two self-gravitating GIA benchmark drivers.

`spada.py` (Spada et al. 2011) and `martinec.py` (Martinec et al. 2018) solve
the same Earth model on the same kind of mesh with the same solver. This
module holds everything that the two drivers share:

1. The scales of the non-dimensional equations.
2. The Earth model M3-L70-V01: the density, shear modulus and viscosity of
   each layer, the closed-form reference gravity, the fluid core and the
   moments of the rotation equations.
3. The mesh: the four-region gmsh sphere, the local refinement of the
   Martinec mesh, the P2 curving and the checks of a generated mesh. The
   resolution of each mesh is fixed in `MESHES`.
4. The time steps: the elastic step at t = 0 and the graded time-step
   sequence that ends on every output epoch.
5. The solver settings, the time loop and the output helpers.

## Non-dimensional scales

    length       D = Re - Rc = 2891 km, the mantle thickness
    density      rho_bar = 5511.68 kg m^-3, the mean density of the model
    gravity      g_bar = 9.81555 m s^-2
    stress       mu_bar = 1e11 Pa
    viscosity    eta_bar = 1e21 Pa s
    time         t_bar = eta_bar / mu_bar = 1e10 s = 316.8809 yr

Two numbers couple the scaled equations. B_mu = rho_bar g_bar D / mu_bar =
1.564 is the ratio of buoyancy stress to elastic stress. Lambda =
4 pi G rho_bar D / g_bar = 1.361 is the self-gravity number: with
psi = psi_hat g_bar D, the Poisson equation nabla^2 psi = -4 pi G rho becomes
nabla_hat^2 psi_hat = -Lambda rho_hat.

The potential `psi` of the solver satisfies nabla^2 psi = -4 pi G rho, so psi
is minus the Newtonian potential and the gravitational acceleration is
g_0 = +grad(psi), pointing inward, with |g_0| = G M(<r) / r^2.

## Import rule

The Firedrake names come from `gadopt`, which re-exports the Firedrake
namespace. This is also what makes `gadopt` import before `firedrake`, which
the library requires. `gmsh` is imported only inside the functions that write
a mesh, so that a module that needs only the constants does not need it.
"""

import hashlib
import json
import os
import time

import numpy as np
from gadopt import (COMM_WORLD, CompressibleInternalVariableApproximation,
                    Constant, Function, FunctionSpace, JacobianDeterminant,
                    Mesh, SpatialCoordinate, Submesh, VectorFunctionSpace,
                    conditional, dot, log, sqrt)
from gadopt.gia_gravity import (FluidCore,
                                selfgrav_dtn_iterative_solver_parameters)
from gadopt.utility import initialise_background_field
from mpi4py import MPI

# ---------------------------------------------------------------------------
# 1. Scales
# ---------------------------------------------------------------------------

#: Newton's constant, prescribed by the benchmark, m^3 kg^-1 s^-2.
G_NEWTON = 6.6732e-11
#: The Earth radius, m.
A_EARTH = 6.371e6
#: The length scale D = Re - Rc, m.
D_SCALE = 2.891e6
#: The same length scale in km, for the mesh sizes.
D_KM = 2891.0
#: The density scale, the mean density of M3-L70-V01, kg m^-3. It is the
#: model's own mean density and not the mean density of the Earth. The TABOO
#: reference uses the same value, and the Earth value makes every Spada
#: number about 0.04 percent wrong.
RHO_BAR = 5511.68
#: The gravity scale, m s^-2.
G_BAR = 9.81555
#: The stress scale, Pa.
MU_BAR = 1.0e11
#: The time scale eta_bar / mu_bar = 1e21 Pa s / 1e11 Pa = 1e10 s, in years.
T_BAR_YR = 316.8809
#: Buoyancy stress over elastic stress, rho_bar g_bar D / mu_bar (1.564037).
B_MU = RHO_BAR * G_BAR * D_SCALE / MU_BAR
#: The self-gravity number 4 pi G rho_bar D / g_bar (1.361324).
LAMBDA = 4 * np.pi * G_NEWTON * RHO_BAR * D_SCALE / G_BAR

# ---------------------------------------------------------------------------
# 2. The Earth model M3-L70-V01
# ---------------------------------------------------------------------------

#: The core-mantle boundary radius, 3480 km / D, non-dimensional.
RC = 1.203736
#: The Earth radius, 6371 km / D, non-dimensional.
RE = 2.203736

#: The density layers of M3-L70-V01, outermost first:
#: `(r_outer_km, r_inner_km, rho_kg_m3)`. The core row runs to r = 0, so it
#: covers both the meshed inner buffer and the unmeshed ball below it.
LAYERS_KM = [(6371.0, 6301.0, 3037.0),
             (6301.0, 5951.0, 3438.0),
             (5951.0, 5701.0, 3871.0),
             (5701.0, 3480.0, 4978.0),
             (3480.0, 0.0, 10750.0)]

#: The mantle layers of M3-L70-V01, outermost first, as
#: `(r_outer, r_inner, rho / rho_bar, mu / mu_bar, eta / eta_bar)`. Radii
#: are non-dimensional. The lithosphere viscosity of 1e19 viscosity units
#: (1e40 Pa s) gives a Maxwell time of about 2e19 t_bar, so the lithosphere
#: is elastic over the whole run.
MANTLE_LAYERS = [
    (RE, 2.179523, 3037.0 / RHO_BAR, 0.50605, 1.0e19),
    (2.179523, 2.058457, 3438.0 / RHO_BAR, 0.70363, 1.0),
    (2.058457, 1.971982, 3871.0 / RHO_BAR, 1.05490, 1.0),
    (1.971982, RC, 4978.0 / RHO_BAR, 2.28340, 2.0),
]

#: The density of the inviscid homogeneous core, non-dimensional (1.950402).
RHO_CORE = 10750.0 / RHO_BAR

#: The polar moment of inertia C, non-dimensional (72.2269), in units of
#: rho_bar D^5. C enters the change of the rotation rate and not the polar
#: motion.
C_NONDIM = 8.0394e37 / (RHO_BAR * D_SCALE**5)

#: The dynamical ellipticity C - A, non-dimensional. A spherically symmetric
#: reference density gives C = A, so C - A is an input of the rotation rows.
#:
#: Spada et al. (2011) give two values that differ by 2.4 percent. Their
#: Table 2 prescribes 2.63e35 kg m^2 ("prescribed"). Their rotation
#: calculation uses the secular Love number k_s = 0.96672389, and a
#: hydrostatic figure with that k_s has C - A = 2.6952e35 kg m^2 ("ks"). The
#: solver's rotational feedback is the physical one, Q k_T(t) with
#: Q = a^5 Omega^2 / (3 G), so only "ks" makes its fluid limit agree with the
#: TABOO transfer function. The polar-motion case uses "ks". The other value
#: stays here because it is the published one, and a reader who compares
#: against Table 2 needs it.
C_MINUS_A = {"ks": 2.6952e35 / (RHO_BAR * D_SCALE**5),          # 0.24214001
             "prescribed": 2.63e35 / (RHO_BAR * D_SCALE**5)}    # 0.23628236

#: The rotation rate squared, Omega = 7.292115e-5 rad/s, in units of g_bar / D.
OMEGA_SQ = 7.292115e-5**2 * D_SCALE / G_BAR          # 1.5661757e-03


def layers_nondim():
    """The density layers as `(r_outer, r_inner, rho)`, non-dimensional.

    Returns:
      A list in the order of `LAYERS_KM`, outermost first.
    """
    return [(ro / D_KM, ri / D_KM, rho / RHO_BAR)
            for ro, ri, rho in LAYERS_KM]


def enclosed_mass_ufl(r):
    """The mass M(<r) inside radius r, non-dimensional, as UFL.

    Inside a layer with inner radius r_i and density rho, the enclosed mass
    is the mass below r_i plus (4/3) pi rho (r^3 - r_i^3), so M(<r) is a
    piecewise cubic in r. Above the outermost layer it is the total mass.

    Args:
      r: the radius, a UFL expression or a `Constant`.

    Returns:
      A UFL expression in units of rho_bar D^3.
    """
    layers = sorted(layers_nondim(), key=lambda t: t[1])  # ascending r_inner
    accumulated = 0.0
    expr = Constant(0.0)
    for r_out, r_in, rho in layers:
        below = accumulated
        accumulated += 4 / 3 * np.pi * rho * (r_out**3 - r_in**3)
        here = Constant(below) + Constant(4 / 3 * np.pi * rho) * (
            r**3 - Constant(r_in**3))
        expr = conditional(r < r_out, conditional(r >= r_in, here, expr), expr)
    return conditional(r >= layers[-1][0], Constant(accumulated), expr)


def gravity_exact_ufl(r):
    """The reference gravity |g_0|(r) of the layered density, as UFL.

    |g_0| = G M(<r) / r^2, which in the scaled variables is
    (Lambda / 4 pi) M_hat(<r) / r_hat^2. It is exact for the piecewise
    constant density, the unmeshed core included, and it reproduces the
    gravity column of Spada et al. (2011) to its printed digits.

    Args:
      r: the radius, a UFL expression or a `Constant`.

    Returns:
      A UFL expression in units of g_bar.
    """
    return (LAMBDA / (4 * np.pi)) * enclosed_mass_ufl(r) / r**2


def layered(mesh, column, name):
    """A DG0 field that takes `MANTLE_LAYERS[*][column]` inside each layer.

    `initialise_background_field` is the library's way to build a
    discontinuous radial profile. It wants the interface radii outermost
    first, one more radius than there are values. Every interface is a mesh
    surface, so each cell lies inside one layer and the DG0 field is exact.

    Args:
      mesh: the mantle mesh.
      column: 2 for the density, 3 for the shear modulus, 4 for the viscosity.
      name: the name of the field.

    Returns:
      A DG0 `Function`, non-dimensional.
    """
    radii = [row[0] for row in MANTLE_LAYERS] + [MANTLE_LAYERS[-1][1]]
    values = [row[column] for row in MANTLE_LAYERS]
    field = Function(FunctionSpace(mesh, "DG", 0), name=name)
    initialise_background_field(field, values, SpatialCoordinate(mesh), radii)
    return field


def maxwell_approximation(mesh, bulk_shear_ratio):
    """The layered Maxwell rheology of M3-L70-V01 on the mantle mesh.

    One Maxwell element per cell, written as an internal variable. The
    reference model is incompressible. The displacement element is a
    penalty-type discretisation of the volume constraint, so the model here
    is compressible, with a bulk modulus of `bulk_shear_ratio` times the shear
    modulus in every layer. K / mu = 100 gives a Poisson ratio of 0.495 and
    K / mu = 1000 gives 0.4995. The residual compressibility is one source of
    the difference to the incompressible references.

    Args:
      mesh: the mantle submesh.
      bulk_shear_ratio: K / mu, dimensionless.

    Returns:
      A `CompressibleInternalVariableApproximation`.
    """
    rho = layered(mesh, 2, "density")
    mu = layered(mesh, 3, "shear_modulus")
    eta = layered(mesh, 4, "viscosity")
    X = SpatialCoordinate(mesh)
    r = sqrt(dot(X, X))
    return CompressibleInternalVariableApproximation(
        bulk_modulus=mu, density=rho, shear_modulus=[mu], viscosity=[eta],
        bulk_shear_ratio=bulk_shear_ratio, g=gravity_exact_ufl(r),
        B_mu=B_MU, self_gravity_number=LAMBDA)


def fluid_core():
    """The inviscid core, eliminated onto the core-mantle boundary.

    `FluidCore` adds a buoyancy spring on Rc, a mass sheet in the Poisson
    equation and one `Real` pressure multiplier that keeps the core volume
    fixed. The gravity it needs is the reference gravity at Rc.

    Returns:
      A `FluidCore`.
    """
    return FluidCore(boundary=SURF_RC, rho_core=RHO_CORE,
                     g=gravity_exact_ufl(Constant(RC)))


# ---------------------------------------------------------------------------
# 3. The mesh
# ---------------------------------------------------------------------------
#
# The gravitational potential lives on the whole mesh (the "parent"). The
# mechanics lives on a `Submesh` of the mantle cells. The two outer regions
# are buffers that carry the potential away from its sources, so that the
# exterior gravity condition (a Dirichlet-to-Neumann map, DtN) sits at a
# distance from the load and from the core-mantle boundary:
#
#   0.5 Rc --- inner (102) --- Rc --- mantle (101) --- Re --- buffer (103) --- 2 Re
#   surface 5               surface 3              surface 2               surface 4
#   interior DtN            interior facet         interior facet          exterior DtN
#
# The three density interfaces (6301, 5951 and 5701 km) are also mesh
# surfaces, tags 6, 7 and 8, so a DG0 density is exact on every cell. The
# core ball r < 0.5 Rc is not meshed.

#: The inner DtN sphere, at half the core radius.
R_INNER = 0.5 * RC
#: The outer DtN sphere, at twice the Earth radius.
R_OUTER = 2.0 * RE
#: The density interfaces strictly inside the mantle, ascending: 5701, 5951
#: and 6301 km divided by D.
DENSITY_INTERFACES = (1.971982, 2.058457, 2.179523)
#: Extra spheres in the outer buffer, so that the size field has somewhere
#: to grade.
BUFFER_SPHERES = (2.75, 3.30)
#: The base of the 70 km lithosphere.
R_LITHO = DENSITY_INTERFACES[-1]

#: The cell groups.
CELL_MANTLE, CELL_INNER, CELL_BUFFER = 101, 102, 103
#: The surface groups.
SURF_RE, SURF_RC, SURF_OUTER, SURF_INNER = 2, 3, 4, 5
#: The density interfaces as interior facet groups. The physics does not need
#: them, because the divergence-form Poisson source makes the interface mass
#: sheets by itself once the mesh conforms to the jumps. They exist so that a
#: surface average can be taken on them.
DENSITY_INTERFACE_TAGS = dict(zip(DENSITY_INTERFACES, (6, 7, 8)))

#: The cap centre of the Martinec ice models L1 and L2, and the two basin
#: centres, as (colatitude, longitude) in degrees. The Martinec mesh is
#: refined around them.
MARTINEC_CAP_CENTRE = (25.0, 75.0)
MARTINEC_B1_CENTRE = (100.0, 320.0)
MARTINEC_B2_CENTRE = (35.0, 25.0)
#: The angular radius of the initial coastline of both basins, in degrees:
#: the zero of zeta0 = bmax - b0 exp(-psi^2 / (2 sigma_b^2)).
MARTINEC_COAST_PSI_DEG = 24.85

#: The fixed meshes. Every length is in km.
#:
#: `h_km` is the lateral cell size in the mantle, and `litho_layers` the
#: number of cell layers across the 70 km lithosphere. `min_cells` is the
#: smallest number of cells around a great circle of any mesh sphere; it sets
#: the resolution of the small inner DtN sphere, where the grading would
#: otherwise make the cells coarse. `grade` is the growth of the cell size
#: away from the mantle.
#:
#: "spada": 250 km lateral size with two 35 km lithosphere layers, so the
#: lithosphere cells are about 7 times wider than tall. The algebraic
#: multigrid on the displacement converges much more slowly on more
#: anisotropic cells.
#:
#: "martinec": the same base mesh, refined to 78 km in the ice cap (to 12
#: degrees, the cap radius is 10) and in a band of +-4 degrees around each of
#: the two initial coastlines. `refinement` holds the parameters of
#: `martinec_size_fields`.
#:
#: "smoke": a very coarse mesh for a check that the code runs, on a laptop.
#: Its numbers are not results.
#:
#: The gmsh Delaunay algorithm puts points in at random, and some seeds give
#: flat cells (see `flat_cells`). With gmsh 4.15.2 on an Apple arm64 machine
#: the default seed 1 gives 3 flat cells on "spada", 1 on "martinec" and 1 on
#: "smoke". The seeds below give none there. The same gmsh version on the x86
#: nodes of Gadi gives other tetrahedra for the same seed, with other cell
#: counts and one flat cell on "spada" and on "martinec". A seed is therefore
#: not portable between machines either.
#: `build_meshes` therefore starts at the seed below and tries the next seeds
#: until a mesh has no flat cell (`SEED_TRIES`). The job log and the summary
#: record the seed it used and the MD5 of the file.
MESHES = {
    "spada": {"h_km": 250.0, "litho_layers": 2, "min_cells": 32,
              "grade": 2.0, "refinement": None,
              "mesh_options": {"Mesh.Algorithm3D": 1, "Mesh.RandomSeed": 2}},
    "martinec": {"h_km": 250.0, "litho_layers": 2, "min_cells": 32,
                 "grade": 2.0,
                 "refinement": {"h_fine_km": 78.0, "cap_deg": 12.0,
                                "band_deg": 4.0, "grade_lateral": 0.5,
                                "depth_km": 500.0},
                 "mesh_options": {"Mesh.Algorithm3D": 1,
                                  "Mesh.RandomSeed": 3}},
    "smoke": {"h_km": 1000.0, "litho_layers": 1, "min_cells": 16,
              "grade": 2.0, "refinement": None,
              "mesh_options": {"Mesh.Algorithm3D": 1, "Mesh.RandomSeed": 2}},
}


def sphere_radii(litho_layers):
    """Every sphere of the mesh, ascending, with the lithosphere sub-spheres.

    The sub-spheres inside the lithosphere carry no physical group and no
    density jump. They only force gmsh to put nodes on them, so that each
    sub-shell is filled with one layer of flat tetrahedra at the lateral
    size. This gives the lithosphere mesh-conforming radial layers whatever
    the lateral size is, which an isotropic size field cannot give cheaply.

    Args:
      litho_layers: the number of cell layers across the lithosphere.

    Returns:
      A sorted list of non-dimensional radii.
    """
    radii = [R_INNER, RC, *DENSITY_INTERFACES, RE, *BUFFER_SPHERES, R_OUTER]
    for k in range(1, litho_layers):
        radii.append(R_LITHO + k * (RE - R_LITHO) / litho_layers)
    return sorted(radii)


def _shells(litho_layers):
    """The shells between consecutive spheres, as `(r_in, r_out, cell_tag)`."""
    radii = sphere_radii(litho_layers)
    out = []
    for r_in, r_out in zip(radii[:-1], radii[1:]):
        mid = 0.5 * (r_in + r_out)
        tag = (CELL_INNER if mid < RC
               else CELL_MANTLE if mid < RE else CELL_BUFFER)
        out.append((r_in, r_out, tag))
    return out


def _unit_vector(colatitude_deg, longitude_deg):
    """The Cartesian unit vector of a point given in geographic degrees."""
    theta, phi = np.radians(colatitude_deg), np.radians(longitude_deg)
    return np.array([np.sin(theta) * np.cos(phi),
                     np.sin(theta) * np.sin(phi), np.cos(theta)])


def _gmsh_number(value):
    """A number as a plain decimal string for a gmsh `MathEval` expression.

    Fixed-point notation with 15 decimals keeps the non-dimensional values
    (all between 1e-3 and 5 here) to about 1e-15. It never writes an
    exponent, and a negative number is put in parentheses, because the gmsh
    expression parser rejects a unary minus after a binary operator.
    """
    text = f"{float(value):.15f}"
    return f"({text})" if text.startswith("-") else text


def martinec_size_fields(h_base, grade, h_fine_km, cap_deg, band_deg,
                         grade_lateral, depth_km):
    """The local refinement of the Martinec mesh, as a gmsh size-field hook.

    For a region centre c, the angular distance of a point X from the centre
    is psi = acos(clamp(c . X / |X|, -1, 1)). The angular distance outside
    the region, d_ang, is max(0, psi - cap) for the cap and
    max(0, |psi - psi_coast| - band) for a coastline band. The lateral size
    grows linearly with the arc length outside the region,

        h_region = h_fine + grade_lateral r d_ang,

    and the refinement fades with the distance from Re over `depth`:

        t = tanh(|r - Re| / depth),
        h = (1 - t) h_region + t h_graded,

    with `h_graded` the base size field of `generate_sphere`. The fade goes
    to `h_graded` and not to the constant `h_base`, because the base grading
    makes the buffer cells much larger than `h_base`, and a fade to `h_base`
    would refine both buffers under the `Min` merge.

    The divisor |X| is floored at 1e-12, which changes no value at a mesh
    point (r >= 0.5 Rc) and keeps gmsh from aborting at the origin.

    Args:
      h_base: the base lateral size, non-dimensional.
      grade: the radial growth rate of the base size.
      h_fine_km: the lateral size inside the regions, km.
      cap_deg: the angular radius of the refined cap, degrees.
      band_deg: the half width of each coastline band, degrees.
      grade_lateral: the growth of the size per unit arc length outside a
        region, dimensionless.
      depth_km: the radial fade length, km.

    Returns:
      A callable that receives the gmsh module, adds one `MathEval` field per
      region and returns their tags.
    """
    n = _gmsh_number
    h_fine, depth = h_fine_km / D_KM, depth_km / D_KM
    regions = [(MARTINEC_CAP_CENTRE, "cap", np.radians(cap_deg), 0.0),
               (MARTINEC_B1_CENTRE, "band", np.radians(MARTINEC_COAST_PSI_DEG),
                np.radians(band_deg)),
               (MARTINEC_B2_CENTRE, "band", np.radians(MARTINEC_COAST_PSI_DEG),
                np.radians(band_deg))]
    r = "sqrt(x*x+y*y+z*z)"
    # The base size field, repeated in the same form as in `generate_sphere`
    # so that the fade reaches exactly the base size away from Re.
    h_graded = (f"({n(h_base)}*(1 + {n(grade)}*max(0, {r} - {n(RE)})"
                f" + {n(grade)}*max(0, {n(RC)} - {r})))")
    # The radial fade weight: 0 on Re, towards 1 far from it.
    fade = f"tanh(abs({r} - {n(RE)})/{n(depth)})"

    def hook(gmsh):
        tags = []
        for centre, kind, radius, half_width in regions:
            c0, c1, c2 = (n(v) for v in _unit_vector(*centre))
            # The clamp keeps `acos` finite where rounding puts the cosine
            # just above 1 on the region axis.
            psi = (f"acos(min(1, max(-1, ({c0}*x + {c1}*y + {c2}*z)"
                   f"/max({r}, {n(1e-12)}))))")
            if kind == "cap":
                d_ang = f"max(0, {psi} - {n(radius)})"
            else:
                d_ang = (f"max(0, abs({psi} - {n(radius)})"
                         f" - {n(half_width)})")
            h_region = f"({n(h_fine)} + {n(grade_lateral)}*{r}*{d_ang})"
            field = gmsh.model.mesh.field.add("MathEval")
            gmsh.model.mesh.field.setString(
                field, "F", f"(1 - {fade})*{h_region} + {fade}*{h_graded}")
            tags.append(field)
        return tags
    return hook


def generate_sphere(filename, h, litho_layers, min_cells, grade=2.0,
                    extra_size_fields=None, mesh_options=None):
    """Write the four-region sphere to a gmsh file.

    Every sphere of `sphere_radii` is an OpenCASCADE solid, and
    `occ.fragment` cuts them into nested shells that share their bounding
    surfaces exactly. The core ball inside 0.5 Rc is then removed without its
    boundary. Each entity is identified by its own measure (a surface radius
    is sqrt(A / 4 pi)), so nothing depends on the order in which gmsh returns
    entities.

    The target cell size grows linearly away from the mantle,

        lc(r) = h (1 + grade max(0, r - Re) + grade max(0, Rc - r)),

    and a second field caps it at 2 pi r / min_cells, so that every sphere
    carries at least `min_cells` cells around a great circle. The inner DtN
    sphere needs that cap: the DtN map must resolve its truncation degree.

    gmsh tetrahedra are not reproducible across gmsh versions, which is why
    `build_meshes` prints the gmsh version and the MD5 of the file.

    Args:
      filename: the output `.msh` path.
      h: the lateral cell size in the mantle, non-dimensional.
      litho_layers: the cell layers across the lithosphere.
      min_cells: the smallest number of cells around a great circle.
      grade: the growth rate of the cell size away from the mantle.
      extra_size_fields: `None`, or a callable `hook(gmsh)` that adds gmsh
        fields and returns their tags. The `Min` merge takes the smallest
        size, so these fields can only refine.
      mesh_options: `None`, or a dictionary of numeric gmsh options, set just
        before the mesh is generated.

    Returns:
      The cell count of each cell group, as a dictionary.
    """
    import gmsh

    radii = sphere_radii(litho_layers)
    gmsh.initialize()
    # A failure anywhere below must finalize gmsh. Otherwise gmsh stays
    # initialised with a dirty model, and the next call in the same process
    # returns an empty mesh.
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("selfgrav_sphere")
        occ = gmsh.model.occ
        balls = [occ.addSphere(0, 0, 0, R) for R in radii]
        occ.fragment([(3, balls[0])], [(3, b) for b in balls[1:]])
        occ.synchronize()

        def radius_of_surface(tag):
            return np.sqrt(occ.getMass(2, tag) / (4 * np.pi))

        # Remove the unmeshed core ball and keep its bounding surface, which
        # is the inner boundary of the domain and carries the interior DtN.
        core = [t for _, t in gmsh.model.getEntities(3)
                if abs(occ.getMass(3, t) - 4 / 3 * np.pi * radii[0]**3) < 1e-8]
        assert len(core) == 1, f"expected one core ball, found {core}"
        occ.remove([(3, core[0])], recursive=False)
        occ.synchronize()

        # A radius recovered from an area is not bit-identical to the radius
        # that made it, so it is snapped back to the list before it is a key.
        def snap(R):
            match = [r for r in radii if abs(r - R) < 1e-6 * r]
            assert len(match) == 1, f"radius {R} is not in {radii}"
            return match[0]

        surface_of_radius = {}
        for _, t in gmsh.model.getEntities(2):
            surface_of_radius.setdefault(snap(radius_of_surface(t)),
                                         []).append(t)
        volume_of_shell = {}
        for _, t in gmsh.model.getEntities(3):
            bounding = sorted(snap(radius_of_surface(abs(s))) for _, s
                              in gmsh.model.getBoundary([(3, t)],
                                                        oriented=True))
            volume_of_shell[(bounding[0], bounding[-1])] = t

        # The surface groups. The buffer and lithosphere sub-spheres get no
        # group: the mesh only needs nodes on them.
        tagged = [(R_INNER, SURF_INNER, "inner_dtn"), (RC, SURF_RC, "Rc"),
                  (RE, SURF_RE, "Re"), (R_OUTER, SURF_OUTER, "outer_dtn")]
        tagged += [(R, tag, f"rho_interface_{i + 1}") for i, (R, tag)
                   in enumerate(DENSITY_INTERFACE_TAGS.items())]
        for R, tag, name in tagged:
            gmsh.model.addPhysicalGroup(2, surface_of_radius[R], tag,
                                        name=name)
        # The cell groups: every mantle sub-shell joins 101.
        layout = _shells(litho_layers)
        for tag, name in [(CELL_MANTLE, "mantle"), (CELL_INNER, "inner"),
                          (CELL_BUFFER, "buffer")]:
            members = [volume_of_shell[(r_in, r_out)]
                       for r_in, r_out, t in layout if t == tag]
            gmsh.model.addPhysicalGroup(3, members, tag, name=name)

        # The graded size field and the angular cap, merged by `Min`.
        r = "sqrt(x*x+y*y+z*z)"
        graded = gmsh.model.mesh.field.add("MathEval")
        gmsh.model.mesh.field.setString(
            graded, "F", f"{h}*(1 + {grade}*max(0, {r} - {RE})"
                         f" + {grade}*max(0, {RC} - {r}))")
        fields = [graded]
        capped = gmsh.model.mesh.field.add("MathEval")
        gmsh.model.mesh.field.setString(
            capped, "F", f"{2 * np.pi / min_cells}*{r}")
        fields.append(capped)
        if extra_size_fields is not None:
            fields.extend(extra_size_fields(gmsh))
        merged = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(merged, "FieldsList", fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(merged)
        # Without these three options, gmsh's own curvature and point rules
        # override the size field near the small inner sphere and refine it
        # by an order of magnitude.
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        for key, value in (mesh_options or {}).items():
            gmsh.option.setNumber(key, value)
        gmsh.model.mesh.generate(3)

        cells = {}
        for tag in (CELL_MANTLE, CELL_INNER, CELL_BUFFER):
            count = 0
            for ent in gmsh.model.getEntitiesForPhysicalGroup(3, tag):
                _, etags, _ = gmsh.model.mesh.getElements(3, ent)
                count += sum(len(e) for e in etags)
            cells[tag] = count
        gmsh.write(filename)
    finally:
        gmsh.finalize()
    return cells


def flat_cells(filename):
    """The number of tetrahedra with all four vertices on one mesh sphere.

    `curve_mesh` moves every P2 node radially onto the linear interpolant of
    the vertex radii. A tetrahedron with all four vertices on one sphere (a
    "flat cell") gets all ten P2 nodes on that sphere. It has almost no
    volume, and after the curving its Jacobian determinant can change sign
    inside the cell, which gives a wrong answer without an error message.
    The Delaunay algorithm of gmsh makes such cells at random; the random
    seed moves or removes them.

    The test is the spread of the four vertex radii against their mean, with
    a relative tolerance of 1e-6. The smallest gap between two mesh spheres
    is one lithosphere layer, 35 km or 0.0121 D at two layers, so a real cell
    never passes the test, and a flat cell passes with a spread near 1e-15.

    Args:
      filename: the `.msh` file.

    Returns:
      The number of flat cells.
    """
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(filename)
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        xyz = np.zeros((int(node_tags.max()) + 1, 3))
        xyz[node_tags.astype(int)] = coords.reshape(-1, 3)
        radius = np.linalg.norm(xyz, axis=1)
        count = 0
        for tag in (CELL_MANTLE, CELL_INNER, CELL_BUFFER):
            for ent in gmsh.model.getEntitiesForPhysicalGroup(3, tag):
                _, _, element_nodes = gmsh.model.mesh.getElements(3, ent)
                for nodes in element_nodes:
                    r4 = radius[nodes.reshape(-1, 4).astype(int)]
                    count += int(np.count_nonzero(
                        np.ptp(r4, axis=1) < 1e-6 * r4.mean(axis=1)))
    finally:
        gmsh.finalize()
    return count


def curve_mesh(linear_mesh, name=None):
    """Remap a straight-sided mesh to P2 coordinates, so the spheres are curved.

    Each P2 node is pushed radially onto the radius that the linear
    interpolant of the vertex radii takes there, X_p2 = (r_p1 / r) X. An
    edge whose two vertices lie on a sphere of radius R has r_p1 = R along
    its length, so its midpoint moves onto that sphere and the edge becomes a
    quadratic arc on it. This reduces the geometric error of the surface
    integrals on Re, Rc and the DtN spheres from O(h^2) to O(h^3).

    Args:
      linear_mesh: a Firedrake mesh with P1 coordinates.
      name: the name of the new mesh.

    Returns:
      A new Firedrake mesh with P2 coordinates.
    """
    X = SpatialCoordinate(linear_mesh)
    r = sqrt(dot(X, X))
    r_p1 = Function(FunctionSpace(linear_mesh, "CG", 1)).interpolate(r)
    X_p2 = Function(VectorFunctionSpace(linear_mesh, "CG", 2)).interpolate(
        (r_p1 / r) * X)
    return Mesh(X_p2) if name is None else Mesh(X_p2, name=name)


def folded_cells(mesh):
    """The number of curved cells whose Jacobian determinant is not of one sign.

    On a P2 tetrahedron the Jacobian determinant is a cubic polynomial on the
    reference cell, so its interpolant into DG3 reproduces it exactly, and
    the 20 DG3 nodes of a cell are 20 exact samples of it. A valid cell has
    the same nonzero sign at every sample. The sign itself can be either one,
    because it follows the vertex order that the mesh reader chose. A folded
    cell has samples of both signs, or a zero.

    The samples are not a proof: a cubic can change sign between two samples
    of one sign. The flat cells of `flat_cells`, which are the cause of the
    folds seen so far, give samples of both signs.

    Collective: every rank must call this.

    Args:
      mesh: a mesh with P2 coordinates.

    Returns:
      The number of folded cells over all ranks.
    """
    space = FunctionSpace(mesh, "DG", 3)
    det = Function(space).interpolate(JacobianDeterminant(mesh))
    samples = det.dat.data_ro[space.cell_node_map().values]
    local = int(np.count_nonzero(
        (samples.min(axis=1) * samples.max(axis=1)) <= 0.0))
    return mesh.comm.allreduce(local, op=MPI.SUM)


#: The number of consecutive gmsh seeds that `build_meshes` tries, from the
#: seed of `MESHES`, before it stops. In the trials so far, 5 of 9
#: mesh-and-machine pairs gave a flat cell with their first seed. If the
#: seeds are independent, ten tries leave a chance of about 1e-3 that no seed
#: works. One try takes 17 s ("spada") to 22 s ("martinec") on Gadi.
SEED_TRIES = 10


def md5sum(filename):
    """The MD5 of a file, as a hexadecimal string."""
    digest = hashlib.md5()
    with open(filename, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_meshes(kind, directory, stem):
    """Generate the mesh of `kind` in the job, check it, and curve it.

    Rank 0 writes the gmsh file and counts the flat cells while the other
    ranks wait. If the mesh has a flat cell, rank 0 writes it again with the
    next gmsh seed, up to `SEED_TRIES` seeds. The tetrahedra of one seed
    differ between machines, so a fixed seed does not carry. Every rank then
    reads the file, the mantle submesh is cut from the curved parent, and
    both meshes are checked for folded cells. `Submesh` does not inherit the
    P2 coordinates of the parent, so the submesh is curved again with the
    same radial map. The two coordinate fields then agree on every shared
    facet.

    The mesh is generated in every run and not copied, so the job records
    what it ran on: the gmsh version, the MD5 of the file and the cell
    counts. Another gmsh version gives other tetrahedra, so two runs with
    different MD5s are runs on different meshes.

    Args:
      kind: a key of `MESHES`.
      directory: where the `.msh` file goes.
      stem: the name of the `.msh` file without its extension. Each case
        needs its own name, because the cases of one benchmark can run at
        the same time in one directory, and one job must not read a file
        that another job is writing.

    Returns:
      `(parent, mantle, info)`: the two curved meshes, and a dictionary with
      the mesh parameters, the gmsh version, the seed used, the flat-cell
      count of every seed tried, the MD5 and the cell counts.

    Raises:
      RuntimeError: if every seed tried gives a flat cell, or the curved
        mesh has a folded cell.
    """
    spec = MESHES[kind]
    filename = os.path.join(directory, f"{stem}.msh")
    h = spec["h_km"] / D_KM
    info = {"kind": kind, "file": filename, **{k: v for k, v in spec.items()}}
    error = None
    if COMM_WORLD.rank == 0:
        import gmsh

        tic = time.time()
        hook = None
        if spec["refinement"] is not None:
            hook = martinec_size_fields(h, spec["grade"],
                                        **spec["refinement"])
        # An error on rank 0 must reach the other ranks, or they wait in
        # the broadcast below for ever.
        try:
            first_seed = spec["mesh_options"]["Mesh.RandomSeed"]
            flat_by_seed = {}
            for seed in range(first_seed, first_seed + SEED_TRIES):
                options = {**spec["mesh_options"], "Mesh.RandomSeed": seed}
                cells = generate_sphere(filename, h, spec["litho_layers"],
                                        spec["min_cells"],
                                        grade=spec["grade"],
                                        extra_size_fields=hook,
                                        mesh_options=options)
                flat_by_seed[seed] = flat_cells(filename)
                if flat_by_seed[seed] == 0:
                    break
            info.update({"gmsh_version": gmsh.__version__,
                         "random_seed": seed,
                         "flat_cells_by_seed": flat_by_seed,
                         "md5": md5sum(filename),
                         "cells_mantle": cells[CELL_MANTLE],
                         "cells_inner": cells[CELL_INNER],
                         "cells_buffer": cells[CELL_BUFFER],
                         "flat_cells": flat_by_seed[seed],
                         "generation_s": time.time() - tic})
        except Exception as exc:  # noqa: BLE001  (re-raised on every rank)
            error = f"mesh generation failed on rank 0: {exc!r}"
    info, error = COMM_WORLD.bcast((info, error), root=0)
    if error is not None:
        raise RuntimeError(error)
    log(f"  mesh {kind}: {filename}, gmsh {info['gmsh_version']}, "
        f"seed {info['random_seed']} (flat cells by seed "
        f"{info['flat_cells_by_seed']}), "
        f"md5 {info['md5']}, cells mantle {info['cells_mantle']}, inner "
        f"{info['cells_inner']}, buffer {info['cells_buffer']} "
        f"({info['generation_s']:.1f} s)")
    if info["flat_cells"]:
        raise RuntimeError(
            f"{filename} has flat cells with every one of the {SEED_TRIES} "
            f"seeds tried ({info['flat_cells_by_seed']}). Flat cells fold "
            "when the mesh is curved. Change Mesh.RandomSeed of this mesh in "
            "selfgrav_common.MESHES or raise SEED_TRIES.")

    parent = curve_mesh(Mesh(filename), name=f"{kind}_parent")
    # The geometry is a sphere, so the "vertical" direction of the G-ADOPT
    # terms is radial and not the last Cartesian axis.
    parent.cartesian = False
    mantle = curve_mesh(Submesh(parent, 3, CELL_MANTLE),
                        name=f"{kind}_mantle")
    mantle.cartesian = False
    info["folded_cells_parent"] = folded_cells(parent)
    info["folded_cells_mantle"] = folded_cells(mantle)
    if info["folded_cells_parent"] or info["folded_cells_mantle"]:
        raise RuntimeError(
            f"{filename}: {info['folded_cells_parent']} parent cells and "
            f"{info['folded_cells_mantle']} mantle cells have a Jacobian "
            "determinant that is not of one sign after the curving. Change "
            "Mesh.RandomSeed of this mesh in selfgrav_common.MESHES.")
    log("  no flat cell, no folded cell after the curving")
    return parent, mantle, info


# ---------------------------------------------------------------------------
# 4. Time steps
# ---------------------------------------------------------------------------

#: The time step of the elastic solve at t = 0, in Maxwell times (about
#: 0.03 yr). The shortest Maxwell time of the mantle is 0.3 kyr, so this step
#: gives no measurable relaxation.
DT_ELASTIC = 1.0e-4

#: The graded time-step sequence of a load that is switched on at t = 0, as
#: `(t_end_yr, dt_yr)` segments. At time t the relaxation modes that still
#: change the answer have Maxwell times of order t, so a backward-Euler step
#: can grow with t. The rule dt <= 2 e t gives a relative time error of about
#: e: this sequence gives 2.5 percent at 1 and 2 kyr, 1 percent at 5 kyr,
#: 0.5 percent at 10 kyr and 1.25 percent at 20 kyr.
STEP_LOAD_LADDER_YR = ((100.0, 10.0), (1000.0, 50.0), (10000.0, 100.0),
                       (20000.0, 500.0))


def truncated_ladder(ladder, t_end_yr):
    """A time-step sequence cut at `t_end_yr`.

    Args:
      ladder: `(t_end_yr, dt_yr)` segments.
      t_end_yr: the end of the run, in years.

    Returns:
      The segments up to `t_end_yr`, the last one shortened to end there.
    """
    out = []
    for upto, dt in ladder:
        if upto >= t_end_yr:
            out.append((t_end_yr, dt))
            break
        out.append((upto, dt))
    return tuple(out)


def time_segments(epochs_kyr, ladder):
    """Segments `(t0_yr, t1_yr, dt_yr, nsteps, is_epoch)` that end on each epoch.

    Each ladder segment is cut at every output epoch inside it, and the step
    of each piece is adjusted so that a whole number of steps lands exactly on
    the cut. Every reported state is therefore a solved state, with no
    interpolation in time. Epoch 0 is not a step: its state is the elastic
    solve or the unloaded start.

    Args:
      epochs_kyr: the output epochs, kyr.
      ladder: `(t_end_yr, dt_yr)` segments; the run ends at the last one.

    Returns:
      A list of segments.
    """
    wanted = sorted(e * 1000.0 for e in epochs_kyr if e > 0)
    out, t0 = [], 0.0
    for t_end, dt in ladder:
        if t_end <= t0:
            continue
        marks = [w for w in wanted if t0 < w <= t_end]
        for cut in sorted(set(marks + [t_end])):
            n = max(1, int(round((cut - t0) / dt)))
            out.append((t0, cut, (cut - t0) / n, n, cut in marks))
            t0 = cut
    return out


# ---------------------------------------------------------------------------
# 5. Solver settings, time loop and output
# ---------------------------------------------------------------------------

#: The polynomial degrees: CG3 displacement, DG2 internal variables and CG2
#: potential. CG3 on the displacement is what keeps the penalty-type
#: volume constraint from locking at K / mu = 100.
DISPLACEMENT_DEGREE = 3
INTERNAL_VARIABLE_DEGREE = 2
POTENTIAL_DEGREE = 2

#: The degree L of `SphericalDtN` on both truncation spheres. Both spheres
#: are a factor of two away from the sources, so a degree-l field falls by
#: 2^-(l+1) before it reaches them, and the truncation error decreases
#: quickly with L.
DTN_DEGREE = 5

#: The representation of the exterior DtN condition. The low-rank
#: representation adds no `Real` rows; the multiplier one adds one row per
#: harmonic mode and costs one block-0 solve per row.
DTN_REPRESENTATION = "lowrank"

#: The relative tolerance of the outer FGMRES and of the mechanics-potential
#: block solve inside it.
OUTER_RTOL = 1.0e-6
BLOCK0_RTOL = 1.0e-4


def solver_parameters(layout, snes_type):
    """The solver settings of both benchmarks.

    The library preset `selfgrav_dtn_iterative_solver_parameters` with its
    own defaults, except the two tolerances. The preset chooses the treatment
    of the `Real` block from its width, `n_real`: the core pressure, and in
    Martinec also the three centre-of-mass rows and the sea-level `Shift`. At
    these widths it forms the exact Schur complement of that block once and
    caches it (`gadopt.DtNTwoBlockSchurPC` under `schur_fact_type full`).

    Args:
      layout: the `GIASpaceLayout` of the mixed space.
      snes_type: `"ksponly"` when the residual is linear in the unknowns (one
        linear solve is exact to the outer tolerance), `"newtonls"` when it is
        not (moving coastline).

    Returns:
      The PETSc options dictionary.
    """
    return selfgrav_dtn_iterative_solver_parameters(
        condensed=layout.condensed, outer_rtol=OUTER_RTOL,
        block0_rtol=BLOCK0_RTOL, snes_type=snes_type,
        dtn_representation=DTN_REPRESENTATION,
        n_real=len(layout.real_fields))


def march(solver, dt, segments, *, elastic, on_step, on_epoch,
          before_step=None, max_steps=None):
    """Backward-Euler time loop over `segments`, with callbacks.

    The time step is one live `Constant`. The forms read its value, and the
    preconditioners rebuild the operators they cache when the value changes,
    so a new segment needs one `assign` and nothing else.

    With `elastic`, the loop first solves the state at t = 0 with the step
    `DT_ELASTIC`: for a load switched on at t = 0 and then held, that state
    is the instantaneous elastic response, the limit dt -> 0 of one step.
    The march then starts again from rest. With the load held from t = 0,
    the first marched step reproduces the elastic response by itself, so a
    march from the elastic state would count it twice. `solution_old` is
    reset together with `solution`, because backward Euler reads the
    internal variables of the previous step from `solution_old`.

    Args:
      solver: the `SelfGravitatingGIASolver`.
      dt: the time step `Constant`, in Maxwell times.
      segments: the list from `time_segments`.
      elastic: whether to solve the elastic state at t = 0 first.
      on_step: `on_step(t_kyr, dt_yr, step, wall_s)`, after every solve.
      on_epoch: `on_epoch(t_kyr)`, at every output epoch.
      before_step: `None`, or `before_step(t_kyr)`, before every solve, with
        the time at the end of the step. A time-dependent load reads it there.
      max_steps: `None`, or the number of marched steps after which the loop
        reports the last state as an epoch and stops.

    Returns:
      The number of marched steps.
    """
    step = 0
    if elastic:
        log(f"\n  t = 0: the elastic response, one step of {DT_ELASTIC:g} "
            "Maxwell times")
        dt.assign(DT_ELASTIC)
        if before_step is not None:
            before_step(0.0)
        tic = time.time()
        solver.solve()
        on_step(0.0, DT_ELASTIC * T_BAR_YR, step, time.time() - tic)
        on_epoch(0.0)
        solver.solution.assign(0.0)
        solver.solution_old.assign(0.0)

    previous_dt_yr = None
    for t0, t1, dt_yr, nsteps, is_epoch in segments:
        if dt_yr != previous_dt_yr:
            dt.assign(dt_yr / T_BAR_YR)
            previous_dt_yr = dt_yr
        log(f"\n  {t0 / 1000:7.3f} -> {t1 / 1000:7.3f} kyr   dt "
            f"{dt_yr:7.2f} yr ({float(dt):.6g} Maxwell times)   "
            f"{nsteps:4d} steps")
        for k in range(nsteps):
            # Backward Euler: a time-dependent load is read at the end of
            # the step.
            t_kyr = (t0 + (k + 1) * dt_yr) / 1000.0
            if before_step is not None:
                before_step(t_kyr)
            step += 1
            tic = time.time()
            solver.solve()
            on_step(t_kyr, dt_yr, step, time.time() - tic)
            if max_steps is not None and step >= max_steps:
                log(f"\n  stopping after {max_steps} steps (smoke run)")
                on_epoch(t_kyr)
                return step
        if is_epoch:
            on_epoch(t1 / 1000.0)
    return step


def iteration_counts(solver):
    """The Newton and outer Krylov iteration counts of the last solve."""
    snes = solver.solver.snes
    return int(snes.getIterationNumber()), int(snes.ksp.getIterationNumber())


def write_json(path, data):
    """Write `data` as indented JSON on rank 0.

    NumPy scalars and arrays are converted to plain Python numbers and lists,
    so that the file can be read without NumPy.
    """
    def plain(value):
        if isinstance(value, dict):
            return {str(k): plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, np.ndarray)):
            return [plain(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    if COMM_WORLD.rank == 0:
        with open(path, "w") as handle:
            json.dump(plain(data), handle, indent=1)
    log(f"  summary written: {path}")
