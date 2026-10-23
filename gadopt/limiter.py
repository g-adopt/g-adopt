r"""This module provides a class that implements limiting of functions defined on
discontinuous function spaces. Users instantiate the class by providing relevant
parameters and call the `apply` method to update the function.

"""

from firedrake import VertexBasedLimiter, FunctionSpace, TrialFunction, LinearSolver, TestFunction, dx, assemble
from firedrake import max_value, min_value, Function, pack
from firedrake import TensorProductElement, VectorElement, HDivElement, MixedElement, EnrichedElement, FiniteElement
from firedrake.functionspaceimpl import WithGeometry
import numpy as np
from firedrake.petsc import PETSc
import pyop3 as op3
from typing import Optional
from firedrake.adjoint import annotate_tape

__all__ = ["VertexBasedP1DGLimiter"]


def assert_function_space(
    fs: WithGeometry, family: str | list[str], degree: int
):
    """Checks the family and degree of the function space.

    If the function space lies on an extruded mesh, checks both spaces of the outer
    product.

    Arguments:
      fs: UFL function space
      family: Name or names of the expected element families
      degree: Expected polynomial degree of the function space

    Raises:
      AssertionError: The family and/or degree of the function
                      space don't match the expected values

    """
    fam_list = family
    if not isinstance(family, list):
        fam_list = [family]

    ufl_elem = fs.ufl_element()
    if isinstance(ufl_elem, VectorElement):
        ufl_elem = ufl_elem.sub_elements[0]

    if isinstance(ufl_elem, TensorProductElement):  # extruded mesh
        A, B = ufl_elem.factor_elements
        assert A.family() in fam_list, 'horizontal space must be one of {0:s}'.format(fam_list)
        assert B.family() in fam_list, 'vertical space must be {0:s}'.format(fam_list)
        assert A.degree() == degree, 'degree of horizontal space must be {0:d}'.format(degree)
        assert B.degree() == degree, 'degree of vertical space must be {0:d}'.format(degree)
        assert A.variant() == 'equispaced', 'variant of horizontal space must be equispaced'
        assert B.variant() == 'equispaced', 'variant of vertical space must be equispaced'
    else:  # assume 2D mesh
        assert ufl_elem.family() in fam_list, 'function space must be one of {0:s}'.format(fam_list)
        assert ufl_elem.degree() == degree, 'degree of function space must be {0:d}'.format(degree)
        assert ufl_elem.variant() == 'equispaced', 'variant of function space must be equispaced'


def get_extruded_base_element(ufl_element: FiniteElement) -> FiniteElement:
    """Gets the base element from an extruded element.

    In case of a non-extruded mesh, returns the element itself.

    Arguments:
      ufl_element: UFL element from which to extract the base element

    Returns:
      The base element, or the provided element for a non-extruded mesh.
    """
    if isinstance(ufl_element, HDivElement):
        ufl_element = ufl_element._element
    if isinstance(ufl_element, MixedElement):
        ufl_element = ufl_element.sub_elements[0]
    if isinstance(ufl_element, VectorElement):
        ufl_element = ufl_element.sub_elements[0]  # take the first component
    if isinstance(ufl_element, EnrichedElement):
        ufl_element = ufl_element._elements[0]
    return ufl_element


def get_facet_mask(
    function_space: WithGeometry, facet: str = 'bottom'
) -> np.ndarray:
    """The meaning of top/bottom depends on the extrusion's direction. Here, we assume
    that the mesh has been extruded upwards (along the positive z axis).

    Arguments:
      function_space: UFL function space
      facet: String specifying the facet ("bottom" or "top")

    Returns:
      The top/bottom nodes of extruded 3D elements.

    Raises:
      AssertionError: The function space is not defined on an extruded mesh

    """

    # get base element
    elem = get_extruded_base_element(function_space.ufl_element())
    assert isinstance(elem, TensorProductElement), \
        f'function space must be defined on an extruded 3D mesh: {elem}'
    # figure out number of nodes in sub elements
    h_elt, v_elt = function_space.finat_element.factors
    nb_nodes_h = h_elt.space_dimension()
    nb_nodes_v = v_elt.space_dimension()
    # compute top/bottom facet indices
    # extruded dimension is the inner loop in index
    # on interval elements, the end points are the first two dofs
    offset = 0 if facet == 'bottom' else 1
    indices = np.arange(nb_nodes_h)*nb_nodes_v + offset
    return indices


class VertexBasedP1DGLimiter(VertexBasedLimiter):
    """Vertex-based limiter for P1DG tracer fields (Kuzmin, 2010).

    Kuzmin, D. (2010). A vertex-based hierarchical slope limiter for p-adaptive
    discontinuous Galerkin methods. Journal of computational and applied mathematics,
    233(12), 3077-3085.

    Arguments:
      p1dg_space: UFL P1DG function space
      clip_min: Minimal threshold to apply
      clip_max: Maximal threshold to apply

    """
    def __init__(
        self,
        p1dg_space: WithGeometry,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None,
    ):
        assert_function_space(p1dg_space, ['Discontinuous Lagrange', 'DQ'], 1)

        self.is_vector = p1dg_space.value_size > 1
        if self.is_vector:
            # Component values are copied dof by dof, so the scalar space must share the node layout
            p1dg_scalar_space = FunctionSpace(p1dg_space.mesh(), p1dg_space.ufl_element().sub_elements[0])
            super(VertexBasedP1DGLimiter, self).__init__(p1dg_scalar_space)
        else:
            super(VertexBasedP1DGLimiter, self).__init__(p1dg_space)

        self.mesh = self.P0.mesh()
        self.dim = self.mesh.geometric_dimension
        self.extruded = hasattr(self.mesh.ufl_cell(), 'sub_cells')

        assert not self.extruded or all(e.variant() == 'equispaced' for e in p1dg_space.ufl_element().sub_elements), \
            "Extruded function space must be equivariant"

        self.clip_min = clip_min
        self.clip_max = clip_max

    def _construct_centroid_solver(self) -> LinearSolver:
        """Constructs a linear problem to compute centroids.

        Executes as part of the call to the parent `__init__` special method.

        Returns:
          Firedrake linear solver.

        """
        u = TrialFunction(self.P0)
        v = TestFunction(self.P0)
        self.a_form = u * v * dx
        a = assemble(self.a_form)
        return LinearSolver(a, solver_parameters={'ksp_type': 'preonly',
                                                  'pc_type': 'bjacobi',
                                                  'sub_pc_type': 'ilu'})

    def _update_centroids(self, field: Function):
        """Updates centroid values.

        Calls the linear solver to calculate centroids.

        Executes as part of the call to the parent `compute_bounds` method.

        Arguments:
          field: Firedrake function onto which the limiter is applied

        """
        b = assemble(TestFunction(self.P0) * field * dx)
        self.centroid_solver.solve(self.centroids, b)

    def compute_bounds(self, field: Function):
        """Re-computes min/max values of all neighbouring centroids.

        Arguments:
          field: Firedrake function onto which the limiter is applied

        """
        # Call general-purpose bound computation.
        super(VertexBasedP1DGLimiter, self).compute_bounds(field)

        # Add the average of lateral boundary facets to min/max fields
        # NOTE this just computes the arithmetic mean of nodal values on the facet,
        # which in general is not equivalent to the mean of the field over the bnd facet.
        # This is OK for P1DG triangles, but not exact for the extruded case (quad facets)
        from finat.finiteelementbase import entity_support_dofs

        if self.extruded:
            entity_dim = (self.dim-2, 1)  # get vertical facets
        else:
            entity_dim = self.dim-1
        boundary_dofs = entity_support_dofs(self.P1DG.finat_element, entity_dim)
        local_facet_nodes = np.array([boundary_dofs[e] for e in sorted(boundary_dofs.keys())])
        n_bnd_nodes = local_facet_nodes.shape[1]
        local_facet_idx = op3.Dat.from_array(local_facet_nodes.ravel(), name='local_facet_idx', constant=True, rank_equal=True)
        if self.extruded:
            facet_number = self.mesh.exterior_facet_vert_local_facet_indices
        else:
            facet_number = self.mesh.exterior_facet_local_facet_indices
        code = """
                double face_mean = 0.0;
                for (int i = 0; i < %(nnodes)d; i++) {
                    unsigned int idx = local_facet_idx[facet[0]*%(nnodes)d + i];
                    face_mean += field[idx];
                }
                face_mean /= %(nnodes)d;
                for (int i = 0; i < %(nnodes)d; i++) {
                    unsigned int idx = local_facet_idx[facet[0]*%(nnodes)d + i];
                    qmax[idx] = fmax(qmax[idx], face_mean);
                    qmin[idx] = fmin(qmin[idx], face_mean);
                }"""
        bnd_kernel = op3.Function.from_c_string(
            "my_kernel",
            code % {'nnodes': n_bnd_nodes},
            [
                ("qmax", op3.ScalarType, op3.MAX_RW),
                ("qmin", op3.ScalarType, op3.MIN_RW),
                ("field", op3.ScalarType, op3.READ),
                ("facet", facet_number.dtype, op3.READ),
                ("local_facet_idx", local_facet_idx.dtype, op3.READ),
            ],
        )
        op3.loop(
            f := self.mesh.iter("exterior_facet_vert" if self.extruded else "exterior_facet"),
            bnd_kernel(
                pack(self.max_field, f),
                pack(self.min_field, f),
                pack(field, f),
                facet_number[f],
                local_facet_idx,
            ),
            eager=True,
        )
        if self.extruded:
            # Add nodal values from surface/bottom boundaries
            # NOTE calling firedrake par_loop with measure=ds_t raises an error
            bottom_nodes = get_facet_mask(self.P1CG, 'bottom').astype(op3.dtypes.IntType)
            top_nodes = get_facet_mask(self.P1CG, 'top').astype(op3.dtypes.IntType)
            bottom_idx = op3.Dat.from_array(bottom_nodes, name='node_idx', constant=True, rank_equal=True)
            top_idx = op3.Dat.from_array(top_nodes, name='node_idx', constant=True, rank_equal=True)
            code = """
                    double face_mean = 0;
                    for (int i=0; i<%(nnodes)d; i++) {
                        face_mean += field[idx[i]];
                    }
                    face_mean /= %(nnodes)d;
                    for (int i=0; i<%(nnodes)d; i++) {
                        qmax[idx[i]] = fmax(qmax[idx[i]], face_mean);
                        qmin[idx[i]] = fmin(qmin[idx[i]], face_mean);
                    }"""
            kernel = op3.Function.from_c_string(
                "my_kernel",
                code % {'nnodes': len(bottom_nodes)},
                [
                    ("qmax", op3.ScalarType, op3.MAX_RW),
                    ("qmin", op3.ScalarType, op3.MIN_RW),
                    ("field", op3.ScalarType, op3.READ),
                    ("idx", bottom_idx.dtype, op3.READ),
                ],
            )

            op3.loop(
                f := self.mesh.iter("exterior_facet_bottom"),
                kernel(
                    pack(self.max_field, f),
                    pack(self.min_field, f),
                    pack(field, f),
                    bottom_idx,
                ),
                eager=True,
            )

            op3.loop(
                f := self.mesh.iter("exterior_facet_top"),
                kernel(
                    pack(self.max_field, f),
                    pack(self.min_field, f),
                    pack(field, f),
                    top_idx,
                ),
                eager=True,
            )

        if self.clip_min is not None:
            self.min_field.interpolate(max_value(self.min_field, self.clip_min))
        if self.clip_max is not None:
            self.max_field.interpolate(min_value(self.max_field, self.clip_max))

    def apply(self, field):
        """Applies the limiter on the given field (in place).

        Args:
          field: Firedrake function onto which the limiter is applied

        """
        if annotate_tape():
            raise RuntimeError('Limiting is currently not compatible for adjoint modelling.')
        with PETSc.Log.Stage("limiter"):

            if self.is_vector:
                tmp_func = self.P1DG.get_work_function()
                fs = field.function_space()
                for i in range(fs.value_size):
                    tmp_func.dat.data_wo_with_halos[:] = field.dat.data_ro_with_halos[:, i]
                    super(VertexBasedP1DGLimiter, self).apply(tmp_func)
                    # Writes one component only, so the other entries must stay valid: use rw.
                    field.dat.data_rw_with_halos[:, i] = tmp_func.dat.data_ro_with_halos[:]
                self.P1DG.restore_work_function(tmp_func)
            else:
                super(VertexBasedP1DGLimiter, self).apply(field)
