# import gc

import firedrake as fd
import gmsh
# from animate.adapt import adapt
# from animate.metric import RiemannianMetric
from ufl.core.operator import Operator
from ufl.indexed import Indexed

# import parameters as prms


# def adapt_mesh(
#     mesh: fd.MeshGeometry,
#     mesh_fields: dict[str, dict[str, fd.Function | bool]],
#     initial: bool = False,
# ) -> tuple[fd.MeshGeometry, dict[str, dict[str, fd.Function | bool]]]:
#     for _ in range(prms.adapt_calls):
#         M = fd.TensorFunctionSpace(mesh, "CG", 1)

#         metric_fields = {}
#         for field, field_specs in mesh_fields.items():
#             if isinstance(field_specs["add_to_metric"], list):
#                 for dim, (add_to_metric, scaling) in enumerate(
#                     zip(field_specs["add_to_metric"], field_specs["scaling"])
#                 ):
#                     if add_to_metric:
#                         metric_fields[field.subfunctions[dim]] = scaling
#             elif field_specs["add_to_metric"]:
#                 metric_fields[field] = field_specs["scaling"]

#         fields = list(metric_fields)
#         for field in fields:
#             if isinstance(field.ufl_element(), fd.VectorElement):
#                 scaling = metric_fields.pop(field)
#                 for dim in range(field.ufl_shape[0]):
#                     metric_fields[field[dim]] = scaling

#         metrics = []
#         for field, scaling in metric_fields.items():
#             # Firedrake function for a metric over a mesh where a field lives
#             metric = RiemannianMetric(M, name=f"Metric ({function_name(field)})")
#             metric.set_parameters(prms.metric_parameters)  # Set metric parameters
#             metric.compute_hessian(field)  # Field Hessian
#             metric.enforce_spd()  # Ensure boundedness (symmetric positive-definite)
#             metric.assign(metric * scaling)
#             metrics.append(metric)

#         overall_metric = metrics[0].copy(deepcopy=True)  # Overall metric
#         overall_metric.rename("Metric (overall)")
#         # Minimum in all directions across all metrics
#         overall_metric.intersect(*metrics[1:])
#         overall_metric.enforce_spd()  # Ensure boundedness (symmetric positive-definite)
#         overall_metric.normalise()  # Rescale metric to achieve the desired target complexity

#         mesh = adapt(mesh, overall_metric)  # Generate new mesh based on overall metric
#         # Ensure boundary coordinates are not exceeded
#         mesh.coordinates.dat.data[:, 0].clip(
#             0.0, prms.domain_dims[0], out=mesh.coordinates.dat.data[:, 0]
#         )
#         mesh.coordinates.dat.data[:, 1].clip(
#             0.0, prms.domain_dims[1], out=mesh.coordinates.dat.data[:, 1]
#         )
#         mesh.cartesian = True  # Geometry tag informing other G-ADOPT objects

#         mesh_fields = interpolate_fields(mesh, mesh_fields)

#         if initial:
#             break

#     # Collect and clean objects associated with the old mesh
#     garbage_collect(mesh)

#     return mesh, mesh_fields


def clip_expression(
    expr: Operator, minimum: float | Operator, maximum: float | Operator
) -> Operator:
    return fd.min_value(fd.max_value(expr, minimum), maximum)


def generate_mesh(
    domain_dims: list[float], mesh_layers: dict[str, float | list[float]]
) -> None:
    gmsh.initialize()
    gmsh.model.add("mesh")

    point_1 = gmsh.model.geo.addPoint(
        0.0, 0.0, 0.0, mesh_layers["horizontal_resolution"]
    )
    point_2 = gmsh.model.geo.addPoint(
        domain_dims[0], 0.0, 0.0, mesh_layers["horizontal_resolution"]
    )

    line_1 = gmsh.model.geo.addLine(point_1, point_2)

    for i, (layer_thickness, layer_resolution) in enumerate(
        zip(mesh_layers["thickness"], mesh_layers["vertical_resolution"])
    ):
        gmsh.model.geo.extrude(
            [(1, line_1 + sum(x**2 for x in range(i + 1)))],
            0.0,
            layer_thickness,
            0.0,
            numElements=[int(layer_thickness / layer_resolution)],
            recombine=False,
        )

    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(1, [line_1], tag=1)
    gmsh.model.addPhysicalGroup(1, [line_1 + 9], tag=2)
    gmsh.model.addPhysicalGroup(1, [line_1 + i for i in [2, 6, 10]], tag=3)
    gmsh.model.addPhysicalGroup(1, [line_1 + i for i in [3, 7, 11]], tag=4)

    gmsh.model.addPhysicalGroup(2, [5, 9, 13], tag=1)

    gmsh.model.mesh.generate(2)

    gmsh.write("mesh.msh")
    gmsh.finalize()


def function_name(field: fd.Function | Indexed) -> str:
    if isinstance(field, Indexed):
        return f"{field.ufl_operands[0].name()}_{field.ufl_operands[1].indices()[0]}"
    else:
        return field.name()


# def garbage_collect(mesh: fd.MeshGeometry) -> None:
#     fd.PETSc.garbage_cleanup(mesh.comm)
#     gc.collect()


def half_space_cooling_model(
    T_cold: float, T_hot: float, depth: Operator, kappa: float, age: float
) -> Operator:
    hscm = T_cold + (T_hot - T_cold) * fd.erf(depth / 2 / fd.sqrt(kappa * age))

    return fd.conditional(fd.eq(age, 0.0), T_hot, hscm)


# def interpolate_fields(
#     mesh: fd.MeshGeometry,
#     mesh_fields: dict[str, dict[str, fd.Function | bool]],
#     new_fields: list[fd.Function] = [],
# ) -> dict[str, dict[str, fd.Function | bool]]:
#     if not new_fields:
#         for field in mesh_fields:
#             field_element = field.ufl_element()

#             if isinstance(field_element, fd.MixedElement):
#                 spaces = [
#                     fd.FunctionSpace(mesh, element)
#                     for element in field_element.sub_elements
#                 ]
#                 mixed_space = fd.MixedFunctionSpace(spaces)

#                 new_fields.append(fd.Function(mixed_space, name=field.name()))
#             else:
#                 space = fd.FunctionSpace(mesh, field_element)
#                 new_fields.append(fd.Function(space, name=field.name()))

#     for field, new_field in zip(mesh_fields, new_fields):
#         field_specs = mesh_fields.pop(field)

#         if isinstance(field.ufl_element(), fd.MixedElement):
#             for sub_field, new_sub_field in zip(
#                 field.subfunctions, new_field.subfunctions
#             ):
#                 new_sub_field.interpolate(sub_field)
#         else:
#             new_field.interpolate(field)

#         mesh_fields[new_field] = field_specs

#     # Collect and clean objects associated with the old mesh
#     garbage_collect(mesh)

#     return mesh_fields


def tensor_second_invariant(tensor: Operator, regularisation: float = 0.0) -> Operator:
    return fd.sqrt(fd.inner(tensor, tensor) / 2.0 + regularisation**2.0)


def write_output(
    output_file,
    time: float,
    *functions: fd.Function,
    field_expressions: dict[fd.Function, Operator] = {},
) -> None:
    for field, expr in field_expressions.items():
        field.interpolate(expr)

    output_file.write(*functions, *field_expressions, time=time)
