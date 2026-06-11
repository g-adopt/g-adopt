import firedrake as fd
import gmsh
from ufl.core.operator import Operator
from ufl.indexed import Indexed


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


def half_space_cooling_model(
    T_cold: float, T_hot: float, depth: Operator, kappa: float, age: float
) -> Operator:
    hscm = T_cold + (T_hot - T_cold) * fd.erf(depth / 2 / fd.sqrt(kappa * age))

    return fd.conditional(fd.eq(age, 0.0), T_hot, hscm)


def tensor_second_invariant(tensor: Operator, regularisation: float = 0.0) -> Operator:
    return fd.sqrt(fd.inner(tensor, tensor) / 2.0 + regularisation**2.0)
