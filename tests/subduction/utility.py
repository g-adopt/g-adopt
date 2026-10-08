import firedrake as fd
import gmsh
from ufl.core.operator import Operator
from ufl.indexed import Indexed


def clip_expression(
    expr: Operator, minimum: float | Operator, maximum: float | Operator
) -> Operator:
    return fd.min_value(fd.max_value(expr, minimum), maximum)


def function_name(field: fd.Function | Indexed) -> str:
    if isinstance(field, Indexed):
        return f"{field.ufl_operands[0].name()}_{field.ufl_operands[1].indices()[0]}"
    else:
        return field.name()


def generate_mesh(
    domain_dims: tuple[float, float], mesh_layers: dict[str, float | list[float]]
) -> None:
    thicknesses = mesh_layers["thickness"]
    vertical_resolutions = mesh_layers["vertical_resolution"]
    if len(thicknesses) != len(vertical_resolutions):
        raise ValueError(
            "Mesh layer thicknesses and vertical resolutions must have equal lengths."
        )
    if not thicknesses or any(value <= 0.0 for value in thicknesses):
        raise ValueError("Mesh layer thicknesses must be positive and non-empty.")
    if any(value <= 0.0 for value in vertical_resolutions):
        raise ValueError("Mesh layer vertical resolutions must be positive.")
    if mesh_layers["horizontal_resolution"] <= 0.0:
        raise ValueError("Mesh horizontal resolution must be positive.")

    gmsh.initialize()
    gmsh.model.add("mesh")

    point_1 = gmsh.model.geo.addPoint(
        0.0, 0.0, 0.0, mesh_layers["horizontal_resolution"]
    )
    point_2 = gmsh.model.geo.addPoint(
        domain_dims[0], 0.0, 0.0, mesh_layers["horizontal_resolution"]
    )

    line_1 = gmsh.model.geo.addLine(point_1, point_2)

    top_curve = line_1
    side_curves = []
    layer_surfaces = []
    for layer_thickness, layer_resolution in zip(thicknesses, vertical_resolutions):
        extruded_entities = gmsh.model.geo.extrude(
            [(1, top_curve)],
            0.0,
            layer_thickness,
            0.0,
            numElements=[round(layer_thickness / layer_resolution)],
            recombine=False,
        )
        surfaces = [tag for dim, tag in extruded_entities if dim == 2]
        curves = [tag for dim, tag in extruded_entities if dim == 1]
        if not surfaces or len(curves) < 3:
            raise RuntimeError("Gmsh extrusion did not return the expected entities.")

        # Gmsh returns the new top curve before the two lateral curves.
        top_curve = curves[0]
        side_curves.extend(curves[1:])
        layer_surfaces.extend(surfaces)

    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(1, [line_1], tag=1)
    gmsh.model.addPhysicalGroup(1, [top_curve], tag=2)
    gmsh.model.addPhysicalGroup(1, side_curves[::2], tag=3)
    gmsh.model.addPhysicalGroup(1, side_curves[1::2], tag=4)

    gmsh.model.addPhysicalGroup(2, layer_surfaces, tag=1)

    gmsh.model.mesh.generate(2)

    gmsh.write("mesh.msh")
    gmsh.finalize()


def half_space_cooling_model(
    T_cold: float, T_hot: float, depth: Operator, kappa: float, age: float
) -> Operator:
    hscm = T_cold + (T_hot - T_cold) * fd.erf(depth / 2 / fd.sqrt(kappa * age))

    return fd.conditional(fd.eq(age, 0.0), T_hot, hscm)


def tensor_second_invariant(tensor: Operator, regularisation: float = 0.0) -> Operator:
    return fd.sqrt(fd.inner(tensor, tensor) / 2.0 + regularisation**2.0)
