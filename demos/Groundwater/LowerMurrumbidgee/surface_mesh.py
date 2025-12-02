import gmsh
import sys

def surface_mesh(dx):

    # Makes the two dimensional surface mesh for the Lower Murrumbidgee 

    gmsh.initialize()

    gmsh.option.setNumber("General.Terminal", 0)

    gmsh.model.add("surface_mesh")

    p3 = gmsh.model.geo.addPoint(0,       35000,  0, dx)
    p4 = gmsh.model.geo.addPoint(140000,  0,      0, dx)
    p5 = gmsh.model.geo.addPoint(280000,  0,      0, dx)
    p6 = gmsh.model.geo.addPoint(280000,  68000,  0, dx)
    p7 = gmsh.model.geo.addPoint(201000,  130000, 0, dx)
    p8 = gmsh.model.geo.addPoint(121000,  130000, 0, dx)
    p9 = gmsh.model.geo.addPoint(0,       100000, 0, dx)

    l1 = gmsh.model.geo.addLine(p5, p6)
    l2 = gmsh.model.geo.addLine(p6, p7)
    l3 = gmsh.model.geo.addLine(p7, p8)
    l4 = gmsh.model.geo.addLine(p8, p9)
    l5 = gmsh.model.geo.addLine(p9, p3)
    l6 = gmsh.model.geo.addLine(p3, p4)
    l7 = gmsh.model.geo.addLine(p4, p5)

    curve_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4, l5, l6, l7])
    surface = gmsh.model.geo.addPlaneSurface([curve_loop])

    gmsh.model.geo.synchronize()

    sides = gmsh.model.addPhysicalGroup(1, [l1, l2, l3, l4, l5, l6, l7], 1)
    gmsh.model.setPhysicalName(1, 1, "sides")

    domain = gmsh.model.addPhysicalGroup(2, [surface], 201)
    gmsh.model.setPhysicalName(2, 1, "FluidDomain")

    gmsh.model.mesh.generate(2)
    gmsh.write("MurrumbidgeeMeshSurface.msh")
    gmsh.finalize()