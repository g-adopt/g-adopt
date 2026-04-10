import gmsh
import sys


def surface_mesh(dx_base):

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    gmsh.option.setNumber("Mesh.Algorithm", 6)

    gmsh.model.add("murrumbidgee_hierarchy")

    # --- Points (Fixed corners of your domain) ---
    p3 = gmsh.model.geo.addPoint(0,       35000,  0, dx_base)
    p4 = gmsh.model.geo.addPoint(140000,  0,      0, dx_base)
    p5 = gmsh.model.geo.addPoint(280000,  0,      0, dx_base)
    p6 = gmsh.model.geo.addPoint(280000,  68000,  0, dx_base)
    p7 = gmsh.model.geo.addPoint(201000,  130000, 0, dx_base)
    p8 = gmsh.model.geo.addPoint(121000,  130000, 0, dx_base)
    p9 = gmsh.model.geo.addPoint(0,       100000, 0, dx_base)

    # --- Lines ---
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
    
    # Boundary IDs
    gmsh.model.addPhysicalGroup(1, [l1, l2, l3, l4, l5, l6, l7], 3)
    gmsh.model.setPhysicalName(1, 3, "sides")
    
    gmsh.model.addPhysicalGroup(2, [surface], 201)
    gmsh.model.setPhysicalName(2, 201, "FluidDomain")

    gmsh.model.mesh.generate(2)
    gmsh.write("Murrumbidgee_SurfaceMesh.msh")

    gmsh.finalize()
