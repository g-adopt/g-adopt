import gmsh


def generate_mesh():
    # Before using any functions in the Python API, Gmsh must be initialized:
    gmsh.initialize()

    gmsh.model.add("t1")

    lc = 500e3
    lcg = 5000e3
    
    Rg = 31855e3
    Re = 6371e3
    grav_mesh_depth = Rg-Re
    Rc = 3480e3
    D = Re-Rc

    Rg_tilde = Rg/D
    Re_tilde = Re/D
    Rc_tilde = Rc/D
    lc_tilde = lc/D
    lcg_tilde = lcg/D
    grav_mesh_depth_tilde = grav_mesh_depth/D

    p1 = gmsh.model.geo.addPoint(0, 0, 0, lc_tilde, 1)
    p2 = gmsh.model.geo.addPoint(Rg_tilde, 0, 0, lcg_tilde, 2)
    p3 = gmsh.model.geo.addPoint(0, Rg_tilde, 0, lcg_tilde, 3)
    p4 = gmsh.model.geo.addPoint(-Rg_tilde, 0, 0, lcg_tilde, 4)
    p5 = gmsh.model.geo.addPoint(0, -Rg_tilde, 0, lcg_tilde, 5)
    p6 = gmsh.model.geo.addPoint(Re_tilde, 0, 0, lc_tilde, 6)
    p7 = gmsh.model.geo.addPoint(0, Re_tilde, 0, lc_tilde, 7)
    p8 = gmsh.model.geo.addPoint(-Re_tilde, 0, 0, lc_tilde, 8)
    p9 = gmsh.model.geo.addPoint(0, -Re_tilde, 0, lc_tilde, 9)
    p10 = gmsh.model.geo.addPoint(Rc_tilde, 0, 0, lc_tilde, 10)
    p11 = gmsh.model.geo.addPoint(0, Rc_tilde, 0, lc_tilde, 11)
    p12 = gmsh.model.geo.addPoint(-Rc_tilde, 0, 0, lc_tilde, 12)
    p13 = gmsh.model.geo.addPoint(0, -Rc_tilde, 0, lc_tilde, 13)
    #
    # Curves at gravity mesh surface
    gmsh.model.geo.addCircleArc(p2, p1, p3, 1)
    gmsh.model.geo.addCircleArc(p3, p1, p4, 2)
    gmsh.model.geo.addCircleArc(p4, p1, p5, 3)
    gmsh.model.geo.addCircleArc(p5, p1, p2, 4)

    # Curves at Earth surface
    gmsh.model.geo.addCircleArc(p6, p1, p7, 5)
    gmsh.model.geo.addCircleArc(p7, p1, p8, 6)
    gmsh.model.geo.addCircleArc(p8, p1, p9, 7)
    gmsh.model.geo.addCircleArc(p9, p1, p6, 8)
    
    # Curves at Earth surface
    gmsh.model.geo.addCircleArc(p10, p1, p11, 9)
    gmsh.model.geo.addCircleArc(p11, p1, p12, 10)
    gmsh.model.geo.addCircleArc(p12, p1, p13, 11)
    gmsh.model.geo.addCircleArc(p13, p1, p10, 12)

    # Add curves and planes (starting with CMB then Re, then Rg)
    gmsh.model.geo.addCurveLoop([9, 10, 11, 12], 1)
    gmsh.model.geo.addCurveLoop([5, 6, 7, 8], 2)
    gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 3)

    # Add surface startnig with mantle and then exterior grav mesh
    gmsh.model.geo.addPlaneSurface([1, 2], 1)
    gmsh.model.geo.addPlaneSurface([2, 3], 2)

    # the number of synchronization points.
    gmsh.model.geo.synchronize()

    # Set physical groups for boundary tags
    # Bottom id: 1, Top id: 2
    gmsh.model.addPhysicalGroup(1, [1, 2, 3, 4], 3)
    gmsh.model.addPhysicalGroup(1, [5, 6, 7, 8], 2)
    gmsh.model.addPhysicalGroup(1, [9, 10, 11, 12], 1)

    # set physical groups for surfaces mantle then exterior grav
    gmsh.model.addPhysicalGroup(2, [1], 101)
    gmsh.model.addPhysicalGroup(2, [2], 102)

    # Refine near the Earth surface
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", [5, 6, 7, 8])
    gmsh.model.mesh.field.setNumber(1, "Sampling", 100)

    # We then define a `Threshold' field, which uses the return value of the
    # `Distance' field 1 in order to define a simple change in element size
    # depending on the computed distances
    #
    # SizeMax -                     /------------------
    #                              /
    #                             /
    #                            /
    # SizeMin -o----------------/
    #          |                |    |
    #        Point         DistMin  DistMax
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", lc_tilde / 2.5)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", 20*lc_tilde)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0.3*Rc_tilde)
    gmsh.model.mesh.field.setNumber(2, "DistMax", 0.5*Rc_tilde)

    # Coarsen near the edge of gravity mesh surface
    gmsh.model.mesh.field.add("Distance", 3)
    gmsh.model.mesh.field.setNumbers(3, "CurvesList", [1, 2, 3, 4])
    gmsh.model.mesh.field.setNumber(3, "Sampling", 100)

    # We then define a `Threshold' field, which uses the return value of the
    # `Distance' field 1 in order to define a simple change in element size
    # depending on the computed distances
    #
    # SizeMax -                     /------------------
    #                              /
    #                             /
    #                            /
    # SizeMin -o----------------/
    #          |                |    |
    #        Point         DistMin  DistMax
    gmsh.model.mesh.field.add("Threshold", 4)
    gmsh.model.mesh.field.setNumber(4, "InField", 3)
    gmsh.model.mesh.field.setNumber(4, "SizeMin", lcg_tilde)
    gmsh.model.mesh.field.setNumber(4, "SizeMax", lc_tilde)
    gmsh.model.mesh.field.setNumber(4, "DistMin", 0.8*Re_tilde)
    gmsh.model.mesh.field.setNumber(4, "DistMax", grav_mesh_depth_tilde-0.5*Re_tilde)

    gmsh.model.mesh.field.add("Min", 5)
    gmsh.model.mesh.field.setNumbers(5, "FieldsList", [2, 4])
    gmsh.model.mesh.field.setAsBackgroundMesh(5)



    # When the element size is fully specified by a mesh size field (as it is in
    # this example), it is thus often desirable to set

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    # This will prevent over-refinement due to small mesh sizes on the boundary.

    # Finally, while the default "Frontal-Delaunay" 2D meshing algorithm
    # (Mesh.Algorithm = 6) usually leads to the highest quality meshes, the
    # "Delaunay" algorithm (Mesh.Algorithm = 5) will handle complex mesh size fields
    # better - in particular size fields with large element size gradients:

    gmsh.option.setNumber("Mesh.Algorithm", 5)

    gmsh.model.mesh.generate(2)

    # ... and save it to disk
    gmsh.write("unstructured_annulus_refined_surface_gravity.msh")

    gmsh.finalize()


if __name__ == "__main__":
    generate_mesh()
