import gmsh

# Before using any functions in the Python API, Gmsh must be initialized:
gmsh.initialize()

gmsh.model.add("t1")

lc = 500e3

Re = 6371e3
Rc = 3480e3


p1 = gmsh.model.geo.addPoint(0, 0, 0, lc, 1)
p2 = gmsh.model.geo.addPoint(Re, 0, 0, lc, 2)
p3 = gmsh.model.geo.addPoint(0, Re, 0, lc, 3)
p4 = gmsh.model.geo.addPoint(-Re, 0, 0, lc, 4)
p5 = gmsh.model.geo.addPoint(0, -Re, 0, lc, 5)
p6 = gmsh.model.geo.addPoint(Rc, 0, 0, lc, 6)
p7 = gmsh.model.geo.addPoint(0, Rc, 0, lc, 7)
p8 = gmsh.model.geo.addPoint(-Rc, 0, 0, lc, 8)
p9 = gmsh.model.geo.addPoint(0, -Rc, 0, lc, 9)
#
# Curves at Earth surface
gmsh.model.geo.addCircleArc(p2, p1, p3, 1)
gmsh.model.geo.addCircleArc(p3, p1, p4, 2)
gmsh.model.geo.addCircleArc(p4, p1, p5, 3)
gmsh.model.geo.addCircleArc(p5, p1, p2, 4)


# Curves at Earth surface
gmsh.model.geo.addCircleArc(p6, p1, p7, 5)
gmsh.model.geo.addCircleArc(p7, p1, p8, 6)
gmsh.model.geo.addCircleArc(p8, p1, p9, 7)
gmsh.model.geo.addCircleArc(p9, p1, p6, 8)

gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
gmsh.model.geo.addCurveLoop([5, 6, 7, 8], 2)
gmsh.model.geo.addPlaneSurface([1, 2], 1)


# the number of synchronization points.
gmsh.model.geo.synchronize()


# Set physical groups for boundary tags
# Bottom id: 1, Top id: 2
gmsh.model.addPhysicalGroup(1, [1, 2, 3, 4], 2)
gmsh.model.addPhysicalGroup(1, [5, 6, 7, 8], 1)
gmsh.model.addPhysicalGroup(2, [1], name="My surface")


# Refine near the top surface
gmsh.model.mesh.field.add("Distance", 1)
gmsh.model.mesh.field.setNumbers(1, "CurvesList", [1, 2, 3, 4])
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
gmsh.model.mesh.field.setNumber(2, "SizeMin", lc / 2.5)
gmsh.model.mesh.field.setNumber(2, "SizeMax", lc)
gmsh.model.mesh.field.setNumber(2, "DistMin", 400e3)
gmsh.model.mesh.field.setNumber(2, "DistMax", 1500e3)


gmsh.model.mesh.field.setAsBackgroundMesh(2)

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
gmsh.write("unstructured_annulus_refined_surface.msh")

gmsh.finalize()
