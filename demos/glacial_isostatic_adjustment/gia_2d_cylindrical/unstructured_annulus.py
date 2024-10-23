import gmsh
import sys

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
gmsh.model.geo.addPlaneSurface([1,2 ], 1)


# the number of synchronization points.
gmsh.model.geo.synchronize()


# Set physical groups for boundary tags
# Bottom id: 1, Top id: 2
gmsh.model.addPhysicalGroup(1, [1,2,3,4], 2)
gmsh.model.addPhysicalGroup(1, [5,6,7,8], 1)
gmsh.model.addPhysicalGroup(2, [1], name="My surface")

#gmsh.option.setNumber("Mesh.MeshSizeMin", 0.001)
#gmsh.option.setNumber("Mesh.MeshSizeMax", 0.003)
#gmsh.model.mesh.setAlgorithm(2,1,8)
gmsh.model.mesh.generate(2)

# ... and save it to disk
gmsh.write("unstructured_annulus.msh")

gmsh.finalize()
