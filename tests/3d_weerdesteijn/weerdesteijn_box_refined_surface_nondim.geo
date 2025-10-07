// Based off tutorial 10 from the Gmsh manual

// Let's create a simple rectangular geometry
km = 1e3;

// Scale mesh by depth of domain to nondimensionalise
D = 2891 * km;
lc = 200*km/D;
L = 1500*km/D;
Point(1) = {0.0,0.0,0,lc}; Point(2) = {L,0.0,0,lc};
Point(3) = {L,L,0,lc};     Point(4) = {0,L,0,lc};

Line(1) = {1,2}; Line(2) = {2,3}; Line(3) = {3,4}; Line(4) = {4,1};

Curve Loop(5) = {1,2,3,4}; Plane Surface(1) = {5};

// Choose boundary ids to match Firedrake's unit meshes... 
//    1: plane x == 0
//    2: plane x == L
//    3: plane y == 0
//    4: plane y == L

Physical Curve(1) = 4;
Physical Curve(2) = 2;
Physical Curve(3) = 1;
Physical Curve(4) = 3;

Physical Surface(1) = {1};


// Set up a distance field from point 1 i.e. (0,0) where the iceloading will be 
Field[1] = Distance;
Field[1].PointsList = {1};


// We then define a `Threshold' field, which uses the return value of the
// `Distance' field 1 in order to define a simple change in element size
// depending on the computed distances. This will make the mesh 40x finer 
// near point 1 i.e. dx = dy = 5 km.
//
// SizeMax -                     /------------------
//                              /
//                             /
//                            /
// SizeMin -o----------------/
//          |                |    |
//        Point         DistMin  DistMax

// Uncomment below for default refinement wtih 5 km resolution near ice load but this 
// can be set in the command line using -setnumber refined_dx 5
// refined_dx = 5;
Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = refined_dx*km/D;
Field[2].SizeMax = lc;
Field[2].DistMin = 450*km/D;
Field[2].DistMax = 900*km/D;

Background Field = 2;
// below is from the tutorial 
// To determine the size of mesh elements, Gmsh locally computes the minimum of
//
// 1) the size of the model bounding box;
// 2) if `Mesh.MeshSizeFromPoints' is set, the mesh size specified at
//    geometrical points;
// 3) if `Mesh.MeshSizeFromCurvature' is positive, the mesh size based on
//    curvature (the value specifying the number of elements per 2 * pi rad);
// 4) the background mesh size field;
// 5) any per-entity mesh size constraint.
//
// This value is then constrained in the interval [`Mesh.MeshSizeMin',
// `Mesh.MeshSizeMax'] and multiplied by `Mesh.MeshSizeFactor'. In addition,
// boundary mesh sizes are interpolated inside surfaces and/or volumes depending
// on the value of `Mesh.MeshSizeExtendFromBoundary' (which is set by default).
//
// When the element size is fully specified by a mesh size field (as it is in
// this example), it is thus often desirable to set

Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;

// This will prevent over-refinement due to small mesh sizes on the boundary.

// Finally, while the default "Frontal-Delaunay" 2D meshing algorithm
// (Mesh.Algorithm = 6) usually leads to the highest quality meshes, the
// "Delaunay" algorithm (Mesh.Algorithm = 5) will handle complex mesh size
// fields better - in particular size fields with large element size gradients:

Mesh.Algorithm = 5;
Mesh 2;
Save Sprintf("weerdesteijn_box_refined_surface_%gkm_nondim.msh", refined_dx);
