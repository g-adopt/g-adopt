SetFactory("OpenCASCADE");  // Use OpenCASCADE geometry kernel

// Define radii
r_inner = 0.1;
r_cmb = 1.22;
r_surface = 2.22;
r_inf = 44.4;

// Define mesh sizes
lc_coarse = 1.0;
lc_fine = 0.1;

// Define four concentric circles
//Circle(1) = {0, 0, 0, r_inner, 0, 2*Pi};
Circle(2) = {0, 0, 0, r_cmb, 0, 2*Pi};
Circle(3) = {0, 0, 0, r_surface, 0, 2*Pi};
Circle(4) = {0, 0, 0, r_inf, 0, 2*Pi};

// Create curve loops
//Curve Loop(1) = {1};
Curve Loop(2) = {2};
Curve Loop(3) = {3};
Curve Loop(4) = {4};

// Define annular surfaces
Plane Surface(1) = {4, 3};  // Outer region (r_inf - r_surface)
Plane Surface(2) = {3, 2};  // Middle region (r_surface - r_cmb)
Plane Surface(3) = {2};  // Inner region (r_cmb - r_inner)

MeshSize {4} = 10.0; // Force coarse mesh on all outer boundaries
MeshSize {3} = 10.0; // Fine mesh for CMB and Surface boundaries
MeshSize {2} = 0.04; // Force coarse mesh on all outer boundaries
MeshSize {1} = 0.2; // Force coarse mesh on all outer boundaries

// Assign physical groups
Physical Curve("Core") = {1};
Physical Curve("CMB") = {2};
Physical Curve("Surface") = {3};
Physical Curve("Infinity") = {4};
Physical Surface("Outside") = {1};
Physical Surface("Inside") = {3};
Physical Surface("Domain") = {2};

// Generate mesh
Mesh 2;