// Define the characteristic length
lc = 0.025; // Mesh element size 80x80

// Define points for a square domain
Point(1) = {0, 0, 0, lc};
Point(2) = {1, 0, 0, lc};
Point(3) = {1, 1, 0, lc};
Point(4) = {0, 1, 0, lc};

// Define lines for the boundaries
Line(1) = {1, 2}; // Bottom boundary
Line(2) = {2, 3}; // Right boundary
Line(3) = {3, 4}; // Top boundary
Line(4) = {4, 1}; // Left boundary

// Define the surface
Line Loop(5) = {1, 2, 3, 4};
Plane Surface(6) = {5};

// Recombine the surface for quadrilateral meshing
Recombine Surface {6};

// Tag boundaries with physical groups
Physical Line(11) = {1}; // Tag bottom boundary
Physical Line(12) = {2};  // Tag right boundary
Physical Line(13) = {3};    // Tag top boundary
Physical Line(14) = {4};   // Tag left boundary

// Tag the surface with a physical group
Physical Surface(15) = {6}; // Tag the entire surface