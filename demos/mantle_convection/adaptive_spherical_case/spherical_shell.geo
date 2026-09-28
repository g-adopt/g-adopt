SetFactory("OpenCASCADE");

Sphere(1) = {0, 0, 0, 1.22, -Pi/2, Pi/2, 2*Pi};
Sphere(2) = {0, 0, 0, 2.22, -Pi/2, Pi/2, 2*Pi};


BooleanDifference(3) = { Volume{2}; Delete; }{ Volume{1}; Delete; };
Characteristic Length{ PointsOf{ Volume{3}; } } = 0.174;

bnd() = CombinedBoundary{ Volume{3}; };
Physical Surface(2) = {bnd(0)};
Physical Surface(1) = {bnd(1)};
Physical Volume(3) = { 3 };

