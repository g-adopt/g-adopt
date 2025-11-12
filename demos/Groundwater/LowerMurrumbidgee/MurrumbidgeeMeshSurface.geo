// Gmsh project created on Thu Jul 17 13:41:50 2025
//+
Lc = 2000;
//+
Point(3) = {0, 35000, 0, Lc};
//+
Point(4) = {140000, 0, 0, Lc};
//+
Point(5) = {280000, 0, 0, Lc};
//+
Point(6) = {280000, 68000, 0, Lc};
//+
Point(7) = {201000, 130000, 0, Lc};
//+
Point(8) = {121000, 130000, 0, Lc};
//+
Point(9) = {0, 100000, 0, Lc};

//+
Line(1) = {5, 6};
Line(2) = {6, 7};
Line(3) = {7, 8};
Line(4) = {8, 9};
Line(5) = {9, 3};
Line(6) = {3, 4};
Line(7) = {4, 5};

Curve Loop(1) = {1, 2, 3, 4, 5, 6, 7};

Plane Surface(1) = {1};
Physical Curve(1) = {1,2,3,4,5,6,7};
//+

Physical Surface(1) = {1};