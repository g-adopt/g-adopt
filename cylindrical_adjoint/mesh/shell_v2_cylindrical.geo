lr = 0.01;

Point(1) = {0, 0, 0, lr};
my_theta=0.6;

Point(201) = { 1.0*Sin(my_theta), 1.0*Cos(my_theta), 0.0, lr};
Point(202) = { 0.0, 1.0, 0.0, lr};
Point(203) = { 0.55*Sin(my_theta), 0.55*Cos(my_theta), 0.0, lr};
Point(204) = { 0.0, 0.55, 0.0, lr};

Circle(301) = {201, 1, 202};
Circle(302) = {203, 1, 204};
Line(303) = {203, 201};
Line(304) = {204, 202};

// They have to be physical to be readable by firedrake
Physical Curve(311) = {301};
Physical Curve(312) = {302};
Physical Curve(313) = {303};
Physical Curve(314) = {304};

Line Loop(401) = {301, -304, -302, 303};
Plane Surface(501) = {401};
Physical Surface(502) = {501} ;

//// Use Distance measure how far we are from upper boudary (301)
//Field[901] = Distance;
//Field[901].EdgesList = {301};
//Field[901].NNodesByEdge = 100;
////Set Threshold Field with distance from top boundary (901)
//Field[902] = Threshold;
//Field[902].IField = 901;
//Field[902].LcMin = lr/5;
//Field[902].LcMax = lr*2;
//Field[902].DistMin = 0.05;
//Field[902].DistMax = 0.1;
//
//// Now we need to measure Distance from the bottom boundary (302)
//Field[903] = Distance;
//Field[903].EdgesList = {302};
//Field[903].NNodesByEdge = 100;
//// Set Threshold Field with distance from top boundary (903)
//Field[904] = Threshold;
//Field[904].IField = 903;
//Field[904].LcMin = lr/5;
//Field[904].LcMax = lr*2;
//Field[904].DistMin = 0.05;
//Field[904].DistMax = 0.1;
//
//// The actual Background field can be set to minimum of the two fields 
//Field[905] = Min;
//Field[905].FieldsList = {902, 904};
//Background Field = 905;

