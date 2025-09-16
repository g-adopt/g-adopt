horizontal_resolution = 0.03125;

domain_dimension_x = 2;
domain_dimension_y = 1;

bottom_thickness = 0.12;
top_thickness = 0.12;
layer_resolution = 0.008;

Point(1) = {0, 0, 0, horizontal_resolution};
Point(2) = {domain_dimension_x, 0, 0, horizontal_resolution};

Line(1) = {1, 2};

Extrude {0, bottom_thickness, 0} {
  Curve{1}; Layers {bottom_thickness / layer_resolution}; Recombine;
}
Extrude {0, domain_dimension_y - bottom_thickness - top_thickness, 0} {
  Curve{2}; Layers {16}; Recombine;  // Vertical resolution: 0.0475
}
Extrude {0, top_thickness, 0} {
  Curve{6}; Layers {top_thickness / layer_resolution}; Recombine;
}

Physical Curve(1) = {3, 7, 11};
Physical Curve(2) = {4, 8, 12};
Physical Curve(3) = {1};
Physical Curve(4) = {10};

Physical Surface(1) = {5, 9, 13};
