horizontal_resolution = 1e4;

domain_dimension_x = 2.8e6;
domain_dimension_y = 8e5;

layer_min_y = 6.9e5;
layer_thickness = 3e4;
layer_vertical_resolution = 2.5e2;

Point(1) = {0, 0, 0, horizontal_resolution};
Point(2) = {domain_dimension_x, 0, 0, horizontal_resolution};

Line(1) = {1, 2};

Extrude {0, layer_min_y, 0} {
  Curve{1}; Layers {10}; Recombine;  // Vertical resolution: 69 km
}
Extrude {0, layer_thickness, 0} {
  Curve{2}; Layers {layer_thickness / layer_vertical_resolution}; Recombine;
}
Extrude {0, domain_dimension_y - layer_min_y - layer_thickness, 0} {
  Curve{6}; Layers {1}; Recombine;  // Vertical resolution: 80 km
}

Physical Curve(1) = {3, 7, 11};
Physical Curve(2) = {4, 8, 12};
Physical Curve(3) = {1};
Physical Curve(4) = {10};

Physical Surface(1) = {5, 9, 13};

