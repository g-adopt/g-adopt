vertical_resolution = 4e3;

domain_dimension_x = 1e6;
domain_dimension_y = 6.6e5;

layer_min_x = 4.2e5;
layer_thickness = 1.6e5;
layer_horizontal_resolution = 1e3;

Point(1) = {0, 0, 0, vertical_resolution};
Point(2) = {0, domain_dimension_y, 0, vertical_resolution};

Line(1) = {1, 2};

Extrude {layer_min_x, 0, 0} {
  Curve{1}; Layers {42}; Recombine;  // Horizontal resolution: 10 km
}
Extrude {layer_thickness, 0, 0} {
  Curve{2}; Layers {layer_thickness / layer_horizontal_resolution}; Recombine;
}
Extrude {domain_dimension_x - layer_min_x - layer_thickness, 0, 0} {
  Curve{6}; Layers {42}; Recombine;  // Horizontal resolution: 10 km
}

Physical Curve(1) = {1};
Physical Curve(2) = {10};
Physical Curve(3) = {3, 7, 11};
Physical Curve(4) = {4, 8, 12};

Physical Surface(1) = {5, 9, 13};

