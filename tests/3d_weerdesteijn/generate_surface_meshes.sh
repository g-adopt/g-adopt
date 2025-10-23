#!/bin/bash

# Generates three surface meshes with refined regions near the ice load
# 'refined_dx' is the the horizontal resolution in km.
# The '-' is needed to prevent gmsh from creating another 2d mesh 
# file with the default name. Maybe there is a neater way of doing this!
gmsh -2 weerdesteijn_box_refined_surface_nondim.geo -setnumber refined_dx 5 -o weerdesteijn_box_refined_surface_5km_nondim.msh
gmsh -2 weerdesteijn_box_refined_surface_nondim.geo -setnumber refined_dx 10 -o weerdesteijn_box_refined_surface_10km_nondim.msh
gmsh -2 weerdesteijn_box_refined_surface_nondim.geo -setnumber refined_dx 20 -o weerdesteijn_box_refined_surface_20km_nondim.msh
