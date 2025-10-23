#!/bin/bash

# Generates three surface meshes with refined regions near the ice load
# 'refined_dx' is the the horizontal resolution in km.
# The '-' is needed to prevent gmsh from creating another 2d mesh 
# file with the default name. Maybe there is a neater way of doing this!
gmsh - weerdesteijn_box_refined_surface.geo -setnumber refined_dx 5
gmsh - weerdesteijn_box_refined_surface.geo -setnumber refined_dx 10
gmsh - weerdesteijn_box_refined_surface.geo -setnumber refined_dx 20
