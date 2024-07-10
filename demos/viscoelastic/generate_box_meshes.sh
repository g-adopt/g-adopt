#!/bin/bash

# Generates three box meshes with refined regions near the ice load
# 'refined_dx' is the the horizontal and vertical resolution in km.
# The '-' is needed to prevent gmsh from creating another 3d mesh 
# file with the default name. Maybe there is a neater way of doing this!
gmsh - weerdesteijn_box_refined.geo -setnumber refined_dx 5
gmsh - weerdesteijn_box_refined.geo -setnumber refined_dx 10
gmsh - weerdesteijn_box_refined.geo -setnumber refined_dx 20
gmsh - weerdesteijn_box_refined.geo -setnumber refined_dx 6.25
gmsh - weerdesteijn_box_refined.geo -setnumber refined_dx 2.5
