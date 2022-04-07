#!/usr/bin/env python3

import os
import numpy

# Range:
ns = numpy.array([2, 4, 8])
resolutions = ["A", "B", "C", "D"]

# Submit jobs:
for n in range(len(ns)):
    n_dir_name = "n"+str(ns[n])
    print("Working on directory: ", n_dir_name)
    retvallm = os.getcwd()
    os.chdir(n_dir_name)
    for resolution in range(len(resolutions)):
        resolution_dir_name = str(resolutions[resolution])
        retvalres = os.getcwd()
        os.chdir(resolution_dir_name)
        print("Working on resolution: ", resolution_dir_name)
        cmd = "qsub job.sh"
        os.system(cmd)
        os.chdir(retvalres)
    os.chdir(retvallm)
