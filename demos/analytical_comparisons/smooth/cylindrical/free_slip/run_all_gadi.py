#!/usr/bin/env python3

import os
import numpy

# Range:
ks = numpy.array([2, 4])
resolutions = ["A", "B", "C", "D"]

# Submit jobs:
for k in range(len(ks)):
    ns = numpy.array([ks[k], ks[k]*2, ks[k]*4], dtype=int)
    for n in range(len(ns)):
        kn_dir_name = "k"+str(ks[k])+"_n"+str(ns[n])
        print("Working on directory: ", kn_dir_name)
        retvallm = os.getcwd()
        os.chdir(kn_dir_name)
        for resolution in range(len(resolutions)):
            resolution_dir_name = str(resolutions[resolution])
            retvalres = os.getcwd()
            os.chdir(resolution_dir_name)
            print("Working on resolution: ", resolution_dir_name)
            cmd = "qsub job.sh"
            os.system(cmd)
            os.chdir(retvalres)
        os.chdir(retvallm)
