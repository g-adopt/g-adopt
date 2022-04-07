#!/usr/bin/env python3

import os
import numpy

# Range:
l = numpy.array([2, 4, 8])
resolutions = ["A", "B", "C", "D"]

# Submit jobs:
for degree in range(len(l)):
    m = numpy.array([l[degree]/2, l[degree]], dtype=int)
    for order in range(len(m)):
        if m[order] <= l[degree]:
            lm_dir_name = f"l{l[degree]}_m{m[order]}"
            print("Working on directory: ", lm_dir_name)
            retvallm = os.getcwd()
            os.chdir(lm_dir_name)
            for resolution in range(len(resolutions)):
                resolution_dir_name = str(resolutions[resolution])
                retvalres = os.getcwd()
                os.chdir(resolution_dir_name)
                print("Working on resolution: ", resolution_dir_name)
                cmd = "qsub job.sh"
                os.system(cmd)
                os.chdir(retvalres)
            os.chdir(retvallm)
