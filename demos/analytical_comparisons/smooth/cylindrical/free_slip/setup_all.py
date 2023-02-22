#!/usr/bin/env python3

import os
import shutil
import numpy

# Range:
ks = numpy.array([2, 4, 8])
levels = [2, 3, 4]
resolutions = ["A", "B", "C"]
cores = ["4", "14", "28"]
mem = ["50", "100", "250"]
queue = ["normalbw", "normalbw", "normalbw"]

# Make directories:
print("****** Making Directories *******")
for k in ks:
    ns = numpy.array([k, k*2, k*4], dtype=int)
    for n in ns:
        kn_dir_name = f"k{k}_n{n}"
        print("Working on directory: ", kn_dir_name)
        # Make directories
        retval = os.getcwd()
        os.mkdir(kn_dir_name)
        os.chdir(kn_dir_name)
        # Make resolution directories and copy across relevant files:
        for resolution in resolutions:
            os.mkdir(resolution)
            shutil.copy2("../template/job.sh", resolution)
            shutil.copy2("../stokes.py", resolution)
        os.chdir(retval)

# Now make job submission scripts:
print("****** Making Job Scripts *******")
for k in ks:
    ns = numpy.array([k, k*2, k*4], dtype=int)
    for n in ns:
        kn_dir_name = f"k{k}_n{n}"
        print("Working on directory: ", kn_dir_name)
        retvallm = os.getcwd()
        os.chdir(kn_dir_name)
        for i, resolution in enumerate(resolutions):
            retvalres = os.getcwd()
            os.chdir(resolution)
            job_name = kn_dir_name+"_"+resolution
            cmd = f'sed -i "s/TEMPLATENAME/#PBS -N {job_name}/g" job.sh'
            os.system(cmd)
            cmd = f'sed -i "s/TEMPLATEQUEUE/#PBS -q {queue[i]}/g" job.sh'
            os.system(cmd)
            cmd = f'sed -i "s/TEMPLATEMEM/#PBS -l mem={mem[i]}GB/g" job.sh'
            os.system(cmd)
            cmd = f'sed -i "s/TEMPLATECPUS/#PBS -l ncpus={cores[i]}/g" job.sh'
            os.system(cmd)
            cmd = f'sed -i "s/TEMPLATELEVELS/{2**levels[i]}/g" job.sh'
            os.system(cmd)
            cmd = f'sed -i "s/KTEMPLATE/{k}/g" job.sh'
            os.system(cmd)
            cmd = f'sed -i "s/NTEMPLATE/{n}/g" job.sh'
            os.system(cmd)
            os.chdir(retvalres)
        os.chdir(retvallm)
