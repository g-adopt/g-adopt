#!/usr/bin/env python3

import os
import shutil

# Range:
ns = [2, 4, 8]
levels = [2, 3, 4, 5]
resolutions = ["A", "B", "C", "D"]
cores = ["4", "14", "28", "112"]
mem = ["50", "100", "250", "500"]
queue = ["normalbw", "normalbw", "normalbw", "normalbw"]

# Make directories:
print("****** Making Directories *******")
for n in ns:
    n_dir_name = f"n{n}"
    print("Working on directory: ", n_dir_name)
    # Make directories
    retval = os.getcwd()
    os.mkdir(n_dir_name)
    os.chdir(n_dir_name)
    # Make resolution directories and copy across relevant files:
    for resolution in resolutions:
        os.mkdir(resolution)
        shutil.copy2("../template/job.sh", resolution)
        shutil.copy2("../stokes_bilinear.py", resolution)
    os.chdir(retval)

# Now make job submission scripts:
print("****** Making Job Scripts *******")
for n in ns:
    n_dir_name = f"n{n}"
    print("Working on directory: ", n_dir_name)
    retvallm = os.getcwd()
    os.chdir(n_dir_name)
    for i, resolution in enumerate(resolutions):
        retvalres = os.getcwd()
        os.chdir(resolution)
        job_name = n_dir_name+"_"+resolution
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
        cmd = f'sed -i "s/NTEMPLATE/{n}/g" job.sh'
        os.system(cmd)
        os.chdir(retvalres)
    os.chdir(retvallm)
