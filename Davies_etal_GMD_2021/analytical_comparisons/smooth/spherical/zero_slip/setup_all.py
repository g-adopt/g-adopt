#!/usr/bin/env python3

import os
import shutil
import numpy

# Range:
ls = numpy.array([2, 4, 8])
ks = ls + 1
levels = [3, 4, 5, 6]
layers = [8, 16, 32, 64]
resolutions = ["A", "B", "C", "D"]
cores = ["56", "448", "896", "1792"]
mem = ["500", "4000", "8000", "16000"]
queue = ["normalbw", "normalbw", "normalbw", "normalbw"]

# Make directories:
print("****** Making Directories *******")
for l in ls:
    ms = numpy.array([l/2, l], dtype=int)
    for m in ms:
        if m <= l:
            lm_dir_name = f"l{l}_m{m}"
            print("Working on directory: ", lm_dir_name)
            # Make directories
            retval = os.getcwd()
            os.mkdir(lm_dir_name)
            os.chdir(lm_dir_name)
            # Make resolution directories and copy across relevant files:
            for resolution in resolutions:
                os.mkdir(resolution)
                shutil.copy2("../template/job.sh", resolution)
                shutil.copy2("../stokes_trilinear.py", resolution)
            os.chdir(retval)

# Now make job submission scripts:
print("****** Making Job Scripts *******")
for k, l in zip(ks, ls):
    ms = numpy.array([l/2, l], dtype=int)
    for m in ms:
        if m <= l:
            lm_dir_name = f"l{l}_m{m}"
            print("Working on directory: ", lm_dir_name)
            retvallm = os.getcwd()
            os.chdir(lm_dir_name)
            for i, resolution in enumerate(resolutions):
                retvalres = os.getcwd()
                os.chdir(resolution)
                job_name = lm_dir_name + "_" + resolution
                cmd = f'sed -i "s/TEMPLATENAME/#PBS -N {job_name}/g" job.sh'
                os.system(cmd)
                cmd = f'sed -i "s/TEMPLATEQUEUE/#PBS -q {queue[i]}/g" job.sh'
                os.system(cmd)
                cmd = f'sed -i "s/TEMPLATEMEM/#PBS -l mem={mem[i]}GB/g" job.sh'
                os.system(cmd)
                cmd = f'sed -i "s/TEMPLATECPUS/#PBS -l ncpus={cores[i]}/g" job.sh'
                os.system(cmd)
                cmd = f'sed -i "s/TEMPLATELEVELS/{levels[i]}/g" job.sh'
                os.system(cmd)
                cmd = f'sed -i "s/TEMPLATELAYERS/{layers[i]}/g" job.sh'
                os.system(cmd)
                cmd = f'sed -i "s/KTEMPLATE/{k}/g" job.sh'
                os.system(cmd)
                cmd = f'sed -i "s/LTEMPLATE/{l}/g" job.sh'
                os.system(cmd)
                cmd = f'sed -i "s/MTEMPLATE/{m}/g" job.sh'
                os.system(cmd)
                os.chdir(retvalres)
            os.chdir(retvallm)
