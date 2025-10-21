from checkpoint_schedules import SingleMemoryStorageSchedule, SingleDiskStorageSchedule

cases = {
    "damping": [-1, -1, +1, -1],  # only having $$ (Tic - Tave) ** 2 * dx $$ in the functional (damping)
    "smoothing": [-1, -1, -1, +1],  # only having $$ grad (Tic - Tave) ** 2 * dx $$ in the functional (smoothing)
    "Tobs": [+1, -1, -1, -1],  # only having $$ (T_f - Tobs) ** 2 * dx $$ in the functional  (Tobs)
    "uobs": [-1, +1, -1, -1],  # only having $$ (u - uobs) ** 2 * ds_t $$ in the functional  (Tobs)
    "uimposed": [+1, -1, -1, -1]  # just like Tobs but with surface velocities imposed
}

schedulers = {
    "noscheduler": None,
    "fullmemory": SingleMemoryStorageSchedule(),
    "fulldisk": SingleDiskStorageSchedule(),
}
