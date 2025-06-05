from checkpoint_schedules import SingleMemoryStorageSchedule, SingleDiskStorageSchedule

cases = [
    "damping",  # Taylor test on the damping term only  $$ (Tic - T_ave) ** 2 dx $$
    "smoothing",  # Taylor test on smoothing only  $$ \nabla (T_ic - T_ave) ** 2 dx$$
    "Tobs",  # Taylor test on temperature misfit only  $$ (T - T_obs) ** 2 dx $$
    "uobs",  # Taylor test on uobs only $$ (u - uobs) ** 2 ds_t $$
    "uimposed",  # Taylor test on temperature misfit, but with imposed velocity $$ (T - T_obs) ** 2 dx $$
]
schedules = {
    "noscheduler": None,  # No scheduler used
    "fullmemory": SingleMemoryStorageSchedule(),  # Store all data in memory
    "fullstorage": SingleDiskStorageSchedule(),  # Store all data on disk
}
