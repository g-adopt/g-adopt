from checkpoint_schedules import SingleMemoryStorageSchedule, SingleDiskStorageSchedule

cases = {
    "damping": [-1, -1, +1, -1],
    "smoothing": [-1, -1, -1, +1],
    "Tobs": [+1, -1, -1, -1],
    "uobs": [-1, +1, -1, -1]
}

schedulers = {
    "fullmemory": SingleMemoryStorageSchedule(),
    # "fulldisk": SingleDiskStorageSchedule(),
}
