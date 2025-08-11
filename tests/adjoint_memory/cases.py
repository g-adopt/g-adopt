from checkpoint_schedules import SingleMemoryStorageSchedule, SingleDiskStorageSchedule

schedules = {
    "noscheduler": None,  # No scheduler used
    "fullmemory": SingleMemoryStorageSchedule(),  # Store all data in memory
    "fullstorage": SingleDiskStorageSchedule(),  # Store all data on disk
}
