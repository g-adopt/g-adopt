import gc
from firedrake.petsc import PETSc

# Wrapper to run garbage collection
def collect_garbage(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        gc.collect()
        PETSc.garbage_cleanup()
        return result

    return wrapper
