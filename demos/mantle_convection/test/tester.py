from firedrake import *
from firedrake.adjoint import *
# from memory_profiler import profile
import gc


def test():
    T_c, rf = rf_generator()
    rf.fwd_call = profile(rf.__call__)
    rf.derivative = profile(rf.derivative)

    for i in range(5):
        gc.collect()
        rf.fwd_call(T_c)
        gc.collect()
        rf.derivative()


@profile
def rf_generator():
    tape = get_working_tape()
    tape.clear_tape()
    continue_annotation()
    enable_disk_checkpointing()
    mesh = RectangleMesh(100, 100, 1.0, 1.0)
    mesh = checkpointable_mesh(mesh)

    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)

    # Define the rotation vector field
    X = SpatialCoordinate(mesh)

    w = Function(V, name="rotation").interpolate(as_vector([-X[1] - 0.5, X[0] - 0.5]))
    T_c = Function(Q, name="control")
    T = Function(Q, name="Temperature")
    T_c.interpolate(0.1 * exp(-0.5 * ((X - as_vector((0.75, 0.5))) / Constant(0.1)) ** 2))
    control = Control(T_c)
    T.assign(T_c)

    for i in range(20):
        T.interpolate(T + inner(grad(T), w) * Constant(0.0001))

    objective = assemble(T**2 * dx)

    pause_annotation()
    return T_c, ReducedFunctional(objective, control)


if __name__ == "__main__":
    test()
