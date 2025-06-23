import adjoint
import inspect
import pyadjoint
import sys

import numpy as np

from checkpoint_schedules import SingleMemoryStorageSchedule
from firedrake import Function
from unittest.mock import patch

class MonitorProxy:
    def __init__(self, obj, name):
        object.__setattr__(self, "_obj", obj)
        object.__setattr__(self, "_name", name)

    def __getattribute__(self, name):
        if name == "_obj":
            return object.__getattribute__(self, name)

        obj = object.__getattribute__(self, "_obj")
        obj_name = object.__getattribute__(self, "_name")
        attr = getattr(obj, name)

        if callable(attr):
            def wrapped(*args, **kwargs):
                print(f"\nCalling method: {name} on {obj_name}")
                for frame in inspect.stack()[1:5]:
                    print(f"  File {frame.filename}, line {frame.lineno}, in {frame.function}")
                result = attr(*args, **kwargs)
                print(f"  → Returned: {result!r}")
                return result
            return wrapped

        print(f"\nAccessing attribute: {name} on {obj_name}")
        # breakpoint()
        for frame in inspect.stack()[1:5]:
            print(f"  File {frame.filename}, line {frame.lineno}, in {frame.function}")
        print(f"  → Returned: {attr!r}")
        return attr

class MonitorMixin:
    def __getattribute__(self, name):
        attr = super().__getattribute__(name)

        if callable(attr):
            def wrapped(*args, **kwargs):
                print(f"\nCalling method: {name}")
                for frame in inspect.stack()[1:5]:
                    print(f"  File {frame.filename}, line {frame.lineno}, in {frame.function}")
                result = attr(*args, **kwargs)
                print(f"  → Returned: {result!r}")
                return result
            return wrapped

        print(f"\nAccessing attribute: {name}")
        # breakpoint()
        for frame in inspect.stack()[1:5]:
            print(f"  File {frame.filename}, line {frame.lineno}, in {frame.function}")
        print(f"  → Returned: {attr!r}")
        return attr

def monitor(obj):
    # class Monitored(obj.__class__, MonitorMixin):
    #     pass

    Monitored = type(
        f"Monitored{obj.__class__.__name__}",
        (MonitorMixin, obj.__class__),
        {},
    )

    obj.__class__ = Monitored
    return obj

callchain = open(f"{sys.argv[1]}-calls.txt", "w")

def patch_all_methods(obj, obj_name, *outer_args):
    patches = []

    for name, attr in inspect.getmembers(obj.__class__, predicate=inspect.isfunction):
        if name.startswith("__") and name.endswith("__"):
            continue

        print(f"wrapping {name}")

        def make_wrapper(method_name, orig_func):
            def wrapped(*args, **kwargs):
                print(f"\nCalling method: {method_name} on {obj_name}")
                callchain.write(f"Calling: {method_name} on {obj_name}\n")

                for frame in inspect.stack()[1:5]:
                    print(f"  File {frame.filename}, line {frame.lineno}, in {frame.function}")
                result = orig_func(*args, **kwargs)
                return result

                # XXX return value printing if needed
                if hasattr(result, "__len__"):
                    l = len(result)
                    print(f"  → Returned ({l}): {result!r}\n")
                    callchain.write(f"  → Returned ({l}): {result!r}\n\n")
                else:
                    print(f"  → Returned: {result!r}\n")
                    callchain.write(f"  → Returned: {result!r}\n\n")
                return result
            return wrapped

        wrapper = make_wrapper(name, attr)
        patcher = patch.object(obj, name, wrapper.__get__(obj, obj.__class__))
        patcher.start()
        patches.append(patcher)

    return patches

match sys.argv[1]:
    case "memory":
        scheduler = SingleMemoryStorageSchedule()

    case "none":
        scheduler = None

    case _:
        raise ValueError

Tic, reduced_functional = adjoint.forward_problem(scheduler=scheduler, age0=10.0)

pyadjoint.pause_annotation()

rng = np.random.default_rng(42)
dtemp = Function(Tic.function_space())
dtemp.dat.data_wo[:] = rng.random(dtemp.dat.data_ro.shape)

tape = pyadjoint.get_working_tape()

s0 = tape.get_blocks().steps[1]
print(s0)
#s1 = tape.get_blocks().steps[2]

Jm = reduced_functional(Tic)
print(f"reduced functional: {Jm}")

dJdm = reduced_functional.derivative()
dJdm = dtemp._ad_dot(dJdm)
print(f"dJdm 1: {dJdm}")

Jm = reduced_functional(Tic)
print(f"reduced functional: {Jm}")

print(f"Assign block: {s0[2]!r}")
assign_out = s0[2]._outputs[0]
#s0[2]._outputs[0] = MonitorProxy(s0[2]._outputs[0], "assign")
#monitor(s0[2]._outputs[0])
#patch_all_methods(s0[2], "assign block")
#patch_all_methods(s0[2]._outputs[0], "assign output")
#patch_all_methods(s0[4], "timestep", assign_out)
#patch_all_methods(s0[5], "intermediate", assign_out)
#patch_all_methods(s0[6], "final", assign_out)

"""
for i, block in enumerate(tape.get_blocks()):
    if block is s0[4]:
        print("found")
        tape._blocks[i] = MonitorProxy(s0[4], "timestep")
        break

if scheduler is not None:
    #print(f"type of block {type(s0[4])}")
    s0[4] = MonitorProxy(s0[4], "timestep")
"""

dJdm = reduced_functional.derivative()
print(f"Gradient along perturbation: {dtemp._ad_dot(dJdm)}")

callchain.close()
