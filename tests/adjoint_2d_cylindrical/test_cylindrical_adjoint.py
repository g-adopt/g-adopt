import pytest
import numpy as np
from pathlib import Path
from cases import cases, schedulers
from gadopt import *


@pytest.mark.longtest
@pytest.mark.parametrize("case_name", cases)
def test_annulus_taylor_test(case_name):
    with open(Path(__file__).parent.resolve() / f"{case_name}_fullmemory.conv", "r") as f:
        minconv = float(f.read())

    assert minconv > 1.96


@pytest.mark.longtest
@pytest.mark.parametrize("case_name", cases)
@pytest.mark.parametrize("scheduler", schedulers)
def test_derivatives_vs_schedulers(case_name, scheduler):
    """
    Test the outputs from different checpoint_schedules outputs agains each other.
    Make sure the control and derivative in each case are the same -> Forward and Reverse on each scheduler should produce the same output
    compares the control/derivatives/ and objective+functional values.
    Control: In this case is the initial condition of the inverse simulation: Tic
    Objective is the computed expression while populating the tape.
    Functional is the result of calling ReducedFunctional.__call__([Tic])
    Derivative is what comes out of ReducedFunctional.derivative([Tic])
    """

    # Making sure all the objective/functional values are the same
    all_filenames = list(Path(__file__).parent.resolve().glob(f"{case_name}_{scheduler}_functional.dat"))
    func_vals = [x for fname in all_filenames for x in np.loadtxt(fname=fname.as_posix(), delimiter=",")]
    assert np.allclose(func_vals, func_vals[0], rtol=1e-12)

    # The filenames will tell us how many different schedulers are run
    all_filenames = list(Path(__file__).parent.resolve().glob(f"{case_name}_{scheduler}_cb_res.h5"))

    if len(all_filenames) == 1:
        # We have only tested one scheduler, no need to compare anything
        assert True
    else:
        # A placeholder dictionary to store controls and derivatives
        fields_to_be_testes = {}
        for filename in all_filenames:
            with CheckpointFile(filename.as_posix(), mode="r") as f:
                # Load the mesh (we are assuming here all the meshes are having the same partitioning)
                mesh = f.load_mesh("firedrake_default_extruded")

                # dictionary of fields for each case
                sub_fields = {}
                # Store fields
                for function_name in ["Control", "Derivative"]:
                    # Get the objective function value
                    sub_fields[function_name] = f.load_function(mesh, name=function_name)

                fields_to_be_testes[filename.with_suffix("").name] = sub_fields

        # Making sure the derivatives and controls are the same for all the schedulers
        for field_type in ["Control", "Derivative"]:
            arrays = [fields_to_be_testes[key][field_type].dat.data[:] for key in fields_to_be_testes]
            base = arrays[0]
            for i, arr in enumerate(arrays[1:], start=1):
                assert np.allclose(arr, base, rtol=1e-12,)
