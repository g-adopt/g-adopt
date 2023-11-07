import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np

from gadopt.diagnostics import domain_volume


def entrainment(composition, domain_length_x, layer_interface_y, entrainment_height):
    mesh_coords = fd.SpatialCoordinate(composition.function_space().mesh())
    return (
        fd.assemble(
            fd.conditional(mesh_coords[1] >= entrainment_height, 1 - composition, 0)
            * fd.dx
        )
        / domain_length_x
        / layer_interface_y
    )


def rms_velocity(u):
    return fd.norm(u) / fd.sqrt(domain_volume(u.ufl_domain()))


def diagnostics(simu_time, benchmark, diag_fields, variables, parameters):
    match benchmark:
        case "van_Keken_1997_isothermal" | "van_Keken_1997_thermochemical":
            diag_fields["output_time"].append(simu_time)
            diag_fields["rms_velocity"].append(rms_velocity(variables["velocity"]))
            diag_fields["entrainment"].append(
                entrainment(
                    variables["composition"],
                    parameters["domain_length_x"],
                    parameters["layer_interface_y"],
                    parameters["entrainment_height"],
                )
            )


def save_and_plot(benchmark, output_dir, diag_fields):
    match benchmark:
        case "van_Keken_1997_isothermal" | "van_Keken_1997_thermochemical":
            np.savez(f"{output_dir}/output", diag_fields=diag_fields)

            fig, ax = plt.subplots(1, 2, figsize=(18, 10), constrained_layout=True)

            ax[0].set_xlabel("Time (non-dimensional)")
            ax[1].set_xlabel("Time (non-dimensional)")
            ax[0].set_ylabel("Root-mean-square velocity (non-dimensional)")
            ax[1].set_ylabel("Entrainment (non-dimensional)")

            ax[0].plot(diag_fields["output_time"], diag_fields["rms_velocity"])
            ax[1].plot(diag_fields["output_time"], diag_fields["entrainment"])

            fig.savefig(
                f"{output_dir}/rms_velocity_and_entrainment.pdf",
                dpi=300,
                bbox_inches="tight",
            )
