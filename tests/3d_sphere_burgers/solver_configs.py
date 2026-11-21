"""Names and option-dictionary layout of the Burgers solver configurations.

This module imports nothing from Firedrake or G-ADOPT, so that the scripts
which only read job output (``tests/parallel_scaling_burgers/burgers_scaling.py`` and
``plot_breakdown.py``) can import it without a Firedrake environment. The
solvers themselves are built in ``coupled_solver_variants.py``.
"""

CONFIGURATIONS = (
    "substituted",
    "multiplicative",
    "schur-a11",
    "schur-substituted",
    "static-condensation",
)
"""Every configuration the comparison knows. ``substituted`` is the reference."""

COUPLED_CONFIGURATIONS = tuple(c for c in CONFIGURATIONS if c != "substituted")
"""The coupled configurations, each measured against the reference."""


def displacement_block_key(config):
    """Key of the displacement block in the solver's option dictionary.

    ``None`` means the top level: the substituted solver has no fieldsplit,
    so its Krylov and GAMG options sit directly in the dictionary. The
    other values are the prefixes under which `StokesSolverBase` nests the
    displacement options for each layout (see ``displacement_block_prefix``
    on that class).
    """
    if config == "substituted":
        return None
    if config == "multiplicative":
        return "fieldsplit_0"
    if config.startswith("schur-"):
        return "fieldsplit_1"
    if config == "static-condensation":
        return "condensed_field"
    raise ValueError(f"Unknown configuration {config!r}")


def displacement_ksp_prefix(config, solver_name="CoupledInternalVariable"):
    """PETSc options prefix of the KSP whose iterations are GAMG V-cycles.

    This is the prefix that appears in ``-ksp_converged_reason`` lines, so it
    is what the output parsers match on. ``solver_name`` is the
    ``options_prefix`` of the coupled solver class.
    """
    if config == "substituted":
        return "InternalVariable_"
    return f"{solver_name}_{displacement_block_key(config)}_"
