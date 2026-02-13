import importlib
import sys

from doit import get_var
from doit.action import CmdAction

from functools import cache
from pathlib import Path

from typing import Any, cast, Final, Iterator, Sequence

from dodo_typing import CaseMeta, CaseMetaDict, DoitTask, StepsModule


REPO_ROOT = Path(__file__).parent
CASE_ROOTS = [
    REPO_ROOT / "demos",
    REPO_ROOT / "tests",
]

# "Batch mode" is intended for CI usage, where we run
# all the cases simultaneously. It would be advantageous to
# run with the `-n` flag to spawn jobs in parallel. When
# this variable is "YES", jobs will be run using `tsp`,
# otherwise there will be no slot-aware scheduling of jobs.
#
# This is a "dumb" variable, and can only be string-valued.
# Pass it on the command line as `doit batch_mode=YES ...`
# Tasks do support a `params` entry for smarter argparse-style
# arguments, but this doesn't play well with subtasks.
batch_mode = get_var("batch_mode", "YES")


def discover_cases() -> Iterator[tuple[Path, CaseMeta]]:
    """Trawl the repository for case definitions.

    This will find *all* files called "meta.py" and
    yield them for processing for finding test cases.

    Yields:
      (directory, module) pairs for further processing.

    """

    for root in CASE_ROOTS:
        for meta in root.rglob("meta.py"):
            case_dir = meta.parent
            yield case_dir, load_meta(meta)


def load_meta(meta_path: Path) -> CaseMeta:
    """Load a meta.py file by path.

    We temporarily add the module to the system collection
    when running the loader so that it is executed as its
    own package. This way, the meta file can import files
    in its directory.

    Returns:
      The module resulting from loading the meta file.

    """

    spec = importlib.util.spec_from_file_location(
        "meta",
        meta_path,
        submodule_search_locations=[],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["meta"] = mod
    spec.loader.exec_module(mod)
    del sys.modules["meta"]
    return mod


@cache
def cases() -> list[tuple[Path, CaseMeta]]:
    """Cache the case discovery.

    This is the actual entry point that should be used
    for interacting with the list of meta files, as
    it will cache the generator output.

    Returns:
      List of (directory, module) pairs.

    """

    return list(discover_cases())


def hpcrun_command(cfg: CaseMetaDict) -> str:
    """Retrieve the run command for a `run_case_hpc` task.

    This uses 4 properties from the meta dictionary:
    - `cores`: number of cores;
    - `hpc_entrypoint`: the script to run;
    - `args`: arguments to pass to the script;
    - `launcher_args`: arguments to pass to the hpcrun launcher.

    Args:
      cfg: Meta dictionary for a step.

    Returns:
      A formatted string for executing the step.

    """

    cores = cfg.get("cores", 1)
    entrypoint = cfg["hpc_entrypoint"]
    args = cfg.get("args", "")
    launcher_args = cfg.get("launcher_args")

    hpcrun_command = f"gadopt_hpcrun -n {cores} "
    if launcher_args is not None:
        hpcrun_command = f"{hpcrun_command}{launcher_args} "

    return f"{hpcrun_command}python3 {entrypoint} {args}"


def mpi_command(cfg: CaseMetaDict) -> str:
    """Retrieve the run command for a `run_case` task.

    This uses 4 properties from the meta dictionary:
    - `cores`: number of cores;
    - `entrypoint`: the script to run;
    - `args`: arguments to pass to the script;
    - `launcher_args`: arguments to pass to the launcher.

    Additionally, it also consumes the `batch_mode` command
    line flag. If this is "YES", the step will be scheduled
    through the `tsp` task spooler. Any other value will
    cause the step to be executed immediately.

    Args:
      cfg: Meta dictionary for a step.

    Returns:
      A formatted string for executing the step.

    """

    cores = cfg.get("cores", 1)
    entrypoint = cfg["entrypoint"]
    args = cfg.get("args", "")
    launcher_args = cfg.get("launcher_args")

    mpi_command = ""
    if cores > 1:
        mpi_command = f"mpiexec -np {cores} "
    if launcher_args is not None:
        mpi_command = f"{mpi_command}{launcher_args} "

    tsp_command = ""
    if batch_mode == "YES" and cfg.get("use_tsp", True):
        tsp_command = f"tsp -N {cores} -f "

    return f"{tsp_command}{mpi_command}python3 {entrypoint} {args}"


def link_dependencies(case_dir: Path, cfg: CaseMetaDict):
    """Link dependencies into a case run directory.

    Each dependency has an `artifact` within a `case` (specifying
    a directory relative to the root of the repository). Unless
    `link` is False, the artifact will be linked into the current
    directory as `link_as`. Cleanup of the dependency is the
    responsibility of the `unlink_dependencies()` method.

    Args:
      case_dir: Run directory for the case.
      cfg: Meta dictionary for the step.

    """

    for dep in cfg.get("deps", []):
        # dependency is consumed where it is, instead
        # of linking into the current directory
        if not dep.get("link", True):
            continue

        name = dep["artifact"]
        src = REPO_ROOT / dep["case"] / name
        target = dep.get("link_as", name)
        dst = case_dir / target
        if not dst.exists():
            dst.symlink_to(src)


def unlink_dependencies(case_dir: Path, cfg: CaseMetaDict):
    """Unlink dependencies from a case run directory.

    This is the counterpart to `link_dependencies()`.
    Artifacts that are linked in, where `teardown` is True
    will be unlinked after their task has completed.

    Args:
      case_dir: Run directory for the case.
      cfg: Meta dictionary for the step.

    """

    for dep in cfg.get("deps", []):
        # notebook dependency, or serialising
        # dependency that shouldn't unlink
        if (
            "artifact" not in dep
            or not dep.get("teardown", True)
            or not dep.get("link", True)
        ):
            continue

        dst = case_dir / dep.get("link_as", dep["artifact"])
        if dst.is_symlink():
            dst.unlink()


def make_run_task(
    case_dir: Path, step: str, cfg: CaseMetaDict, task_type: str | None = None
) -> DoitTask:
    """Parse a case meta step dictionary to a doit run task.

    This glues together the dependency and run actions,
    depending on the contents of the case dictionary. It also
    builds the dependency and target lists. It returns a
    doit-compatible task dictionary.

    The default None `task_type` runs directly with mpi and/or
    tsp, as configured elsewhere. `hpc` tasks are dispatched
    to a PBS scheduler.

    Args:
      case_dir: Run directory for the case.
      step: Name of the step.
      cfg: Meta dictionary for the step.
      task_type: None or "hpc" string.

    Returns:
      A doit task dictionary.

    """

    case_path = case_dir.relative_to(REPO_ROOT).as_posix()
    name = f"{case_path}:{step}"

    match task_type:
        case "hpc":
            entrypoint = cfg["hpc_entrypoint"]
        case None:
            entrypoint = cfg["entrypoint"]
        case _:
            raise TypeError(f"Unknown task type: {task_type}")

    file_deps = []
    actions = [(link_dependencies, [case_dir, cfg])]
    if entrypoint is not None:
        file_deps.append(case_dir / entrypoint)
        match task_type:
            case "hpc":
                actions.append(CmdAction((hpcrun_command, [cfg], {}), cwd=case_dir))
            case None:
                actions.append(CmdAction((mpi_command, [cfg], {}), cwd=case_dir))

    for dep in cfg.get("deps", []):
        if "artifact" in dep:
            dep_case = REPO_ROOT / dep["case"]
            file_deps.append(dep_case / dep["artifact"])

    if "mesh" in cfg:
        file_deps.append(case_dir / cfg["mesh"]["msh"])

    targets = [case_dir / out for out in cfg["outputs"]]

    return {
        "name": name,
        "actions": actions,
        "file_dep": file_deps,
        "targets": targets,
        "teardown": [(unlink_dependencies, [case_dir, cfg])],
    }


def normalise_meta(meta: CaseMeta) -> dict[str, CaseMetaDict]:
    """Convert a case meta module into a step dictionary.

    There are two ways to define a meta module: with all the
    case metadata at the toplevel; or with the metadata in a
    steps dictionary.

    For a module where all the metadata is at the toplevel,
    this will normalise to a single step named `run`. All steps
    will be filtered to a set of standard properties that
    are expected to exist.

    This is a bit fiddly typewise, because we can't assume
    that the data coming from meta.py is valid yet. Given the
    enforcement of this method, we can cast the result to a
    map to `CaseMetaDict`.

    Args:
      meta: A case meta module from `load_meta()`.

    Returns:
      A dictionary mapping step names to their metadata.

    """

    properties: Sequence[tuple[str, Any]] = (
        ("entrypoint", None),
        ("hpc_entrypoint", None),
        ("notebook", None),
        ("notebook_outputs", None),
        ("mesh", None),
        ("args", None),
        ("launcher_args", None),
        ("cores", 1),
        ("outputs", []),
        ("deps", []),
        ("use_tsp", True),
    )

    if hasattr(meta, "steps"):
        meta = cast(StepsModule, meta)
        for step_name, step_meta in meta.steps.items():
            step = {}
            for prop, default in properties:
                if default is not None:
                    step[prop] = step_meta.get(prop, default)
                elif prop in step_meta:
                    step[prop] = step_meta[prop]
            meta.steps[step_name] = step
        return cast(dict[str, CaseMetaDict], meta.steps)

    step = {}
    for prop, default in properties:
        if default is not None:
            step[prop] = getattr(meta, prop, default)
        elif hasattr(meta, prop):
            step[prop] = getattr(meta, prop)

    return {"run": cast(CaseMetaDict, step)}


def make_convert_task(case_dir: Path, step: str, cfg: CaseMetaDict) -> DoitTask:
    """Parse a case meta step dictionary to a doit convert task.

    This pulls out dependencies in the same way as the
    run task. However, we expect to be running all the notebooks
    in the same context, and don't want to start a "regular"
    run task to generate the dependency. Instead, we provide the
    output `.ipynb` notebook file as the `file_dep` to doit.

    In the simplest case, the meta only needs a `notebook` entry,
    which will be generated from the Python script of the same
    basename.

    Args:
      case_dir: Run directory for the case.
      step: Name of the step.
      cfg: Meta dictionary for the step.

    Returns:
      A doit task dictionary.

    """

    case_path = case_dir.relative_to(REPO_ROOT).as_posix()
    name = f"{case_path}:{step}"

    notebook_file = case_dir / cfg["notebook"]
    py_file = notebook_file.with_suffix(".py")

    file_deps = [py_file]
    for dep in cfg.get("deps", []):
        if "notebook" in dep:
            dep_case = REPO_ROOT / dep["case"]
            file_deps.append(dep_case / dep["notebook"])

    return {
        "name": name,
        "actions": [
            (link_dependencies, [case_dir, cfg]),
            f"python3 -m jupytext --to ipynb {py_file}",
            CmdAction(
                "jupyter-nbconvert --to notebook --execute --inplace "
                """--TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags='["exercise"]' """
                f"{notebook_file}",
                cwd=case_dir,
            ),
        ],
        "file_dep": file_deps,
        "targets": [notebook_file],
        "teardown": [(unlink_dependencies, [case_dir, cfg])],
    }


def task_run_case() -> Iterator[DoitTask]:
    """Top level doit run_case task.

    Every meta that contains an `entrypoint` is considered
    for this task.

    Yields:
      run_case subtasks.

    """

    for case_dir, meta in cases():
        for step, cfg in normalise_meta(meta).items():
            if "entrypoint" not in cfg:
                continue

            yield make_run_task(case_dir, step, cfg)


def task_run_case_hpc() -> Iterator[DoitTask]:
    """Top level doit run_case_hpc task.

    This is the HPC-specific counterpart to the run_case task:
    metas with `hpc_entrypoint` are considered.

    Yields:
      run_case_hpc subtasks.

    """

    for case_dir, meta in cases():
        for step, cfg in normalise_meta(meta).items():
            if "hpc_entrypoint" not in cfg:
                continue

            yield make_run_task(case_dir, step, cfg, task_type="hpc")


def task_convert() -> Iterator[DoitTask]:
    """Top level doit convert task.

    This task runs the full notebook-to-archive workflow:
    all metas that define a `notebook` are converted from `.py`
    to `.ipynb` and executed, then placed in an archive with
    the contents directory and tutorial relationship flowcharts.

    Yields:
      convert subtasks.

    """

    demo_path = REPO_ROOT / "demos"

    diagrams = list(demo_path.glob("**/.diagram.mermaid"))
    diagram_paths = " ".join(str(d.relative_to(demo_path)) for d in diagrams)

    actions = [
        f"tar --transform='s/.pages/CONTENTS.md/' --create --file artifact.tar --directory demos .pages {diagram_paths}",
    ]

    notebook_deps = []
    for case_dir, meta in cases():
        for step, cfg in normalise_meta(meta).items():
            if "notebook" not in cfg:
                continue

            outputs = [
                case_dir / output
                for output in cfg.get("notebook_outputs", []) + [cfg["notebook"]]
            ]
            notebook_deps.extend(outputs)
            paths = " ".join(str(o.relative_to(demo_path)) for o in outputs)
            actions.append(
                f"tar --transform='s|/.*/|/|' --append --file artifact.tar --directory demos {paths}"
            )

            yield make_convert_task(case_dir, step, cfg)

    yield {
        "name": "convert",
        "actions": actions,
        "file_dep": notebook_deps + [demo_path / ".pages"] + diagrams,
        "targets": [REPO_ROOT / "artifact.tar"],
    }


def pytest_command(case_dir: Path, meta: CaseMeta) -> str | None:
    """Determine the test strategy for a meta file.

    The top-level `pytest` attribute can have a few values:
    - `auto`: uses the `test_all` harness with the case name;
    - `local`: uses `test_*` files in the case directory;
    - other strings are used directly;
    - None indicates that this case shouldn't be part
      of the `check` task.

    Args:
      case_dir: Run directory for the case.
      meta: Meta module for the case.

    Returns:
      A check command to run, or None if the case shouldn't
      be part of the `check` task.

    """

    match getattr(meta, "pytest", None):
        case None:
            return None
        case "auto":
            test_file = REPO_ROOT / "demos/test_all.py"
            # TODO: when support for Python <3.12 is dropped replace the next 4 lines with
            #       test_name = case_dir.relative_to(REPO_ROOT / "demos", walk_up=True)
            if case_dir.is_relative_to(REPO_ROOT / "demos"):
                test_name = case_dir.relative_to(REPO_ROOT / "demos")
            else:
                test_name = Path("../tests") / case_dir.relative_to(REPO_ROOT / "tests")
            return f"python3 -m pytest {test_file} -k {test_name}"
        case "local":
            return f"python3 -m pytest {case_dir}"
        case other:
            return other


def task_check() -> Iterator[DoitTask]:
    """Top level doit check task.

    This generates subtasks for each case directory, which
    will run the tests for the case, then call pytest (or
    a custom test command). It probably shouldn't be used
    for whole-repository testing, because it will run
    individual pytest instances per-directory. For this
    use-case, use the `run_case` task and `pytest` directory.

    Yields:
      check subtasks.

    """

    for case_dir, meta in cases():
        if not (cmd := pytest_command(case_dir, meta)):
            continue

        case_name = case_dir.relative_to(REPO_ROOT).as_posix()

        file_dep = []
        if hasattr(meta, "steps"):
            meta = cast(StepsModule, meta)
            for cfg in meta.steps.values():
                file_dep += [case_dir / out for out in cfg["outputs"]]
        else:
            file_dep = [case_dir / out for out in meta.outputs]

        yield {
            "name": case_name,
            "actions": [cmd],
            "file_dep": file_dep,
        }


def task_mesh() -> Iterator[DoitTask]:
    """Top level doit mesh task.

    For cases that require mesh creation, this task
    is added as a dependency. It is requested when a task
    has a `mesh` entry, which should define a `geo` input
    file and a `msh` output file (relative to the case
    directory). Arguments can be passed through to gmsh
    with `args`.

    Yields:
      mesh subtasks.

    """

    for case_dir, meta in cases():
        for step, cfg in normalise_meta(meta).items():
            if "mesh" not in cfg:
                continue

            geo = case_dir / cfg["mesh"]["geo"]
            msh = case_dir / cfg["mesh"]["msh"]
            args = cfg["mesh"].get("args", "")

            case_path = case_dir.relative_to(REPO_ROOT).as_posix()
            name = f"{case_path}:{step}"

            yield {
                "name": name,
                "actions": [f"gmsh -2 {args} {geo} -o {msh}"],
                "file_dep": [geo],
                "targets": [msh],
            }
