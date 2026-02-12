import importlib
import sys

from doit import get_var
from doit.action import CmdAction

from functools import cache
from pathlib import Path

from typing import Any
from collections.abc import Iterator

REPO_ROOT = Path(__file__).parent
CASE_ROOTS = [
    REPO_ROOT / "demos",
    REPO_ROOT / "tests",
]

batch_mode = get_var("batch_mode", "YES")


def discover_cases():
    for root in CASE_ROOTS:
        for meta in root.rglob("meta.py"):
            case_dir = meta.parent
            yield case_dir, load_meta(meta)


def load_meta(meta_path):
    spec = importlib.util.spec_from_file_location(
        "meta", meta_path, submodule_search_locations=[],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["meta"] = mod
    spec.loader.exec_module(mod)
    del sys.modules["meta"]
    return mod


@cache
def cases():
    return list(discover_cases())

def hpcrun_command(cfg: dict[str, Any]) -> str:
    """_summary_

    _extended_summary_

    Args:
        cfg: _description_

    Returns:
        _description_
    """
    cores = cfg.get("cores", 1)
    entrypoint = cfg["hpc_entrypoint"]
    args = cfg.get("args", "")
    launcher_args = cfg.get("launcher_args")

    hpcrun_command = f"gadopt_hpcrun -n {cores} "
    if launcher_args is not None:
        hpcrun_command = f"{hpcrun_command}{launcher_args} "

    return f"{hpcrun_command}python3 {entrypoint} {args}"


def mpi_command(cfg):
    cores = cfg.get("cores", 1)
    entrypoint = cfg["entrypoint"]
    args = cfg.get("args", "")
    launcher_args=cfg.get("launcher_args")

    mpi_command = ""
    if cores > 1:
        mpi_command = f"mpiexec -np {cores} "
    if launcher_args is not None:
        mpi_command = f"{mpi_command}{launcher_args} "

    tsp_command = ""
    if batch_mode == "YES" and cfg.get("use_tsp", True):
        tsp_command = f"tsp -N {cores} -f "

    return f"{tsp_command}{mpi_command}python3 {entrypoint} {args}"


def link_dependencies(case_dir, cfg):
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


def unlink_dependencies(case_dir, cfg):
    for dep in cfg.get("deps", []):
        # notebook dependency, or serialising
        # dependency that shouldn't unlink
        if "artifact" not in dep or not dep.get("teardown", True) or not dep.get("link", True):
            continue

        dst = case_dir / dep.get("link_as", dep["artifact"])
        if dst.is_symlink():
            dst.unlink()


def make_run_task(case_dir, step, cfg, task_type=None):
    case_path = case_dir.relative_to(REPO_ROOT).as_posix()
    name = f"{case_path}:{step}"
    entrypoint = "entrypoint" if task_type is None else f"{task_type}_entrypoint"

    file_deps = []
    actions = [(link_dependencies, [case_dir, cfg])]
    if cfg[entrypoint] is not None:
        file_deps.append(case_dir / cfg[entrypoint])
        match task_type:
            case "hpc":
                actions.append(CmdAction((hpcrun_command, [cfg], {}), cwd=case_dir))
            case None:
                actions.append(CmdAction((mpi_command, [cfg], {}), cwd=case_dir))
            case _:
                raise TypeError(f"Unknown task type: {task_type}")

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


def normalise_meta(meta):

    properties = [
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
    ]

    if hasattr(meta, "steps"):
        for step_name, step_meta in meta.steps.items():
            step = {}
            for prop, default in properties:
                if default is not None:
                    step[prop] = step_meta.get(prop, default)
                elif prop in step_meta:
                    step[prop] = step_meta[prop]
            meta.steps[step_name] = step
        return meta.steps

    step = {}
    for prop, default in properties:
        if default is not None:
            step[prop] = getattr(meta, prop, default)
        elif hasattr(meta, prop):
            step[prop] = getattr(meta, prop)

    return {"run": step}


def make_convert_task(case_dir, step, cfg):
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
                cwd=case_dir
            ),
        ],
        "file_dep": file_deps,
        "targets": [notebook_file],
        "teardown": [(unlink_dependencies, [case_dir, cfg])],
    }


def task_run_case():
    for case_dir, meta in cases():
        for step, cfg in normalise_meta(meta).items():
            if "entrypoint" not in cfg:
                continue

            yield make_run_task(case_dir, step, cfg)


def task_run_case_hpc() -> Iterator[dict[str,dict[str,Any]]]:
    for case_dir, meta in cases():
        for step, cfg in normalise_meta(meta).items():
            if "hpc_entrypoint" not in cfg:
                continue

            yield make_run_task(case_dir, step, cfg, task_type="hpc")

def task_convert():
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

            outputs = [case_dir / output for output in cfg.get("notebook_outputs", []) + [cfg["notebook"]]]
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


def pytest_command(case_dir, meta):
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


def task_check():
    for case_dir, meta in cases():
        if not (cmd := pytest_command(case_dir, meta)):
            continue

        case_name = case_dir.relative_to(REPO_ROOT).as_posix()

        file_dep = []
        if hasattr(meta, "steps"):
            for cfg in meta.steps.values():
                file_dep += [case_dir / out for out in cfg["outputs"]]
        else:
            file_dep = [case_dir / out for out in meta.outputs]

        yield {
            "name": case_name,
            "actions": [cmd],
            "file_dep": file_dep,
        }


def task_mesh():
    for case_dir, meta in cases():
        if not hasattr(meta, "mesh"):
            continue

        geo = case_dir / meta.mesh["geo"]
        msh = case_dir / meta.mesh["msh"]
        args = meta.mesh.get("args", "")

        case_name = case_dir.relative_to(REPO_ROOT).as_posix()

        yield {
            "name": case_name,
            "actions": [f"gmsh -2 {args} {geo} -o {msh}"],
            "file_dep": [geo],
            "targets": [msh],
        }
