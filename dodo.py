import importlib
from doit.action import CmdAction
from pathlib import Path

REPO_ROOT = Path(__file__).parent
CASE_ROOTS = [
    REPO_ROOT / "demos",
    REPO_ROOT / "tests",
]

def discover_cases():
    for root in CASE_ROOTS:
        for meta in root.rglob("meta.py"):
            case_dir = meta.parent
            yield case_dir, load_meta(meta)

def load_meta(meta_path):
    spec = importlib.util.spec_from_file_location(
        "case_meta", meta_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def mpi_command(cfg):
    cores = cfg.get("cores", 1)
    entrypoint = cfg["entrypoint"]
    mpi_command = ""
    if cores > 1:
        mpi_command = f"mpiexec -np {cores} "

    return (
        f"tsp -N {cores} -f "
        f"{mpi_command}"
        f"python3 {entrypoint}"
    )

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
        dst = case_dir / dep.get("link_as", dep["artifact"])
        if dst.is_symlink():
            dst.unlink()

def make_run_task(case_dir, step, cfg):
    case_path = case_dir.relative_to(REPO_ROOT).as_posix()
    name = f"{case_path}:{step}"

    file_deps = [case_dir / cfg["entrypoint"]]
    for dep in cfg.get("deps", []):
        dep_case = REPO_ROOT / dep["case"]
        dep_step = dep.get("step")
        file_deps.append(dep_case / dep["artifact"])

    targets = [case_dir / out for out in cfg["outputs"]]

    return {
        "name": name,
        "actions": [
            (link_dependencies, [case_dir, cfg]),
            CmdAction(mpi_command(cfg), cwd=case_dir),
        ],
        "file_dep": file_deps,
        "targets": targets,
        "teardown": [(unlink_dependencies, [case_dir, cfg])],
    }

def normalise_meta(meta):
    if hasattr(meta, "steps"):
        return meta.steps

    return {
        "run": {
            "entrypoint": meta.entrypoint,
            "cores": getattr(meta, "cores", 1),
            "outputs": meta.outputs,
            "deps": getattr(meta, "deps", []),
        }
    }

def make_convert_task(case_dir, meta):
    case_name = case_dir.relative_to(REPO_ROOT).as_posix()

    file_deps = [case_dir / meta.notebook.with_suffix(".py")]
    notebook_file = case_dir / meta.notebook

    steps = normalise_meta(meta)
    if "run" in steps:
        dep_step = [(link_dependencies, [case_dir, steps["run"]])]
        undep_step = [(unlink_dependencies, [case_dir, steps["run"]])]
    else:
        dep_step = []
        undep_step = []

    return {
        "name": case_name,
        "actions": dep_step + [
            f"jupytext --to ipynb {notebook_file}",
            CmdAption(
                "jupyter-nbconvert --to notebook --execute --inplace "
                """--TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags='["exercise"]'"""
                f"{notebook_file}",
                cwd=case_dir
            ),
        ],
        "file_dep": file_deps,
        "targets": [notebook_file],
        "teardown": undep_step,
    }

def task_run_case():
    for case_dir, meta in discover_cases():
        for step, cfg in normalise_meta(meta).items():
            yield make_run_task(case_dir, step, cfg)

def pytest_command(case_dir, meta):
    match getattr(meta, "pytest", None):
        case None:
            return None
        case "auto":
            test_file = REPO_ROOT / "demos/test_all.py"
            test_name = Path(case_dir.parts[-2])
            return f"python3 -m pytest {test_file} -k {test_name}"
        case "local":
            return f"python3 -m pytest {case_dir}"
        case other:
            return other

def task_check():
    for case_dir, meta in discover_cases():
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
    for case_dir, meta in discover_cases():
        if not hasattr(meta, "mesh"):
            continue

        geo = case_dir / meta.mesh["geo"]
        msh = case_dir / meta.mesh["msh"]

        case_name = case_dir.relative_to(REPO_ROOT).as_posix()

        yield {
            "name": case_dir.name,
            "actions": [f"gmsh -3 {geo} -o {msh}"],
            "file_dep": [geo],
            "targets": [msh],
        }

def task_convert():
    for case_dir, meta in discover_cases():
        if not hasattr(meta, "notebook"):
            continue

        yield make_convert_task(case_dir, meta)
