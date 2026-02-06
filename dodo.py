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
        dst = case_dir / name
        if not dst.exists():
            dst.symlink_to(src)

def unlink_dependencies(case_dir, cfg):
    for dep in cfg.get("deps", []):
        dst = case_dir / dep["artifact"]
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

def run_single(case_dir, meta):
    cfg = {
        "entrypoint": meta.entrypoint,
        "cores": getattr(meta, "cores", 1),
        "outputs": meta.outputs,
        "deps": getattr(meta, "deps", []),
    }
    return make_run_task(case_dir, "run", cfg)

def task_run_case():
    for case_dir, meta in discover_cases():
        if hasattr(meta, "steps"):
            for step, cfg in meta.steps.items():
                yield make_run_task(case_dir, step, cfg)
        else:
            yield run_single(case_dir, meta)
