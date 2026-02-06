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

def mpi_command(case_dir, meta):
    cores = meta.cores
    mpi_command = ""
    if cores > 1:
        mpi_command = f"mpiexec -np {cores} "

    return (
        f"tsp -N {cores} -f "
        f"{mpi_command}"
        f"python3 {meta.entrypoint}"
    )

def link_dependencies(case_dir, meta):
    for dep in getattr(meta, "deps", []):
        src = REPO_ROOT / dep
        dst = case_dir / dep.name
        if not dst.exists():
            dst.symlink_to(src)

def unlink_dependencies(case_dir, meta):
    for dep in getattr(meta, "deps", []):
        dst = case_dir / dep.name
        if dst.is_symlink():
            dst.unlink()

def task_run_case():
    for case_dir, meta in discover_cases():
        case_name = case_dir.relative_to(REPO_ROOT).as_posix()

        file_deps = [REPO_ROOT / meta.entrypoint]
        for dep in getattr(meta, "deps", []):
            file_deps.append(REPO_ROOT / dep)

        targets = [case_dir / out for out in meta.outputs]

        yield {
            "name": case_name,
            "actions": [
                (link_dependencies, [case_dir, meta]),
                CmdAction(mpi_command(case_dir, meta), cwd=case_dir),
            ],
            "file_dep": file_deps,
            "targets": targets,
            "clean": [(unlink_dependencies, [case_dir, meta])],
        }
