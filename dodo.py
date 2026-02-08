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

    tsp_command = ""
    if cfg.get("use_tsp", True):
        tsp_command = f"tsp -N {cores} -f "

    return f"{tsp_command}{mpi_command}python3 {entrypoint}"

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
        if "artifact" not in dep:
            continue

        dst = case_dir / dep.get("link_as", dep["artifact"])
        if dst.is_symlink():
            dst.unlink()

def make_run_task(case_dir, step, cfg):
    case_path = case_dir.relative_to(REPO_ROOT).as_posix()
    name = f"{case_path}:{step}"

    file_deps = [case_dir / cfg["entrypoint"]]
    for dep in cfg.get("deps", []):
        if "artifact" in dep:
            dep_case = REPO_ROOT / dep["case"]
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

    properties = [
        ("entrypoint", None),
        ("notebook", None),
        ("cores", 1),
        ("outputs", []),
        ("deps", []),
    ]

    step = {}
    for prop, default in properties:
        if default is not None:
            step[prop] = getattr(meta, prop, default)
        elif hasattr(meta, prop):
            step[prop] = getattr(meta, prop)

    return { "run": step }

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
                "python3 -m nbconvert --to notebook --execute --inplace "
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
    for case_dir, meta in discover_cases():
        for step, cfg in normalise_meta(meta).items():
            if "entrypoint" not in cfg:
                continue

            yield make_run_task(case_dir, step, cfg)

def task_convert():
    for case_dir, meta in discover_cases():
        for step, cfg in normalise_meta(meta).items():
            if "notebook" not in cfg:
                continue

            yield make_convert_task(case_dir, step, cfg)

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
