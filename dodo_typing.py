from pathlib import Path
from typing import Any, Protocol, Sequence, TypedDict


class MeshConfigDict(TypedDict):
    geo: str  # name of the input file
    msh: str  # name of the output file
    args: str  # arguments to pass through to gmsh


class DepConfigDictBase(TypedDict):
    case: str  # Name of the case containing dependency
    step: str  # Step that generates the dependency
    artifact: str  # Dependency filename

class DepConfigDict(DepConfigDictBase, total=False):
    link_as: str  # Link name
    link: bool  # Whether to link the dep into the running case
    teardown: bool  # Whether to teardown the link
    notebook: str  # Notebook that generates the dependency


class CaseMetaModule(Protocol):
    entrypoint: str
    hpc_entrypoint: str
    notebook: str
    notebook_outputs: list[str]
    mesh: MeshConfigDict
    args: list[str]
    launcher_args: list[str]
    cores: int
    outputs: Sequence[str]
    deps: Sequence[DepConfigDict]
    pytest: str


class CaseMetaDict(TypedDict):
    entrypoint: str
    hpc_entrypoint: str
    notebook: str
    notebook_outputs: list[str]  # We do list operations on this
    mesh: MeshConfigDict
    args: Sequence[str]
    launcher_args: Sequence[str]
    cores: int
    outputs: Sequence[str]
    deps: Sequence[DepConfigDict]


class StepsModule(Protocol):
    # The enforcement of this as a map to CaseMetaDict
    # only happens in normalise_meta().
    steps: dict[str, Any]
    pytest: str


CaseMeta = CaseMetaModule | StepsModule


class DoitTaskBase(TypedDict):
    name: str
    actions: list

class DoitTask(DoitTaskBase, total=False):
    file_dep: Sequence[str | Path]
    targets: Sequence[str | Path]
    teardown: list | None
