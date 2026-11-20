"""Copy driver outputs of the GIA cases to `expected/`, with a provenance header.

Usage: `python gia_record_expected.py --case <case> [--entry <entry>]`.

For each recorded file of the case (`gia_entries.RECORDED`), or of one entry
of it, the output in `out/` is copied to `expected/` with a header of
comment lines (`#`): the g-adopt commit, the Firedrake commit, the PETSc
version, the host, the date and the command. The test files read the
expected files with `#` as the comment character.

Record an entry once, on the development machine, after its values were
reviewed. A later change of an expected file needs a line in the commit
message that names the cause. Do not record on the CI runner to make a test
pass.
"""

import argparse
import datetime
import platform
import subprocess
import sys
from pathlib import Path

from gia_entries import CASE_DIR, RECORDED


def _git_commit(directory, paths=(".",)):
    """The commit of the git checkout that holds `directory`, or "unknown".

    " (with local changes)" is added when a file under `paths` (relative to
    the root of the checkout) is modified or untracked, so that a recording
    from code that is not in the named commit says so."""
    try:
        out = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "-C", str(directory), "status", "--porcelain",
                                "--", *paths],
                               capture_output=True, text=True, check=True)
        return out.stdout.strip() + (" (with local changes)" if dirty.stdout.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


#: The paths whose state the g-adopt commit line describes: the library and
#: the cases, but not the expected files that the recording writes.
SOURCE_PATHS = (":(top)gadopt", ":(top)tests/gia", ":(top,exclude)tests/gia/expected")


def header():
    """The provenance lines of an expected file, each starting with `# `."""
    import firedrake
    from petsc4py import PETSc
    petsc = ".".join(str(v) for v in PETSc.Sys.getVersion())
    lines = [f"g-adopt commit: {_git_commit(CASE_DIR, SOURCE_PATHS)}",
             f"Firedrake commit: {_git_commit(Path(firedrake.__file__).parent)}",
             f"PETSc version: {petsc}",
             f"host: {platform.node()}",
             f"date: {datetime.datetime.now().isoformat(timespec='seconds')}",
             f"command: python {' '.join(sys.argv)}"]
    return "".join(f"# {line}\n" for line in lines)


def record(case, entry=None):
    """Copy the outputs of `case` (or of its `entry`) to their expected files."""
    pairs = RECORDED[case]
    names = [entry] if entry is not None else sorted(pairs)
    text = header()
    for name in names:
        source, target = (CASE_DIR / path for path in pairs[name])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + source.read_text())
        print(f"recorded {source.relative_to(CASE_DIR)} -> {target.relative_to(CASE_DIR)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", required=True, choices=sorted(RECORDED))
    parser.add_argument("--entry")
    args = parser.parse_args()
    record(args.case, args.entry)
