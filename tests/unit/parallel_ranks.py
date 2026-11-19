"""Check that a test with a `parallel` marker runs on the ranks it asks for.

The `parallel(nprocs=...)` marker comes from the `mpi-pytest` plugin. With
the plugin, a plain `pytest` run starts each marked test under `mpiexec` with
the requested number of ranks, and a list of counts gives one test per count
(the plugin's `_nprocs` parameter). Without the plugin, pytest only warns
about an unknown marker and runs the test once on one rank, where a test
written for two ranks passes without testing what it claims. Each marked
test calls `assert_marked_rank_count` first, so that this case fails.

The check sits in the test body, not in a fixture: in a plain `pytest` run
the plugin sets up the fixtures of a marked test in the one-rank process
that starts `mpiexec`, and only the test body runs on the requested ranks.
"""
from mpi4py import MPI


def assert_marked_rank_count(request) -> None:
    """Fail unless `COMM_WORLD` has the number of ranks of the test's `parallel` marker.

    Args:
      request: the pytest `request` fixture of the running test.

    Raises:
      AssertionError: when the test has no `parallel` marker, when a list of
        counts was not split into one test per count (the plugin is not
        active), or when the number of ranks differs from the requested one.
    """
    node = request.node
    marker = node.get_closest_marker("parallel")
    assert marker is not None, f"{node.name} has no parallel marker"
    callspec = getattr(node, "callspec", None)
    if callspec is not None and "_nprocs" in callspec.params:
        # The plugin split a list of counts into one test per count.
        requested = callspec.params["_nprocs"]
    else:
        # The plugin's reading of the marker: a positional count, an
        # `nprocs` keyword, or no count, which means its default of 3.
        counts = marker.kwargs.get("nprocs", marker.args[0] if marker.args else 3)
        counts = tuple(counts) if isinstance(counts, (list, tuple)) else (counts,)
        assert len(counts) == 1, (
            f"{node.name} asks for {counts} ranks, but no test was made per "
            "count; the mpi-pytest plugin is not active")
        requested, = counts
    size = MPI.COMM_WORLD.size
    assert size == requested, (
        f"{node.name} asks for {requested} rank(s) and runs on {size}; "
        "without the mpi-pytest plugin a parallel test runs on one rank")
