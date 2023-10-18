Regression Testing
==================

The suite of benchmarks can be automatically regression-tested against
a set of "expected" results. This file gives some instructions for how
to add or modify a tested case.

Organisation of test cases
--------------------------

All the test cases are organised under the `demos/` directory in the
G-ADOPT repository. These serve to simultaneously demonstrate how to
use the G-ADOPT library, and test the functionality being shown. Most
of these demos are set up such that they can simply be run using
Python under a Firedrake environment into which G-ADOPT is
installed. Some of the demos are larger or more computationally
expensive, and may benefit from being run under MPI.

Within the set of test cases, there are two regimes of testing. Most
of the cases form the regular regression suite, which is run on all
pull requests and merges into the master branch of G-ADOPT. The full
regression suite, executed in parallel should take around 2 hours to
complete.

There are a couple of test cases which have substantially larger
computational requirements than the standard suite, and must be
executed on HPC. These are the long tests, which are only triggered
manually.

Testing infrastructure
----------------------

The testing infrastructure makes use of a couple of Makefiles located
alongside the demos. All the main configuration takes place in
`demos/Makefile`, which is usually called from the top-level Makefile
by either `make -j test` or `make -j longtest`, depending on which suite is
being run.

In all the Makefiles, we consider the default behaviour to be to run
the regression suite. The test cases which make up the suite are
defined as `cases` within `demos/Makefile`, and similarly the long
tests are `long_cases`. These variables should be adjusted as
necessary when adding or removing test cases from the corresponding
suites.

The Makefiles should be written such that when `make` is executed for
a given test case, it waits for the test case to complete and generate
all relevant output. Because there are several tests with different
requirements on the number of cores they occupy, *task spooler*
(called by the `tsp` program) is used as a mini batch system to
orchestrate the parallel execution of the tests. The upshot of this is
that `make -j` can and should be used, with at least as many jobs as
there are tests that will be executed. This defers all the scheduling
to *task spooler*, and will only exit when all the cases have
completed. Additionally, *task spooler* is responsible for storing the
standard output/error of the jobs, which makes it much easier to
handle the vast output for several jobs running in parallel.

Once all the test cases have run, they should each write some kind of
file to disk containing metrics that can be tested. The actual
regression tests that define whether a test run is acceptable are
written in Python, using the *pytest* framework. On this level, the
tests are delineated by the `longtest` mark: the regular regression
suite is run with `pytest -m 'not longtest'`, and conversely the long
tests can be tested using `pytest -m longtest`.

Convergence tests
-----------------

Generating expected results
---------------------------

Once a benchmark has been run, it should be validated by hand to
ensure it is suitable for recording as a reference result. The
`generate_expected_out.py` can be run to record `u_rms` and `nu_top`
fields from the last timestep in `params.log` as the reference result,
for later use in the test suite.

Manually running a test
-----------------------

Automatic running through CI
----------------------------

This entire process is automatically captured in a GitHub Actions
workflow, which will use the latest Firedrake Docker container, run
all the tests, and then ensure their results match those which have
been checked in.

**Important**: Because some tests may be expensive to run, they can be
disabled in two places:

1. To stop the results from being generated for a given case, it can
   be removed from the Makefile hierarchy. e.g. only the 2D cases may
   be actually run, even though Makefiles exist for all benchmarks.

2. Once a test isn't expected to be run through CI, it should also be
   deselected through pytest's `-k` flag in the workflow definition
   file.

Long test framework
-------------------