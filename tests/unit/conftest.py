"""Make the GIA helper module of `tests/gia` importable from the unit tests.

`tests/gia/gia_cases.py` holds the configurations and builders of the
self-gravitating GIA solver. The case runs in `tests/gia` and the unit tests
here both use it, so it lives with the cases and not here. pytest puts only
the directory of each test file on `sys.path`, so this file adds
`tests/gia` before any test module is imported. The unit tests then use a
plain `import gia_cases`.
"""

import sys
from pathlib import Path

# The directory of the GIA cases and of their shared helper module.
GIA_CASES_DIR = str(Path(__file__).resolve().parents[1] / "gia")

if GIA_CASES_DIR not in sys.path:
    sys.path.insert(0, GIA_CASES_DIR)
