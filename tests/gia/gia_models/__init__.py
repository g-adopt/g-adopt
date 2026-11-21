"""Builders, meshes and constants that the GIA case drivers and the unit tests share.

Each module holds the helpers of one file of `tests/unit`. The drivers in
`tests/gia` import them as `from gia_models import <module>`, and the unit
tests import them from here too (`tests/unit/conftest.py` puts `tests/gia` on
`sys.path`).
"""
