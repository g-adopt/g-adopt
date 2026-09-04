"""Doc-lint: every name a class docstring cites in ``backticks`` must exist.

A docstring that names a function nobody can call is a bug no test catches,
because nothing executes prose. ``GplatesScalarFunction`` carried one for two
releases: it advertised four convenience factories that had been replaced
before the branch was cut, so the names reached ``main`` attached to nothing.

This check reads the class docstrings in ``gadopt.gplates.gplates``, pulls out
every double-backticked token that looks like an identifier, and asserts each
one resolves — as a package export, a module-level name, or an attribute of the
class doing the citing. Trailing call syntax is stripped, so
``update_plate_reconstruction(ndtime)`` is checked as the method name.

Scope is deliberately one module. Widened to the whole package the same rule
produces five false positives, all of them names that legitimately exist
without being attributes of anything importable:

    interpolation.InterpolationConfig ``outside_source_range``  a geometry key
    connectors.ScalarFieldConnector   ``gc_collect_frequency``  a constructor argument
    connectors.ScalarFieldConnector   ``None``                  a keyword
    connectors.ScalarFieldConnector   ``step_to``               a gtrack method
    outputs.OutputStrategy            ``requires``              an annotation-only ClassVar

Resolving those needs a namespace built from parameter lists, annotations and
imported third-party attributes, which is a great deal more machinery than the
defect warrants. Anyone widening the scope should expect to write it, and
should start from this list.
"""

import ast
import re
from pathlib import Path

import gadopt.gplates
from gadopt.gplates import gplates as gplates_module


BACKTICKED = re.compile(r"``([^`]+)``")
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def cited_names(docstring):
    """Extract the identifier-shaped names a docstring cites in backticks.

    Args:
        docstring: the docstring text.

    Returns:
        A list of ``(raw_token, name)`` pairs, where ``name`` is the token with
        any trailing call syntax removed. Tokens that are not identifier-shaped
        (quoted strings, expressions, prose) are dropped.
    """
    pairs = []
    for raw in BACKTICKED.findall(docstring):
        name = raw.split("(")[0].strip()
        if IDENTIFIER.match(name):
            pairs.append((raw, name))
    return pairs


def resolves(name, owner):
    """Report whether ``name`` refers to something that exists.

    Args:
        name: the cited identifier.
        owner: the class whose docstring cited it, or None if it is not
            importable under that name.

    Returns:
        True if the name is a package export, a module-level name in
        ``gadopt.gplates.gplates``, or an attribute of ``owner``.
    """
    return (
        hasattr(gadopt.gplates, name)
        or hasattr(gplates_module, name)
        or (owner is not None and hasattr(owner, name))
    )


class TestClassDocstringReferences:
    def test_every_cited_name_exists(self):
        source = Path(gplates_module.__file__).read_text()
        dangling = []
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.ClassDef):
                continue
            docstring = ast.get_docstring(node)
            if not docstring:
                continue
            owner = getattr(gplates_module, node.name, None)
            for raw, name in cited_names(docstring):
                if not resolves(name, owner):
                    dangling.append(f"{node.name} cites ``{raw}`` — no such name")
        assert not dangling, "\n".join(dangling)

    def test_the_check_can_fail(self):
        # A doc-lint that cannot fail is decoration. This is the shape of the
        # defect it exists to catch: a class docstring naming a function that
        # was removed before the branch was cut.
        assert not resolves("lithosphere_indicator", gplates_module.GplatesScalarFunction)
        assert not resolves("polygon_geotherm", gplates_module.GplatesScalarFunction)

    def test_it_accepts_the_names_that_do_exist(self):
        cls = gplates_module.GplatesScalarFunction
        assert resolves("ScalarFieldConnector", cls)
        assert resolves("Source", cls)
        assert resolves("OutputStrategy", cls)
        assert resolves("update_plate_reconstruction", cls)
