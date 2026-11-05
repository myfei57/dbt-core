from argparse import Namespace
from copy import deepcopy

import pytest

import dbt.compilation
import dbt.graph.cli as graph_cli
from dbt.contracts.selection import SelectorFile
from dbt.flags import set_from_args
from dbt.graph import NodeSelector, parse_difference
from dbt.graph.selector_methods import BoundarySelectorMethod, MethodName
from dbt.graph.selector_spec import IndirectSelection, SelectionCriteria
from dbt_common.exceptions import DbtRuntimeError
from tests.unit.utils.manifest import make_generic_test, make_manifest, make_model

set_from_args(Namespace(WARN_ERROR=False), None)


def _model(name, refs=None, tags=None):
    return make_model(
        pkg="pkg",
        name=name,
        code=f"select 1 as {name}",
        refs=refs or [],
        tags=tags or [],
    )


@pytest.fixture
def dag(runtime_config):
    # Chain:      a -> b -> c -> d
    # Fork:      froot -> fleft, froot -> fright
    # Outside:   edge_start -> mid -> edge_end  (mid is never tag-selected)
    # Isolated:  lone
    a = _model("a", tags=["chain", "all"])
    b = _model("b", refs=[a], tags=["chain", "all"])
    c = _model("c", refs=[b], tags=["chain", "all"])
    d = _model("d", refs=[c], tags=["chain", "all"])

    froot = _model("froot", tags=["fork", "all"])
    fleft = _model("fleft", refs=[froot], tags=["fork", "all"])
    fright = _model("fright", refs=[froot], tags=["fork", "all"])

    edge_start = _model("edge_start", tags=["edge", "all"])
    mid = _model("mid", refs=[edge_start], tags=["all"])
    edge_end = _model("edge_end", refs=[mid], tags=["edge", "all"])

    lone = _model("lone", tags=["solo", "all"])

    test_a = make_generic_test("pkg", "unique", a, {}, refs=[a], column_name="id")
    test_b = make_generic_test("pkg", "unique", b, {}, refs=[b], column_name="id")
    test_d = make_generic_test("pkg", "unique", d, {}, refs=[d], column_name="id")

    models = [a, b, c, d, froot, fleft, fright, edge_start, mid, edge_end, lone]
    manifest = make_manifest(nodes=models + [test_a, test_b, test_d])

    compiler = dbt.compilation.Compiler(runtime_config)
    graph = compiler.compile(manifest)
    node_selector = NodeSelector(graph, manifest)

    ids = {node.name: node.unique_id for node in models}
    ids.update(
        {
            "test_a": test_a.unique_id,
            "test_b": test_b.unique_id,
            "test_d": test_d.unique_id,
        }
    )
    return node_selector, manifest, ids


def _boundary_spec(direction, definition):
    return SelectionCriteria(
        raw=f"boundary:{direction}",
        method=MethodName.Boundary,
        method_arguments=[],
        value={"direction": direction, "definition": definition},
        childrens_parents=False,
        parents=False,
        parents_depth=None,
        children=False,
        children_depth=None,
    )


def _select(node_selector, spec):
    direct, _ = node_selector.select_nodes(spec)
    return direct


class TestBoundarySelection:
    def test_chain_upstream_and_downstream(self, dag):
        node_selector, _, ids = dag
        spec = _boundary_spec("upstream", {"tag": "chain"})
        assert node_selector.get_direct_selection(spec) == {ids["a"]}

        spec = _boundary_spec("downstream", {"tag": "chain"})
        assert node_selector.get_direct_selection(spec) == {ids["d"]}

    def test_fork_boundaries(self, dag):
        node_selector, _, ids = dag
        assert node_selector.get_direct_selection(_boundary_spec("upstream", {"tag": "fork"})) == {
            ids["froot"]
        }
        assert node_selector.get_direct_selection(
            _boundary_spec("downstream", {"tag": "fork"})
        ) == {ids["fleft"], ids["fright"]}

    def test_nodes_outside_selection_do_not_bridge(self, dag):
        node_selector, _, ids = dag
        # edge_start -> mid -> edge_end, but `mid` is outside the selection.
        # Neither endpoint has an in-selection neighbor, so both are on both
        # boundaries (the missing middle must not be bridged through).
        expected = {ids["edge_start"], ids["edge_end"]}
        assert (
            node_selector.get_direct_selection(_boundary_spec("upstream", {"tag": "edge"}))
            == expected
        )
        assert (
            node_selector.get_direct_selection(_boundary_spec("downstream", {"tag": "edge"}))
            == expected
        )

    def test_isolated_node_is_both_boundaries(self, dag):
        node_selector, _, ids = dag
        assert node_selector.get_direct_selection(_boundary_spec("upstream", {"tag": "solo"})) == {
            ids["lone"]
        }
        assert node_selector.get_direct_selection(
            _boundary_spec("downstream", {"tag": "solo"})
        ) == {ids["lone"]}

    def test_isolated_node_mixed_into_larger_selection(self, dag):
        node_selector, _, ids = dag
        assert node_selector.get_direct_selection(_boundary_spec("upstream", {"tag": "all"})) == {
            ids["a"],
            ids["froot"],
            ids["edge_start"],
            ids["lone"],
        }
        assert node_selector.get_direct_selection(
            _boundary_spec("downstream", {"tag": "all"})
        ) == {
            ids["d"],
            ids["fleft"],
            ids["fright"],
            ids["edge_end"],
            ids["lone"],
        }

    def test_empty_selection_is_empty_boundary(self, dag):
        node_selector, _, _ = dag
        assert (
            node_selector.get_direct_selection(
                _boundary_spec("upstream", {"tag": "does_not_exist"})
            )
            == set()
        )
        assert (
            node_selector.get_direct_selection(
                _boundary_spec("downstream", {"tag": "does_not_exist"})
            )
            == set()
        )
        direct, indirect = node_selector.select_nodes(
            _boundary_spec("downstream", {"tag": "does_not_exist"})
        )
        assert direct == set()
        assert indirect == set()

    def test_duplicate_hits_are_deduplicated(self, dag):
        node_selector, _, ids = dag
        # The wrapped selection reaches the same chain three different ways.
        wrapped = {
            "union": [
                {"tag": "chain"},
                {"fqn": "b"},
                {"fqn": "c"},
            ]
        }
        assert node_selector.get_direct_selection(_boundary_spec("upstream", wrapped)) == {
            ids["a"]
        }
        # Repeating boundary criteria in an outer union cannot duplicate nodes.
        spec = graph_cli.parse_from_definition(
            {
                "union": [
                    {"boundary": {"direction": "upstream", "definition": {"tag": "chain"}}},
                    {"boundary": {"direction": "upstream", "definition": {"tag": "chain"}}},
                ]
            }
        )
        assert node_selector.get_direct_selection(spec) == {ids["a"]}

    def test_inline_shorthand_spec(self, dag):
        node_selector, _, ids = dag
        spec = SelectionCriteria.from_single_spec("boundary:upstream:tag:chain")
        assert spec.method == MethodName.Boundary
        assert node_selector.get_direct_selection(spec) == {ids["a"]}

        spec = SelectionCriteria.from_single_spec("boundary:downstream:tag:chain")
        assert node_selector.get_direct_selection(spec) == {ids["d"]}

    def test_union_intersection_exclude(self, dag):
        node_selector, _, ids = dag

        # intersection: upstream boundary of every model, kept only if tagged chain
        spec = graph_cli.parse_from_definition(
            {
                "intersection": [
                    {"boundary": {"direction": "upstream", "definition": {"tag": "all"}}},
                    {"tag": "chain"},
                ]
            }
        )
        assert node_selector.get_direct_selection(spec) == {ids["a"]}

        # union + exclude: every upstream root except the chain's
        spec = graph_cli.parse_from_definition(
            {
                "union": [
                    {"boundary": {"direction": "upstream", "definition": {"tag": "all"}}},
                    {"exclude": [{"tag": "chain"}]},
                ]
            }
        )
        assert node_selector.get_direct_selection(spec) == {
            ids["froot"],
            ids["edge_start"],
            ids["lone"],
        }

    def test_cli_difference_syntax(self, dag):
        node_selector, _, ids = dag
        spec = parse_difference(["boundary:upstream:tag:all"], ["tag:chain"])
        direct, _ = node_selector.select_nodes(spec)
        assert direct == {ids["froot"], ids["edge_start"], ids["lone"]}

    def test_indirect_tests_follow_boundary_direct_selection(self, dag):
        node_selector, _, ids = dag

        # Upstream boundary is `a`; under the default eager strategy the test
        # directly attached to `a` is indirectly selected, while tests attached
        # to non-boundary models (`b`) are not.
        direct, indirect = node_selector.select_nodes(_boundary_spec("upstream", {"tag": "chain"}))
        assert direct == {ids["a"], ids["test_a"]}
        assert indirect == set()

        # Downstream boundary is `d`; only the test attached to `d` follows.
        direct, indirect = node_selector.select_nodes(
            _boundary_spec("downstream", {"tag": "chain"})
        )
        assert direct == {ids["d"], ids["test_d"]}
        assert ids["test_a"] not in direct
        assert ids["test_b"] not in direct
        assert indirect == set()

    def test_indirect_selection_empty_skips_tests(self, dag):
        node_selector, _, ids = dag
        spec = _boundary_spec("downstream", {"tag": "chain"})
        spec.indirect_selection = IndirectSelection.Empty
        direct, indirect = node_selector.select_nodes(spec)
        assert direct == {ids["d"]}
        assert indirect == set()

    def test_indirect_selection_cautious_promotes_attached_test(self, dag):
        node_selector, _, ids = dag
        spec = _boundary_spec("downstream", {"tag": "chain"})
        spec.indirect_selection = IndirectSelection.Cautious
        direct, _ = node_selector.select_nodes(spec)
        # The only parent of test_d (model d) is selected, so it is promoted
        # from the indirect set into the direct set.
        assert direct == {ids["d"], ids["test_d"]}

    def test_directly_selected_tests_participate_in_boundary(self, dag):
        node_selector, _, ids = dag
        # The wrapped definition selects the chain models AND the unique tests
        # directly. Tests are real nodes with real edges, so d (parent of
        # test_d) is no longer a downstream boundary; the tests themselves are.
        wrapped = {
            "union": [
                {"tag": "chain"},
                {"test_name": "unique"},
            ]
        }
        assert node_selector.get_direct_selection(_boundary_spec("upstream", wrapped)) == {
            ids["a"]
        }
        assert node_selector.get_direct_selection(_boundary_spec("downstream", wrapped)) == {
            ids["test_a"],
            ids["test_b"],
            ids["test_d"],
        }

    def test_wrapped_indirect_tests_do_not_define_boundary(self, dag):
        node_selector, _, ids = dag
        # Even though eager indirect selection would attach all three tests to
        # the wrapped chain selection, boundary computation must use direct
        # selection only: `d` remains the (single) downstream boundary.
        spec = _boundary_spec("downstream", {"tag": "chain"})
        assert node_selector.get_direct_selection(spec) == {ids["d"]}

    def test_named_selector_yaml_parsing(self):
        sf = SelectorFile.from_dict(
            {
                "selectors": [
                    {
                        "name": "chain_up",
                        "definition": {
                            "boundary": {
                                "direction": "upstream",
                                "definition": {"tag": "chain"},
                            }
                        },
                    },
                    {
                        "name": "chain_down_scalar",
                        "definition": {"boundary": "downstream:tag:chain"},
                    },
                ]
            }
        )
        parsed = graph_cli.parse_from_selectors_definition(sf)
        up = parsed["chain_up"]["definition"]
        assert isinstance(up, SelectionCriteria)
        assert up.method == MethodName.Boundary
        assert up.value == {"direction": "upstream", "definition": {"tag": "chain"}}

        down = parsed["chain_down_scalar"]["definition"]
        assert isinstance(down, SelectionCriteria)
        assert down.method == MethodName.Boundary
        assert down.value == "downstream:tag:chain"

    def test_named_selector_executes(self, dag):
        node_selector, _, ids = dag
        sf = SelectorFile.from_dict(
            {
                "selectors": [
                    {
                        "name": "chain_up",
                        "definition": {
                            "boundary": {
                                "direction": "upstream",
                                "definition": {"tag": "chain"},
                            }
                        },
                    }
                ]
            }
        )
        parsed = graph_cli.parse_from_selectors_definition(sf)
        direct, _ = node_selector.select_nodes(parsed["chain_up"]["definition"])
        assert direct == {ids["a"], ids["test_a"]}

    def test_manifest_is_not_modified(self, dag):
        node_selector, manifest, _ = dag
        parent_map_before = deepcopy(manifest.parent_map)
        child_map_before = deepcopy(manifest.child_map)
        graph_edges_before = set(node_selector.graph.edges())

        for direction in ("upstream", "downstream"):
            node_selector.select_nodes(_boundary_spec(direction, {"tag": "all"}))
            node_selector.get_direct_selection(_boundary_spec(direction, {"tag": "edge"}))

        assert manifest.parent_map == parent_map_before
        assert manifest.child_map == child_map_before
        assert set(node_selector.graph.edges()) == graph_edges_before

    @pytest.mark.parametrize(
        "value",
        [
            "upstream",
            "upstream:",
            "sideways:tag:chain",
            "nope",
            {"direction": "upstream"},
            {"definition": "tag:chain"},
            {"direction": "sideways", "definition": "tag:chain"},
            {"direction": "upstream", "definition": "tag:chain", "extra": "x"},
            {"direction": "upstream", "definition": ["tag:chain"]},
            {"direction": "upstream", "definition": 123},
            123,
        ],
    )
    def test_invalid_boundary_values_raise(self, value):
        with pytest.raises(DbtRuntimeError):
            BoundarySelectorMethod._parse_boundary_value(value)

    @pytest.mark.parametrize("direction", ["sideways", ""])
    def test_invalid_boundary_errors_propagate(self, dag, direction):
        node_selector, _, _ = dag
        spec = _boundary_spec(direction, {"tag": "chain"})
        with pytest.raises(DbtRuntimeError):
            node_selector.select_nodes(spec)

    def test_malformed_wrapped_definition_raises(self, dag):
        node_selector, _, _ = dag
        # `union` must be a list; a scalar here cannot be a selection definition.
        spec = _boundary_spec("upstream", {"union": "not-a-list"})
        with pytest.raises(DbtRuntimeError):
            node_selector.select_nodes(spec)

        # An empty dict is neither a set operator nor a method/value pair.
        spec = _boundary_spec("downstream", {})
        with pytest.raises(DbtRuntimeError):
            node_selector.select_nodes(spec)
