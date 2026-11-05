import string
from argparse import Namespace
from queue import Empty
from typing import List
from unittest.mock import MagicMock

import networkx as nx
import pytest

import dbt.compilation
import dbt.config
import dbt.exceptions
import dbt.graph.cli as graph_cli
import dbt.graph.selector as graph_selector
import dbt.parser
import dbt.parser.manifest
import dbt.utils
import dbt_common.exceptions
from dbt.config.runtime import RuntimeConfig
from dbt.config.selectors import selector_config_from_data
from dbt.flags import set_from_args
from dbt.graph import NodeSelector, parse_difference, sample_selection
from dbt.graph.selector_spec import SelectionCriteria
from dbt.node_types import NodeType
from tests.unit.utils.manifest import make_generic_test, make_manifest, make_model

set_from_args(Namespace(WARN_ERROR=False), None)


def _get_graph():
    integer_graph = nx.balanced_tree(2, 2, nx.DiGraph())

    package_mapping = {
        i: "m." + ("X" if i % 2 == 0 else "Y") + "." + letter
        for (i, letter) in enumerate(string.ascii_lowercase)
    }

    # Edges: [(X.a, Y.b), (X.a, X.c), (Y.b, Y.d), (Y.b, X.e), (X.c, Y.f), (X.c, X.g)]
    return graph_selector.Graph(nx.relabel_nodes(integer_graph, package_mapping))


def _get_manifest(graph):
    nodes = {}
    for unique_id in graph:
        fqn = unique_id.split(".")
        node = MagicMock(
            unique_id=unique_id,
            fqn=fqn,
            package_name=fqn[0],
            tags=[],
            resource_type=NodeType.Model,
            empty=False,
            config=MagicMock(enabled=True),
            is_versioned=False,
        )
        nodes[unique_id] = node

    nodes["m.X.a"].tags = ["abc"]
    nodes["m.Y.b"].tags = ["abc", "bcef"]
    nodes["m.X.c"].tags = ["abc", "bcef"]
    nodes["m.Y.d"].tags = []
    nodes["m.X.e"].tags = ["efg", "bcef"]
    nodes["m.Y.f"].tags = ["efg", "bcef"]
    nodes["m.X.g"].tags = ["efg"]
    return MagicMock(nodes=nodes)


@pytest.fixture
def graph():
    return _get_graph()


@pytest.fixture
def mock_manifest_with_mock_graph(graph):
    return _get_manifest(graph)


def id_macro(arg):
    if isinstance(arg, str):
        return arg
    try:
        return "_".join(arg)
    except TypeError:
        return arg


run_specs = [
    # include by fqn
    (["X.a"], [], {"m.X.a"}),
    # include by tag
    (["tag:abc"], [], {"m.X.a", "m.Y.b", "m.X.c"}),
    # exclude by tag
    (["*"], ["tag:abc"], {"m.Y.d", "m.X.e", "m.Y.f", "m.X.g"}),
    # tag + fqn
    (["tag:abc", "a"], [], {"m.X.a", "m.Y.b", "m.X.c"}),
    (["tag:abc", "d"], [], {"m.X.a", "m.Y.b", "m.X.c", "m.Y.d"}),
    # multiple node selection across packages
    (["X.a", "b"], [], {"m.X.a", "m.Y.b"}),
    (["X.a+"], ["b"], {"m.X.a", "m.X.c", "m.Y.d", "m.X.e", "m.Y.f", "m.X.g"}),
    # children
    (["X.c+"], [], {"m.X.c", "m.Y.f", "m.X.g"}),
    (["X.a+1"], [], {"m.X.a", "m.Y.b", "m.X.c"}),
    (["X.a+"], ["tag:efg"], {"m.X.a", "m.Y.b", "m.X.c", "m.Y.d"}),
    # parents
    (["+Y.f"], [], {"m.X.c", "m.Y.f", "m.X.a"}),
    (["1+Y.f"], [], {"m.X.c", "m.Y.f"}),
    # childrens parents
    (["@X.c"], [], {"m.X.a", "m.X.c", "m.Y.f", "m.X.g"}),
    # multiple selection/exclusion
    (["tag:abc", "tag:bcef"], [], {"m.X.a", "m.Y.b", "m.X.c", "m.X.e", "m.Y.f"}),
    (["tag:abc", "tag:bcef"], ["tag:efg"], {"m.X.a", "m.Y.b", "m.X.c"}),
    (["tag:abc", "tag:bcef"], ["tag:efg", "a"], {"m.Y.b", "m.X.c"}),
    # intersections
    (["a,a"], [], {"m.X.a"}),
    (["+c,c+"], [], {"m.X.c"}),
    (["a,b"], [], set()),
    (["tag:abc,tag:bcef"], [], {"m.Y.b", "m.X.c"}),
    (["*,tag:abc,a"], [], {"m.X.a"}),
    (["a,tag:abc,*"], [], {"m.X.a"}),
    (["tag:abc,tag:bcef"], ["c"], {"m.Y.b"}),
    (["tag:bcef,tag:efg"], ["tag:bcef,@b"], {"m.Y.f"}),
    (["tag:bcef,tag:efg"], ["tag:bcef,@a"], set()),
    (["*,@a,+b"], ["*,tag:abc,tag:bcef"], {"m.X.a"}),
    (["tag:bcef,tag:efg", "*,tag:abc"], [], {"m.X.a", "m.Y.b", "m.X.c", "m.X.e", "m.Y.f"}),
    (["tag:bcef,tag:efg", "*,tag:abc"], ["e"], {"m.X.a", "m.Y.b", "m.X.c", "m.Y.f"}),
    (["tag:bcef,tag:efg", "*,tag:abc"], ["e"], {"m.X.a", "m.Y.b", "m.X.c", "m.Y.f"}),
    (["tag:bcef,tag:efg", "*,tag:abc"], ["e", "f"], {"m.X.a", "m.Y.b", "m.X.c"}),
    (["tag:bcef,tag:efg", "*,tag:abc"], ["tag:abc,tag:bcef"], {"m.X.a", "m.X.e", "m.Y.f"}),
    (["tag:bcef,tag:efg", "*,tag:abc"], ["tag:abc,tag:bcef", "tag:abc,a"], {"m.X.e", "m.Y.f"}),
]


@pytest.mark.parametrize("include,exclude,expected", run_specs, ids=id_macro)
def test_run_specs(include, exclude, expected, graph, mock_manifest_with_mock_graph):
    selector = graph_selector.NodeSelector(graph, mock_manifest_with_mock_graph)
    spec = graph_cli.parse_difference(include, exclude)
    selected, _ = selector.select_nodes(spec)

    assert selected == expected


param_specs = [
    ("a", False, None, False, None, "fqn", "a", False),
    ("+a", True, None, False, None, "fqn", "a", False),
    ("256+a", True, 256, False, None, "fqn", "a", False),
    ("a+", False, None, True, None, "fqn", "a", False),
    ("a+256", False, None, True, 256, "fqn", "a", False),
    ("+a+", True, None, True, None, "fqn", "a", False),
    ("16+a+32", True, 16, True, 32, "fqn", "a", False),
    ("@a", False, None, False, None, "fqn", "a", True),
    ("a.b", False, None, False, None, "fqn", "a.b", False),
    ("+a.b", True, None, False, None, "fqn", "a.b", False),
    ("256+a.b", True, 256, False, None, "fqn", "a.b", False),
    ("a.b+", False, None, True, None, "fqn", "a.b", False),
    ("a.b+256", False, None, True, 256, "fqn", "a.b", False),
    ("+a.b+", True, None, True, None, "fqn", "a.b", False),
    ("16+a.b+32", True, 16, True, 32, "fqn", "a.b", False),
    ("@a.b", False, None, False, None, "fqn", "a.b", True),
    ("a.b.*", False, None, False, None, "fqn", "a.b.*", False),
    ("+a.b.*", True, None, False, None, "fqn", "a.b.*", False),
    ("256+a.b.*", True, 256, False, None, "fqn", "a.b.*", False),
    ("a.b.*+", False, None, True, None, "fqn", "a.b.*", False),
    ("a.b.*+256", False, None, True, 256, "fqn", "a.b.*", False),
    ("+a.b.*+", True, None, True, None, "fqn", "a.b.*", False),
    ("16+a.b.*+32", True, 16, True, 32, "fqn", "a.b.*", False),
    ("@a.b.*", False, None, False, None, "fqn", "a.b.*", True),
    ("tag:a", False, None, False, None, "tag", "a", False),
    ("+tag:a", True, None, False, None, "tag", "a", False),
    ("256+tag:a", True, 256, False, None, "tag", "a", False),
    ("tag:a+", False, None, True, None, "tag", "a", False),
    ("tag:a+256", False, None, True, 256, "tag", "a", False),
    ("+tag:a+", True, None, True, None, "tag", "a", False),
    ("16+tag:a+32", True, 16, True, 32, "tag", "a", False),
    ("@tag:a", False, None, False, None, "tag", "a", True),
    ("source:a", False, None, False, None, "source", "a", False),
    ("source:a+", False, None, True, None, "source", "a", False),
    ("source:a+1", False, None, True, 1, "source", "a", False),
    ("source:a+32", False, None, True, 32, "source", "a", False),
    ("@source:a", False, None, False, None, "source", "a", True),
]


@pytest.mark.parametrize(
    "spec,parents,parents_depth,children,children_depth,filter_type,filter_value,childrens_parents",
    param_specs,
    ids=id_macro,
)
def test_parse_specs(
    spec,
    parents,
    parents_depth,
    children,
    children_depth,
    filter_type,
    filter_value,
    childrens_parents,
):
    parsed = graph_selector.SelectionCriteria.from_single_spec(spec)
    assert parsed.parents == parents
    assert parsed.parents_depth == parents_depth
    assert parsed.children == children
    assert parsed.children_depth == children_depth
    assert parsed.method == filter_type
    assert parsed.value == filter_value
    assert parsed.childrens_parents == childrens_parents


invalid_specs = [
    "@a+",
    "@a.b+",
    "@a.b*+",
    "@tag:a+",
    "@source:a+",
]


@pytest.mark.parametrize("invalid", invalid_specs, ids=lambda k: str(k))
def test_invalid_specs(invalid):
    with pytest.raises(dbt_common.exceptions.DbtRuntimeError):
        graph_selector.SelectionCriteria.from_single_spec(invalid)


class TestCompiler:
    def test_single_model(self, runtime_config: RuntimeConfig):
        model = make_model(pkg="pkg", name="model_one", code="SELECT * FROM events")
        manifest = make_manifest(nodes=[model])

        compiler = dbt.compilation.Compiler(config=runtime_config)
        linker = compiler.compile(manifest)

        assert linker.nodes() == {model.unique_id}
        assert linker.edges() == set()

    def test_two_models_simple_ref(self, runtime_config: RuntimeConfig):
        model_one = make_model(pkg="pkg", name="model_one", code="SELECT * FROM events")
        model_two = make_model(
            pkg="pkg",
            name="model_two",
            code="SELECT * FROM {{ref('model_one')}}",
            refs=[model_one],
        )
        models = [model_one, model_two]
        manifest = make_manifest(nodes=models)

        compiler = dbt.compilation.Compiler(config=runtime_config)
        linker = compiler.compile(manifest)

        expected_nodes: List[str] = [model.unique_id for model in models]
        assert linker.nodes() == set(expected_nodes)
        assert list(linker.edges()) == [tuple(expected_nodes)]


class TestNodeSelector:
    def test_dependency_list(self, runtime_config: RuntimeConfig):
        model_one = make_model(pkg="pkg", name="model_one", code="SELECT * FROM events")
        model_two = make_model(
            pkg="pkg",
            name="model_two",
            code="SELECT * FROM {{ref('model_one')}}",
            refs=[model_one],
        )
        model_three = make_model(
            pkg="pkg",
            name="model_three",
            code="""
                SELECT * FROM {{ ref("model_1") }}
                union all
                SELECT * FROM {{ ref("model_2") }}
            """,
            refs=[model_one, model_two],
        )
        model_four = make_model(
            pkg="pkg",
            name="model_four",
            code="SELECT * FROM {{ref('model_three')}}",
            refs=[model_three],
        )
        models = [model_one, model_two, model_three, model_four]
        manifest = make_manifest(nodes=models)

        # Get the graph
        compiler = dbt.compilation.Compiler(runtime_config)
        graph = compiler.compile(manifest)

        # Create the selector and get the queue
        selector = NodeSelector(graph, manifest)
        queue = selector.get_graph_queue(
            parse_difference(
                None,
                None,
            )
        )

        for model in models:
            assert not queue.empty()
            got = queue.get(block=False)
            assert got.unique_id == model.unique_id
            with pytest.raises(Empty):
                queue.get(block=False)
            queue.mark_done(got.unique_id)
        assert queue.empty()

    def test_select_downstream_of_empty_model(self, runtime_config: RuntimeConfig):
        # empty model
        model_one = make_model(pkg="other", name="model_one", code="")
        # non-empty model
        model_two = make_model(
            pkg="pkg",
            name="model_two",
            code="""select * from {{ref('model_one')}}""",
            refs=[model_one],
        )
        models = [model_one, model_two]
        manifest = make_manifest(nodes=models)

        # Get the graph
        compiler = dbt.compilation.Compiler(runtime_config)
        graph = compiler.compile(manifest)

        # Ensure that model_two is selected as downstream of model_one
        selector = NodeSelector(graph, manifest)
        spec = graph_selector.SelectionCriteria.from_single_spec("model_one+")
        assert selector.get_selected(spec) == {"model.pkg.model_two"}

        # Ensure that --indirect-selection empty returns the same result
        spec.indirect_selection = graph_selector.IndirectSelection.Empty
        assert selector.get_selected(spec) == {"model.pkg.model_two"}


def _sample_selector(name, count, seed, definition="tag:nightly", **extra):
    return {
        "name": name,
        "definition": {
            "sample": {"definition": definition, "count": count, "seed": seed, **extra}
        },
    }


def _sample_config(selectors):
    return selector_config_from_data({"selectors": selectors})


def _build_sample_project(n_models=7, n_multi_parent_tests=0):
    models = [
        make_model("pkg", f"model_{i}", "select 1 as id", tags=["nightly"])
        for i in range(n_models)
    ]
    nodes = list(models)
    single_tests = []
    for model in models:
        test = make_generic_test("pkg", "not_null", model, {}, column_name="id", refs=[model])
        nodes.append(test)
        single_tests.append(test)
    multi_tests = []
    for _ in range(n_multi_parent_tests):
        test = make_generic_test(
            "pkg",
            "relationships",
            models[1],
            {},
            column_name="id",
            refs=[models[0], models[1]],
        )
        nodes.append(test)
        multi_tests.append(test)
    manifest = make_manifest(nodes=nodes)
    graph = dbt.compilation.Linker().get_graph(manifest)
    return models, single_tests, multi_tests, manifest, graph


class TestSampleSelection:
    def test_sample_selection_helper_is_order_independent(self):
        candidates = [f"model.pkg.model_{i}" for i in range(10)]

        forward = sample_selection(candidates, 4, 99)
        backward = sample_selection(list(reversed(candidates)), 4, 99)
        shuffled = sample_selection(list(reversed(candidates[3:])) + candidates[:3], 4, 99)

        assert forward == backward == shuffled
        assert len(forward) == 4
        # duplicate inputs are deduplicated
        assert sample_selection(candidates + candidates, 4, 99) == forward
        # different seeds are free to (and do) produce different samples
        assert sample_selection(candidates, 4, 99) != sample_selection(candidates, 4, 100)

    def test_sample_is_reproducible_distinct_and_restricted(self):
        cfg = _sample_config([_sample_selector("three", 3, 42, indirect_selection="empty")])
        models, _, _, manifest, graph = _build_sample_project()
        model_ids = {m.unique_id for m in models}
        expected = sample_selection(model_ids, 3, 42)

        selector = NodeSelector(graph, manifest, selectors=cfg)
        first = selector.get_selected(cfg["three"]["definition"])
        # a brand-new selector simulates a brand-new dbt process
        second = NodeSelector(graph, manifest, selectors=cfg).get_selected(
            cfg["three"]["definition"]
        )

        assert first == second == expected
        assert len(first) == 3
        assert first <= model_ids

    def test_sample_zero_count_returns_empty(self):
        cfg = _sample_config([_sample_selector("zero", 0, 1, indirect_selection="empty")])
        models, _, _, manifest, graph = _build_sample_project()

        selected = NodeSelector(graph, manifest, selectors=cfg).get_selected(
            cfg["zero"]["definition"]
        )

        assert selected == set()

    def test_sample_count_at_or_above_candidate_count_selects_all(self):
        cfg = _sample_config(
            [
                _sample_selector("exact", 7, 1, indirect_selection="empty"),
                _sample_selector("more", 100, 1, indirect_selection="empty"),
            ]
        )
        models, _, _, manifest, graph = _build_sample_project()
        model_ids = {m.unique_id for m in models}
        selector = NodeSelector(graph, manifest, selectors=cfg)

        assert selector.get_selected(cfg["exact"]["definition"]) == model_ids
        assert selector.get_selected(cfg["more"]["definition"]) == model_ids

    def test_sample_eager_selection_only_attaches_tests_of_sampled_nodes(self):
        cfg = _sample_config([_sample_selector("three", 3, 42)])
        models, tests, _, manifest, graph = _build_sample_project()
        model_ids = {m.unique_id for m in models}
        sampled_models = sample_selection(model_ids, 3, 42)
        tests_by_model = {m.unique_id: t.unique_id for m, t in zip(models, tests)}

        selected = NodeSelector(graph, manifest, selectors=cfg).get_selected(
            cfg["three"]["definition"]
        )

        # only the sampled models are directly selected
        assert selected & model_ids == sampled_models
        # every sampled model gets its attached test (eager indirect selection)
        for model_id in sampled_models:
            assert tests_by_model[model_id] in selected
        # tests of un-sampled models are not pulled in by the sampling step
        for model_id in model_ids - sampled_models:
            assert tests_by_model[model_id] not in selected
            assert model_id not in selected

    def test_sample_empty_indirect_selection_drops_tests(self):
        cfg = _sample_config([_sample_selector("three", 3, 42, indirect_selection="empty")])
        models, tests, _, manifest, graph = _build_sample_project()

        selected = NodeSelector(graph, manifest, selectors=cfg).get_selected(
            cfg["three"]["definition"]
        )

        assert selected == sample_selection({m.unique_id for m in models}, 3, 42)
        assert not (selected & {t.unique_id for t in tests})

    def test_sample_cautious_selection_with_multi_parent_test(self):
        cfg = _sample_config(
            [
                _sample_selector("one", 1, 42, indirect_selection="cautious"),
                _sample_selector("all", 100, 42, indirect_selection="cautious"),
            ]
        )
        models, _, multi_tests, manifest, graph = _build_sample_project(
            n_models=4, n_multi_parent_tests=1
        )
        multi_test = multi_tests[0]
        selector = NodeSelector(graph, manifest, selectors=cfg)

        # the single sampled model is one of the test's parents, but the test
        # has an un-sampled parent, so cautious selection keeps it indirect
        one = selector.get_selected(cfg["one"]["definition"])
        assert len(one & {m.unique_id for m in models}) == 1
        assert multi_test.unique_id not in one

        # once every parent is sampled, the cautious rule selects the test
        all_selected = selector.get_selected(cfg["all"]["definition"])
        assert multi_test.unique_id in all_selected

    def test_sample_participates_in_union_intersection_and_exclusion(self):
        cfg = _sample_config(
            [
                _sample_selector("three", 3, 42, indirect_selection="empty"),
                _sample_selector("all", 100, 42),
                {
                    "name": "union_sample",
                    "definition": {
                        "union": [
                            {
                                "sample": {
                                    "definition": "tag:nightly",
                                    "count": 3,
                                    "seed": 42,
                                    "indirect_selection": "empty",
                                }
                            },
                            "fqn:model_2",
                        ]
                    },
                },
                {
                    "name": "intersection_sample",
                    "definition": {
                        "intersection": [
                            {"sample": {"definition": "tag:nightly", "count": 100, "seed": 42}},
                            "fqn:model_2",
                        ]
                    },
                },
                {
                    "name": "excluded_sample",
                    "definition": {
                        "union": [
                            {"sample": {"definition": "tag:nightly", "count": 3, "seed": 42}},
                            {"exclude": ["fqn:model_0"]},
                        ]
                    },
                },
            ]
        )
        models, tests, _, manifest, graph = _build_sample_project()
        model_ids = {m.unique_id for m in models}
        tests_by_model = {m.unique_id: t.unique_id for m, t in zip(models, tests)}
        selector = NodeSelector(graph, manifest, selectors=cfg)

        sampled_three = sample_selection(model_ids, 3, 42)

        # union: the sample plus a node outside of it
        union_selected = selector.get_selected(cfg["union_sample"]["definition"])
        assert union_selected & model_ids == sampled_three | {"model.pkg.model_2"}

        # intersection: the full sample keeps only the intersected node
        # (plus its eagerly-attached test)
        intersection_selected = selector.get_selected(cfg["intersection_sample"]["definition"])
        assert intersection_selected == {
            "model.pkg.model_2",
            tests_by_model["model.pkg.model_2"],
        }

        # exclusion removes an explicitly excluded sampled node
        excluded_selected = selector.get_selected(cfg["excluded_sample"]["definition"])
        assert "model.pkg.model_0" not in excluded_selected
        assert excluded_selected & model_ids == sampled_three - {"model.pkg.model_0"}

    def test_sample_wraps_a_named_selector(self):
        cfg = _sample_config(
            [
                _sample_selector("three", 3, 42, indirect_selection="empty"),
                _sample_selector(
                    "wrapped",
                    2,
                    7,
                    definition="selector:three",
                    indirect_selection="empty",
                ),
            ]
        )
        _, _, _, manifest, graph = _build_sample_project()
        selector = NodeSelector(graph, manifest, selectors=cfg)

        base = selector.get_selected(cfg["three"]["definition"])
        wrapped = selector.get_selected(cfg["wrapped"]["definition"])

        assert wrapped == sample_selection(base, 2, 7)
        assert wrapped <= base

        # and the sampled named selector is usable through the selector: method
        via_method = selector.get_selected(SelectionCriteria.from_single_spec("selector:three"))
        assert via_method == base
