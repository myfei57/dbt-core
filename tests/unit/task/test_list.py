import json
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from dbt.flags import get_flags, set_from_args
from dbt.node_types import NodeType
from dbt.task.list import ListTask
from dbt_common.events.types import PrintEvent
from dbt_common.exceptions import DbtRuntimeError
from tests.unit.fixtures import generic_test_node, model_node


def test_list_output_results():
    set_from_args(Namespace(models=None), {})
    task = ListTask(get_flags(), None, None)
    results = ["node1", "node2", "node3"]
    expected_node_results = ["node1", "node2", "node3"]

    with patch("dbt.task.list.fire_event") as mock_fire_event:
        node_results = task.output_results(results)

    assert node_results == expected_node_results
    # assert called with PrintEvent type object and message 'node1', 'node2', 'node3'
    for call_args in mock_fire_event.call_args_list:
        assert isinstance(call_args[0][0], PrintEvent)
        assert call_args[0][0].msg in expected_node_results


class TestGetNestedValue:
    """Unit tests for the _get_nested_value method"""

    def setup_method(self):
        set_from_args(Namespace(models=None), {})
        self.task = ListTask(get_flags(), None, None)

    def test_get_nested_value_simple_key(self):
        """Test getting a simple top-level key"""
        data = {"name": "test_model", "type": "model"}
        result = self.task._get_nested_value(data, "name")
        assert result == "test_model"

    def test_get_nested_value_nested_key(self):
        """Test getting a nested key"""
        data = {"name": "test_model", "config": {"materialized": "table", "tags": ["important"]}}
        result = self.task._get_nested_value(data, "config.materialized")
        assert result == "table"

    def test_get_nested_value_deep_nested_key(self):
        """Test getting a deeply nested key"""
        data = {
            "name": "test_model",
            "config": {"meta": {"owner": "data-team", "contact": {"email": "team@company.com"}}},
        }
        result = self.task._get_nested_value(data, "config.meta.owner")
        assert result == "data-team"

        result = self.task._get_nested_value(data, "config.meta.contact.email")
        assert result == "team@company.com"

    def test_get_nested_value_nonexistent_top_level(self):
        """Test getting a non-existent top-level key returns None"""
        data = {"name": "test_model", "type": "model"}
        result = self.task._get_nested_value(data, "nonexistent")
        assert result is None

    def test_get_nested_value_nonexistent_nested(self):
        """Test getting a non-existent nested key returns None"""
        data = {"name": "test_model", "config": {"materialized": "table"}}
        result = self.task._get_nested_value(data, "config.nonexistent")
        assert result is None

    def test_get_nested_value_nonexistent_parent(self):
        """Test getting a nested key where parent doesn't exist returns None"""
        data = {"name": "test_model", "type": "model"}
        result = self.task._get_nested_value(data, "nonexistent.child")
        assert result is None

    def test_get_nested_value_non_dict_parent(self):
        """Test getting a nested key where parent is not a dict returns None"""
        data = {"name": "test_model", "config": "not_a_dict"}
        result = self.task._get_nested_value(data, "config.materialized")
        assert result is None

    def test_get_nested_value_none_parent(self):
        """Test getting a nested key where parent is None returns None"""
        data = {"name": "test_model", "config": None}
        result = self.task._get_nested_value(data, "config.materialized")
        assert result is None

    def test_get_nested_value_list_value(self):
        """Test getting a nested key that contains a list"""
        data = {"name": "test_model", "config": {"tags": ["important", "daily"]}}
        result = self.task._get_nested_value(data, "config.tags")
        assert result == ["important", "daily"]

    def test_get_nested_value_dict_value(self):
        """Test getting a nested key that contains a dict"""
        data = {
            "name": "test_model",
            "config": {"meta": {"owner": "data-team", "criticality": "high"}},
        }
        result = self.task._get_nested_value(data, "config.meta")
        assert result == {"owner": "data-team", "criticality": "high"}


class TestGenerateJson:
    """Unit tests for the generate_json method to ensure coverage of nested key logic"""

    def setup_method(self):
        set_from_args(Namespace(models=None, output_keys=None), {})
        self.task = ListTask(get_flags(), None, None)

    def create_mock_node(self, name="test_model", materialized="table", meta=None):
        """Helper to create a mock node with specified properties"""
        from unittest.mock import Mock

        node = Mock()
        node.to_dict.return_value = {
            "name": name,
            "resource_type": "model",
            "unique_id": f"model.test.{name}",
            "config": {
                "materialized": materialized,
                "tags": [],
                "meta": meta or {},
            },
            "original_file_path": f"models/{name}.sql",
        }
        return node

    def test_generate_json_with_nested_keys(self):
        """Test generate_json with nested output keys"""
        # Mock args to have nested output keys
        with patch.object(self.task, "args") as mock_args:
            mock_args.output_keys = ["name", "config.materialized"]

            # Mock _iterate_selected_nodes to return our test node
            with patch.object(self.task, "_iterate_selected_nodes") as mock_iterate:
                node = self.create_mock_node("test_model", "table")
                mock_iterate.return_value = [node]

                # Get the result
                results = list(self.task.generate_json())

                assert len(results) == 1
                import json

                result_data = json.loads(results[0])
                assert result_data["name"] == "test_model"
                assert result_data["config.materialized"] == "table"

    def test_generate_json_with_nonexistent_nested_keys(self):
        """Test generate_json with non-existent nested keys"""
        # Mock args to have non-existent nested key
        with patch.object(self.task, "args") as mock_args:
            mock_args.output_keys = ["name", "config.nonexistent"]

            # Mock _iterate_selected_nodes to return our test node
            with patch.object(self.task, "_iterate_selected_nodes") as mock_iterate:
                node = self.create_mock_node("test_model", "table")
                mock_iterate.return_value = [node]

                # Get the result
                results = list(self.task.generate_json())

                assert len(results) == 1
                import json

                result_data = json.loads(results[0])
                assert result_data["name"] == "test_model"
                # Non-existent key should not be in result
                assert "config.nonexistent" not in result_data

    def test_generate_json_with_mixed_keys(self):
        """Test generate_json with mix of regular and nested keys"""
        # Mock args to have mixed keys
        with patch.object(self.task, "args") as mock_args:
            mock_args.output_keys = [
                "name",
                "resource_type",
                "config.materialized",
                "config.meta.owner",
            ]

            # Mock _iterate_selected_nodes to return our test node
            with patch.object(self.task, "_iterate_selected_nodes") as mock_iterate:
                node = self.create_mock_node("test_model", "incremental", {"owner": "data-team"})
                mock_iterate.return_value = [node]

                # Get the result
                results = list(self.task.generate_json())

                assert len(results) == 1
                import json

                result_data = json.loads(results[0])
                assert result_data["name"] == "test_model"
                assert result_data["resource_type"] == "model"
                assert result_data["config.materialized"] == "incremental"
                assert result_data["config.meta.owner"] == "data-team"

    def test_generate_json_with_no_output_keys(self):
        """Test generate_json without output_keys (default behavior)"""
        # Mock args to have None output_keys
        with patch.object(self.task, "args") as mock_args:
            mock_args.output_keys = None

            # Mock _iterate_selected_nodes to return our test node
            with patch.object(self.task, "_iterate_selected_nodes") as mock_iterate:
                node = self.create_mock_node("test_model", "table")
                mock_iterate.return_value = [node]

                # Get the result
                results = list(self.task.generate_json())

                assert len(results) == 1
                import json

                result_data = json.loads(results[0])
                # Should contain ALLOWED_KEYS
                for allowed_key in self.task.ALLOWED_KEYS:
                    if allowed_key in node.to_dict.return_value:
                        assert allowed_key in result_data

    def test_generate_json_deep_nested_path(self):
        """Test generate_json with deeply nested paths"""
        # Mock args to have deep nested key
        with patch.object(self.task, "args") as mock_args:
            mock_args.output_keys = ["config.meta.contact.email"]

            # Mock _iterate_selected_nodes to return our test node
            with patch.object(self.task, "_iterate_selected_nodes") as mock_iterate:
                node = self.create_mock_node(
                    "test_model", "table", {"contact": {"email": "team@company.com"}}
                )
                mock_iterate.return_value = [node]

                # Get the result
                results = list(self.task.generate_json())

                assert len(results) == 1
                import json

                result_data = json.loads(results[0])
                assert result_data["config.meta.contact.email"] == "team@company.com"


class TestDirectParentsInLsOutput:
    def setup_method(self):
        set_from_args(Namespace(models=None, output_keys=None), {})
        self.task = ListTask(get_flags(), None, None)

    def _ls_json(self, node) -> dict:
        with patch.object(self.task, "args") as args:
            args.output_keys = None
            with patch.object(self.task, "_iterate_selected_nodes", return_value=[node]):
                [raw] = list(self.task.generate_json())
        return json.loads(raw)

    def test_model_emits_direct_parents(self):
        model = model_node()
        model.direct_parents = ["model.test.upstream", "seed.test.seed"]

        assert self._ls_json(model)["direct_parents"] == [
            "model.test.upstream",
            "seed.test.seed",
        ]

    def test_empty_direct_parents_is_emitted_not_omitted(self):
        # Empty is authoritative ("no parents"), not "field missing".
        model = model_node()
        model.direct_parents = []

        emitted = self._ls_json(model)

        assert emitted["direct_parents"] == []

    def test_non_model_has_no_direct_parents_field(self):
        emitted = self._ls_json(generic_test_node())

        assert "direct_parents" not in emitted

    def test_emitted_list_is_a_copy(self):
        model = model_node()
        model.direct_parents = ["model.test.upstream"]

        emitted = self._ls_json(model)
        emitted["direct_parents"].append("tampered")

        assert model.direct_parents == ["model.test.upstream"]

    def test_serialization_strips_direct_parents(self):
        # Pins the contract that motivates the re-injection in generate_json.
        # If __post_serialize__ ever stops stripping, the re-injection becomes
        # a duplicate write.
        model = model_node()
        model.direct_parents = ["model.test.upstream"]

        assert "direct_parents" not in model.to_dict(omit_none=False)


def _count_node(unique_id, resource_type=NodeType.Model, package_name="root"):
    return SimpleNamespace(
        unique_id=unique_id,
        resource_type=resource_type,
        package_name=package_name,
    )


class TestGenerateCountSummary:
    """Unit tests for the `--output count` summary generator."""

    def setup_method(self):
        set_from_args(
            Namespace(models=None, output="count", output_group_by=()),
            {},
        )
        self.task = ListTask(get_flags(), None, None)

    def _summary(self, nodes, group_by=()):
        # Flags is a frozen dataclass, so patch the task's args wholesale
        # rather than mutate the attribute in place.
        with patch.object(self.task, "args") as mock_args:
            mock_args.output_group_by = group_by
            with patch.object(self.task, "_iterate_selected_nodes", return_value=nodes):
                raw = list(self.task.generate_count_summary())
        assert len(raw) == 1
        return json.loads(raw[0])

    def test_default_grouping_by_resource_type(self):
        nodes = [
            _count_node("model.root.a", NodeType.Model),
            _count_node("test.root.t1", NodeType.Test),
            _count_node("source.root.s.t", NodeType.Source),
            _count_node("model.root.b", NodeType.Model),
        ]
        summary = self._summary(nodes)
        assert summary == {
            "total": 4,
            "groups": [
                {"resource_type": "model", "count": 2},
                {"resource_type": "source", "count": 1},
                {"resource_type": "test", "count": 1},
            ],
        }

    def test_grouping_by_package(self):
        nodes = [
            _count_node("model.root.a", NodeType.Model, "root"),
            _count_node("model.pkg_a.a", NodeType.Model, "pkg_a"),
            _count_node("test.pkg_a.t", NodeType.Test, "pkg_a"),
            _count_node("model.pkg_b.b", NodeType.Model, "pkg_b"),
        ]
        summary = self._summary(nodes, ("package",))
        assert summary == {
            "total": 4,
            "groups": [
                {"package": "pkg_a", "count": 2},
                {"package": "pkg_b", "count": 1},
                {"package": "root", "count": 1},
            ],
        }

    def test_grouping_by_resource_type_and_package(self):
        nodes = [
            _count_node("model.root.a", NodeType.Model, "root"),
            _count_node("test.root.t", NodeType.Test, "root"),
            _count_node("model.pkg_a.a", NodeType.Model, "pkg_a"),
        ]
        summary = self._summary(nodes, ("resource_type", "package"))
        assert summary == {
            "total": 3,
            "groups": [
                {"resource_type": "model", "package": "pkg_a", "count": 1},
                {"resource_type": "model", "package": "root", "count": 1},
                {"resource_type": "test", "package": "root", "count": 1},
            ],
        }

    def test_group_keys_use_canonical_order(self):
        # Dimensions passed in reverse order are still emitted and sorted in
        # the canonical resource_type -> package order.
        summary = self._summary(
            [_count_node("model.pkg_a.a", NodeType.Model, "pkg_a")],
            ("package", "resource_type"),
        )
        assert summary == {
            "total": 1,
            "groups": [{"resource_type": "model", "package": "pkg_a", "count": 1}],
        }

    def test_duplicate_dimensions_are_deduplicated(self):
        with patch.object(self.task, "args") as mock_args:
            mock_args.output_group_by = ("package", "package")
            assert self.task.count_group_by == ("package",)

    def test_empty_selection_returns_zero_total_and_empty_groups(self):
        assert self._summary([]) == {"total": 0, "groups": []}

    def test_resource_hit_multiple_times_is_counted_once(self):
        # The same unique_id can be yielded via several selection criteria;
        # it must only contribute once to the total and its group.
        nodes = [
            _count_node("model.root.a", NodeType.Model),
            _count_node("model.root.a", NodeType.Model),
        ]
        summary = self._summary(nodes)
        assert summary == {
            "total": 1,
            "groups": [{"resource_type": "model", "count": 1}],
        }

    def test_resources_classified_by_actual_type(self):
        nodes = [
            _count_node("model.root.m", NodeType.Model),
            _count_node("snapshot.root.s", NodeType.Snapshot),
            _count_node("seed.root.d", NodeType.Seed),
            _count_node("source.root.src.tbl", NodeType.Source),
            _count_node("test.root.t", NodeType.Test),
            _count_node("exposure.root.e", NodeType.Exposure),
            _count_node("metric.root.m", NodeType.Metric),
            _count_node("semantic_model.root.sm", NodeType.SemanticModel),
            _count_node("saved_query.root.sq", NodeType.SavedQuery),
            _count_node("unit_test.root.ut", NodeType.Unit),
        ]
        summary = self._summary(nodes)
        assert summary["total"] == 10
        assert [group["resource_type"] for group in summary["groups"]] == sorted(
            group["resource_type"] for group in summary["groups"]
        )
        assert {group["resource_type"] for group in summary["groups"]} == {
            "model",
            "snapshot",
            "seed",
            "source",
            "test",
            "exposure",
            "metric",
            "semantic_model",
            "saved_query",
            "unit_test",
        }
        assert all(group["count"] == 1 for group in summary["groups"])

    def test_groups_are_stably_sorted(self):
        nodes = [
            _count_node("model.c.c", NodeType.Model, "c"),
            _count_node("model.a.a", NodeType.Model, "a"),
            _count_node("model.b.b", NodeType.Model, "b"),
            _count_node("test.a.t", NodeType.Test, "a"),
        ]
        summary = self._summary(nodes, ("resource_type", "package"))
        assert [(g["resource_type"], g["package"]) for g in summary["groups"]] == [
            ("model", "a"),
            ("model", "b"),
            ("model", "c"),
            ("test", "a"),
        ]

    def test_output_is_a_single_machine_readable_json_line(self):
        with patch.object(self.task, "_iterate_selected_nodes", return_value=[]):
            with patch("dbt.task.list.fire_event") as mock_fire_event:
                results = self.task.output_results(self.task.generate_count_summary())
        assert len(results) == 1
        assert len(mock_fire_event.call_args_list) == 1
        event = mock_fire_event.call_args_list[0][0][0]
        assert isinstance(event, PrintEvent)
        assert json.loads(event.msg) == {"total": 0, "groups": []}


class TestCountOutputValidation:
    def _make_task(self, output, output_group_by=()):
        set_from_args(
            Namespace(
                models=None,
                output=output,
                output_group_by=output_group_by,
            ),
            {},
        )
        return ListTask(get_flags(), None, None)

    def test_group_by_requires_count_output(self):
        with pytest.raises(DbtRuntimeError, match="output_group_by"):
            self._make_task("name", output_group_by=("package",))

    def test_group_by_accepted_with_count_output(self):
        task = self._make_task("count", output_group_by=("package",))
        assert task.count_group_by == ("package",)

    def test_default_group_by_is_resource_type(self):
        task = self._make_task("count")
        assert task.count_group_by == ("resource_type",)
