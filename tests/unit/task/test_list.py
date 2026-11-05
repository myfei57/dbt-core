import csv
import io
import json
from argparse import Namespace
from unittest.mock import Mock, patch

from dbt.flags import get_flags, set_from_args
from dbt.task.list import ListTask
from dbt_common.events.types import PrintEvent
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


def parse_csv(rows):
    """Parse the line-oriented output of generate_csv back into records."""
    return list(csv.reader(io.StringIO("\n".join(rows))))


class TestCsvFormatValue:
    """Unit tests for the _csv_format_value helper"""

    def test_none_is_empty_cell(self):
        assert ListTask._csv_format_value(None) == ""

    def test_string_is_unwrapped(self):
        assert ListTask._csv_format_value("my_model") == "my_model"

    def test_empty_string_stays_empty(self):
        assert ListTask._csv_format_value("") == ""

    def test_list_is_compact_json(self):
        assert ListTask._csv_format_value(["a", "b"]) == '["a","b"]'

    def test_empty_list_is_compact_json(self):
        assert ListTask._csv_format_value([]) == "[]"

    def test_dict_is_compact_json(self):
        assert ListTask._csv_format_value({"a": 1}) == '{"a":1}'

    def test_empty_dict_is_compact_json(self):
        assert ListTask._csv_format_value({}) == "{}"

    def test_tuple_serializes_as_json_array(self):
        assert ListTask._csv_format_value(("a", 1)) == '["a",1]'

    def test_nested_containers_are_compact_json(self):
        assert ListTask._csv_format_value({"tags": ["a", "b"]}) == '{"tags":["a","b"]}'

    def test_scalars_stringify(self):
        assert ListTask._csv_format_value(1) == "1"
        assert ListTask._csv_format_value(1.5) == "1.5"
        assert ListTask._csv_format_value(True) == "True"
        assert ListTask._csv_format_value(False) == "False"


class TestCsvRow:
    """Unit tests for the _csv_row helper"""

    def test_simple_row(self):
        assert ListTask._csv_row(["a", "b", "c"]) == "a,b,c"

    def test_comma_is_quoted(self):
        assert ListTask._csv_row(["a,b", "c"]) == '"a,b",c'

    def test_quote_is_doubled_and_quoted(self):
        assert ListTask._csv_row(['a "quoted" value', "c"]) == '"a ""quoted"" value",c'

    def test_newline_is_quoted(self):
        row = ListTask._csv_row(["line1\nline2", "c"])
        # The embedded newline stays inside the quoted field so it remains one
        # CSV record, and no terminator is appended.
        assert row == '"line1\nline2",c'

    def test_carriage_return_is_quoted(self):
        assert ListTask._csv_row(["a\rb", "c"]) == '"a\rb",c'

    def test_empty_cells(self):
        assert ListTask._csv_row(["", "", "c"]) == ",,c"


class TestGenerateCsv:
    """Unit tests for the generate_csv method"""

    def setup_method(self):
        set_from_args(Namespace(models=None, output_keys=None), {})
        self.task = ListTask(get_flags(), None, None)

    def create_mock_node(self, name="test_model", **overrides):
        node = Mock()
        node_dict = {
            "name": name,
            "resource_type": "model",
            "package_name": "test",
            "unique_id": f"model.test.{name}",
            "original_file_path": f"models/{name}.sql",
            "alias": None,
            "tags": ["tag_a", "tag_b"],
            "config": {
                "materialized": "table",
                "enabled": True,
                "meta": {"owner": "data-team"},
            },
        }
        node_dict.update(overrides)
        node.to_dict.return_value = node_dict
        return node

    def generate(self, nodes, output_keys=()):
        with patch.object(self.task, "args") as mock_args:
            mock_args.output_keys = output_keys
            with patch.object(
                self.task, "_iterate_selected_nodes", return_value=nodes
            ) as mock_iterate:
                rows = list(self.task.generate_csv())
        return rows, mock_iterate

    def test_default_header_and_row(self):
        rows, _ = self.generate([self.create_mock_node()])
        assert rows[0] == "unique_id,resource_type,package_name,name,original_file_path"
        assert rows[1] == "model.test.test_model,model,test,test_model,models/test_model.sql"
        assert len(rows) == 2

    def test_header_is_always_first(self):
        rows, _ = self.generate([self.create_mock_node(), self.create_mock_node("other")])
        assert rows[0] == "unique_id,resource_type,package_name,name,original_file_path"
        assert len(rows) == 3

    def test_default_header_with_empty_selection(self):
        rows, _ = self.generate([])
        assert rows == ["unique_id,resource_type,package_name,name,original_file_path"]

    def test_custom_header_with_empty_selection(self):
        rows, _ = self.generate([], output_keys=["name", "config.materialized"])
        assert rows == ["name,config.materialized"]

    def test_custom_keys_preserve_given_order(self):
        node = self.create_mock_node()
        rows, _ = self.generate(
            [node],
            output_keys=["name", "unique_id", "package_name"],
        )
        records = parse_csv(rows)
        assert records == [
            ["name", "unique_id", "package_name"],
            ["test_model", "model.test.test_model", "test"],
        ]

    def test_nested_keys_use_dot_notation_as_header(self):
        node = self.create_mock_node()
        rows, _ = self.generate(
            [node],
            output_keys=["name", "config.materialized", "config.meta.owner"],
        )
        records = parse_csv(rows)
        assert records == [
            ["name", "config.materialized", "config.meta.owner"],
            ["test_model", "table", "data-team"],
        ]

    def test_missing_and_none_fields_are_empty_cells(self):
        node = self.create_mock_node()
        rows, _ = self.generate(
            [node],
            output_keys=[
                "name",
                "does_not_exist",
                "config.does_not_exist",
                "alias",
                "unique_id",
            ],
        )
        records = parse_csv(rows)
        assert records == [
            [
                "name",
                "does_not_exist",
                "config.does_not_exist",
                "alias",
                "unique_id",
            ],
            ["test_model", "", "", "", "model.test.test_model"],
        ]

    def test_list_and_dict_values_are_compact_json(self):
        node = self.create_mock_node()
        rows, _ = self.generate(
            [node],
            output_keys=["tags", "config.meta", "config"],
        )
        records = parse_csv(rows)
        assert records[0] == ["tags", "config.meta", "config"]
        assert json.loads(records[1][0]) == ["tag_a", "tag_b"]
        assert json.loads(records[1][1]) == {"owner": "data-team"}
        assert json.loads(records[1][2]) == {
            "materialized": "table",
            "enabled": True,
            "meta": {"owner": "data-team"},
        }

    def test_empty_list_serializes_as_empty_json_array(self):
        node = self.create_mock_node(tags=[])
        rows, _ = self.generate([node], output_keys=["tags"])
        records = parse_csv(rows)
        assert records == [["tags"], ["[]"]]

    def test_special_characters_are_escaped(self):
        node = self.create_mock_node(name='a, "b"\nc')
        rows, _ = self.generate([node], output_keys=["name"])
        records = parse_csv(rows)
        assert records == [["name"], ['a, "b"\nc']]

    def test_bool_and_number_values_stringify(self):
        node = self.create_mock_node()
        rows, _ = self.generate([node], output_keys=["config.enabled", "resource_type"])
        records = parse_csv(rows)
        assert records == [["config.enabled", "resource_type"], ["True", "model"]]

    def test_node_order_follows_selected_nodes(self):
        first = self.create_mock_node("first")
        second = self.create_mock_node("second")
        rows, _ = self.generate([second, first], output_keys=["name"])
        records = parse_csv(rows)
        assert records == [["name"], ["second"], ["first"]]

    def test_uses_same_node_selection(self):
        # generate_csv must source nodes from _iterate_selected_nodes, so it
        # keeps the existing selection and sorting rules.
        nodes, mock_iterate = self.generate([self.create_mock_node()])
        assert mock_iterate.called
        assert len(nodes) == 2
