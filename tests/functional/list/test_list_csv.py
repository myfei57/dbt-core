import csv
import io
import json
import os

import pytest

from dbt.tests.util import run_dbt, run_dbt_and_capture

MODEL_SQL = """
select 1 as id
"""

# The description intentionally contains a comma, double quotes, and a newline
# so that CSV escaping is exercised end to end.
SCHEMA_YML = """
version: 2
models:
  - name: my_model
    description: "line one, with \\"quotes\\"\\nline two"
    config:
      materialized: table
      tags:
        - "tag,with,comma"
        - normal_tag
      meta:
        owner: data-team
"""

DEFAULT_HEADER = [
    "unique_id",
    "resource_type",
    "package_name",
    "name",
    "original_file_path",
]


def parse_csv(result):
    """Parse the list of printed CSV lines back into a list of records."""
    return list(csv.reader(io.StringIO("\n".join(result))))


def norm(path):
    return os.path.normcase(os.path.normpath(path))


class TestListCsvDefault:
    @pytest.fixture(scope="class")
    def models(self):
        return {"my_model.sql": MODEL_SQL}

    def test_ls_csv_default_columns(self, project):
        results = run_dbt(["ls", "--select", "my_model", "--output", "csv"])

        # Header is always first, followed by exactly one record per resource.
        assert results[0] == "unique_id,resource_type,package_name,name,original_file_path"
        assert len(results) == 2

        records = parse_csv(results)
        assert records[0] == DEFAULT_HEADER
        assert records[1] == [
            "model.test.my_model",
            "model",
            "test",
            "my_model",
            norm("models/my_model.sql"),
        ]


class TestListCsvOutputKeys:
    @pytest.fixture(scope="class")
    def models(self):
        return {"my_model.sql": MODEL_SQL, "schema.yml": SCHEMA_YML}

    def test_ls_csv_custom_keys_nested_and_escaping(self, project):
        results = run_dbt(
            [
                "ls",
                "--select",
                "my_model",
                "--output",
                "csv",
                "--output-keys",
                "name",
                "config.materialized",
                "tags",
                "description",
                "config.meta",
            ]
        )

        records = parse_csv(results)
        assert records[0] == [
            "name",
            "config.materialized",
            "tags",
            "description",
            "config.meta",
        ]
        assert len(records) == 2
        name, materialized, tags, description, meta = records[1]

        # Field order follows the user-given --output-keys order.
        assert name == "my_model"
        assert materialized == "table"

        # Lists and objects are compact JSON inside a single CSV cell.
        assert json.loads(tags) == ["tag,with,comma", "normal_tag"]
        assert json.loads(meta) == {"owner": "data-team"}

        # Commas, quotes, and newlines in strings survive a CSV round trip.
        assert description == 'line one, with "quotes"\nline two'

        # The raw data row is quoted wherever required, with quotes doubled.
        assert '""quotes""' in results[1]


class TestListCsvEmptySelection:
    @pytest.fixture(scope="class")
    def models(self):
        return {"my_model.sql": MODEL_SQL}

    def test_ls_csv_empty_selection_prints_header_only(self, project):
        results = run_dbt(["ls", "--select", "nothing_matches", "--output", "csv"])

        assert results == ["unique_id,resource_type,package_name,name,original_file_path"]
        assert parse_csv(results) == [DEFAULT_HEADER]

    def test_ls_csv_empty_selection_with_custom_keys_prints_header_only(self, project):
        results = run_dbt(
            [
                "ls",
                "--select",
                "nothing_matches",
                "--output",
                "csv",
                "--output-keys",
                "name",
                "config.materialized",
            ]
        )

        assert results == ["name,config.materialized"]


class TestListCsvSelectionAndOrdering:
    @pytest.fixture(scope="class")
    def models(self):
        return {
            "a_model.sql": MODEL_SQL,
            "b_model.sql": MODEL_SQL,
            "sub": {"c_model.sql": MODEL_SQL},
        }

    def test_ls_csv_reuses_selection_and_sorting(self, project):
        results = run_dbt(["ls", "--resource-type", "model", "--output", "csv"])

        records = parse_csv(results)
        assert records[0] == DEFAULT_HEADER
        # One record per selected resource, sorted by unique_id like the other
        # output formats.
        unique_ids = [record[0] for record in records[1:]]
        assert unique_ids == [
            "model.test.a_model",
            "model.test.b_model",
            "model.test.c_model",
        ]

        results = run_dbt(
            ["ls", "--resource-type", "model", "--select", "a_model", "--output", "csv"]
        )
        records = parse_csv(results)
        assert len(records) == 2
        assert records[1][0] == "model.test.a_model"


class TestListCsvStdoutOnly:
    @pytest.fixture(scope="class")
    def models(self):
        return {"a_model.sql": MODEL_SQL, "b_model.sql": MODEL_SQL}

    def test_ls_csv_output_is_not_mixed_with_logs(self, project):
        # Like the JSON format, CSV rows are emitted as print events, so under
        # --quiet the captured output is pure CSV with no run log lines.
        results, log_output = run_dbt_and_capture(
            ["ls", "--quiet", "--resource-type", "model", "--output", "csv"]
        )

        lines = log_output.splitlines()
        assert lines == [
            "unique_id,resource_type,package_name,name,original_file_path",
            "model.test.a_model,model,test,a_model," + norm("models/a_model.sql"),
            "model.test.b_model,model,test,b_model," + norm("models/b_model.sql"),
        ]

        # The whole captured stream parses as well-formed CSV: a header plus
        # exactly one data record per selected resource, all of header width.
        records = list(csv.reader(io.StringIO(log_output)))
        assert len(records) == 3
        assert all(len(record) == 5 for record in records)

        # The structured return value matches what was printed to stdout.
        assert lines == results
