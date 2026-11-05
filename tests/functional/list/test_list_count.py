import json
import shutil
from pathlib import Path

import pytest

from dbt.cli.exceptions import DbtUsageException
from dbt.tests.util import run_dbt
from dbt_common.exceptions import DbtRuntimeError

model_a_sql = """
select 1 as id
"""

model_b_sql = """
select * from {{ ref('model_a') }}
"""

schema_yml = """
version: 2

sources:
  - name: my_source
    tables:
      - name: my_table

models:
  - name: model_a
    columns:
      - name: id
        tests:
          - not_null
  - name: model_b
    columns:
      - name: id
        tests:
          - unique
"""


def _count_summary(results):
    # The count summary is the only thing on the machine-readable output
    # channel: a single JSON object on a single line.
    assert len(results) == 1
    return json.loads(results[0])


class TestListCountOutput:
    @pytest.fixture(scope="class")
    def models(self):
        return {
            "model_a.sql": model_a_sql,
            "model_b.sql": model_b_sql,
            "schema.yml": schema_yml,
        }

    @pytest.fixture(scope="class")
    def packages(self):
        return {
            "packages": [
                {"local": "count_packages/pkg_alpha"},
                {"local": "count_packages/pkg_beta"},
            ]
        }

    @pytest.fixture(scope="class", autouse=True)
    def setup_local_packages(self, project):
        # Two installed packages so we can verify grouping across packages.
        shutil.copytree(
            project.test_dir / Path("count_packages"),
            project.project_root / Path("count_packages"),
        )
        run_dbt(["deps"])

    def test_count_default_grouping_by_resource_type(self, project):
        results = run_dbt(["ls", "--output", "count", "--select", "*", "source:*"])
        summary = _count_summary(results)
        assert summary == {
            "total": 7,
            "groups": [
                {"resource_type": "model", "count": 4},
                {"resource_type": "source", "count": 1},
                {"resource_type": "test", "count": 2},
            ],
        }

    def test_count_grouping_by_package(self, project):
        results = run_dbt(
            [
                "ls",
                "--output",
                "count",
                "--output-group-by",
                "package",
                "--select",
                "*",
                "source:*",
            ]
        )
        summary = _count_summary(results)
        assert summary == {
            "total": 7,
            "groups": [
                {"package": "pkg_alpha", "count": 1},
                {"package": "pkg_beta", "count": 1},
                {"package": "test", "count": 5},
            ],
        }

    def test_count_grouping_by_resource_type_and_package(self, project):
        results = run_dbt(
            [
                "ls",
                "--output",
                "count",
                "--output-group-by",
                "resource_type",
                "package",
                "--select",
                "*",
                "source:*",
            ]
        )
        summary = _count_summary(results)
        assert summary == {
            "total": 7,
            "groups": [
                {"resource_type": "model", "package": "pkg_alpha", "count": 1},
                {"resource_type": "model", "package": "pkg_beta", "count": 1},
                {"resource_type": "model", "package": "test", "count": 2},
                {"resource_type": "source", "package": "test", "count": 1},
                {"resource_type": "test", "package": "test", "count": 2},
            ],
        }

    def test_count_group_by_dimensions_may_be_space_separated(self, project):
        results = run_dbt(
            [
                "ls",
                "--output",
                "count",
                "--output-group-by",
                "package resource_type",
                "--select",
                "*",
            ]
        )
        summary = _count_summary(results)
        # Keys are always emitted in the canonical resource_type -> package
        # order, regardless of the order dimensions are passed in.
        assert list(summary["groups"][0].keys()) == ["resource_type", "package", "count"]
        assert summary["total"] == 6

    def test_count_overlapping_selections_count_each_resource_once(self, project):
        # model_a is matched by both "model_a" and "model_a+"; attached tests
        # are included indirectly. The union (4 resources) must match a single
        # selection, proving no double counting from overlap.
        overlapping = run_dbt(
            [
                "ls",
                "--output",
                "count",
                "--select",
                "model_a",
                "model_a+",
            ]
        )
        single = run_dbt(["ls", "--output", "count", "--select", "model_a+"])

        assert (
            _count_summary(overlapping)
            == _count_summary(single)
            == {
                "total": 4,
                "groups": [
                    {"resource_type": "model", "count": 2},
                    {"resource_type": "test", "count": 2},
                ],
            }
        )

    def test_count_respects_exclude(self, project):
        results = run_dbt(
            [
                "ls",
                "--output",
                "count",
                "--resource-type",
                "model",
                "--select",
                "*",
                "--exclude",
                "model_b",
            ]
        )
        summary = _count_summary(results)
        assert summary == {
            "total": 3,
            "groups": [{"resource_type": "model", "count": 3}],
        }

    def test_count_indirect_tests_are_classified_as_tests(self, project):
        results = run_dbt(
            ["ls", "--output", "count", "--resource-type", "test", "--select", "model_a"]
        )
        summary = _count_summary(results)
        assert summary == {
            "total": 1,
            "groups": [{"resource_type": "test", "count": 1}],
        }

    def test_count_empty_selection_returns_zero_total_and_empty_groups(self, project):
        results = run_dbt(["ls", "--output", "count", "--select", "no_such_resource_zzz"])
        summary = _count_summary(results)
        assert summary == {"total": 0, "groups": []}

    def test_existing_output_formats_are_unchanged(self, project):
        names = run_dbt(["ls", "--output", "name", "--resource-type", "model"])
        assert set(names) == {"model_a", "model_b", "alpha_model", "beta_model"}
        # Plain listing is newline-delimited names, not JSON objects.
        for name in names:
            with pytest.raises(json.JSONDecodeError):
                json.loads(name)

        default_selectors = run_dbt(["ls", "--resource-type", "model"])
        assert set(default_selectors) == {
            "test.model_a",
            "test.model_b",
            "pkg_alpha.alpha_model",
            "pkg_beta.beta_model",
        }

    def test_output_group_by_requires_count_output(self, project):
        with pytest.raises(DbtRuntimeError, match="output_group_by"):
            run_dbt(["ls", "--output", "name", "--output-group-by", "package"])

    def test_output_group_by_rejects_unknown_dimension(self, project):
        with pytest.raises(DbtUsageException):
            run_dbt(["ls", "--output", "count", "--output-group-by", "schema"])
