import pytest

from dbt.tests.util import run_dbt, run_dbt_and_capture

#
# DAG under test (all edges are direct ref() dependencies):
#
#   chain:       a -> b -> c
#   fork:        froot -> fleft
#                     \\-> fright
#   gap:         edge_start -> mid -> edge_end   (only the ends are tagged)
#   isolated:    lone
#
# Generic tests are attached to `a` and `c` (see schema_yml).
#

models_yml = """
version: 2

models:
  - name: a
    columns:
      - name: id
        data_tests:
          - not_null
  - name: c
    columns:
      - name: id
        data_tests:
          - not_null
"""

selectors_yml = """
selectors:
  - name: chain_up
    definition:
      boundary:
        direction: upstream
        definition:
          tag: chain
  - name: chain_down
    definition:
      boundary:
        direction: downstream
        definition:
          tag: chain
  - name: fork_down
    definition:
      boundary:
        direction: downstream
        definition:
          tag: fork
  - name: edge_up
    definition:
      boundary:
        direction: upstream
        definition:
          tag: edge
  - name: roots_minus_chain
    definition:
      union:
        - boundary:
            direction: upstream
            definition:
              tag: all
        - exclude:
          - tag: chain
  - name: chain_roots_intersection
    definition:
      intersection:
        - boundary:
            direction: upstream
            definition:
              tag: all
        - tag: chain
  - name: empty_boundary
    definition:
      boundary:
        direction: upstream
        definition:
          tag: never_tagged
  - name: invalid_direction
    definition:
      boundary:
        direction: sideways
        definition:
          tag: chain
"""


class TestBoundarySelection:
    @pytest.fixture(scope="class")
    def models(self):
        return {
            "schema.yml": models_yml,
            "a.sql": "{{ config(tags=['chain', 'all']) }} select 1 as id",
            "b.sql": "{{ config(tags=['chain', 'all']) }} select * from {{ ref('a') }}",
            "c.sql": "{{ config(tags=['chain', 'all']) }} select * from {{ ref('b') }}",
            "froot.sql": "{{ config(tags=['fork', 'all']) }} select 1 as id",
            "fleft.sql": ("{{ config(tags=['fork', 'all']) }} select * from {{ ref('froot') }}"),
            "fright.sql": ("{{ config(tags=['fork', 'all']) }} select * from {{ ref('froot') }}"),
            "edge_start.sql": "{{ config(tags=['edge', 'all']) }} select 1 as id",
            "mid.sql": "{{ config(tags=['all']) }} select * from {{ ref('edge_start') }}",
            "edge_end.sql": ("{{ config(tags=['edge', 'all']) }} select * from {{ ref('mid') }}"),
            "lone.sql": "{{ config(tags=['solo', 'all']) }} select 1 as id",
        }

    @pytest.fixture(scope="class")
    def selectors(self):
        return selectors_yml

    def _models(self, selector_args):
        results = run_dbt(["ls", "--resource-type", "model"] + selector_args)
        return set(results)

    def test_chain_boundaries(self, project):
        assert self._models(["--selector", "chain_up"]) == {"model.test.a"}
        assert self._models(["--selector", "chain_down"]) == {"model.test.c"}

    def test_fork_downstream_boundary(self, project):
        assert self._models(["--selector", "fork_down"]) == {
            "model.test.fleft",
            "model.test.fright",
        }

    def test_outside_node_is_not_bridged(self, project):
        # `mid` sits between the two edge-tagged models but is outside the
        # selection, so both ends are upstream boundaries and no edge is
        # created through `mid`.
        assert self._models(["--selector", "edge_up"]) == {
            "model.test.edge_end",
            "model.test.edge_start",
        }
        chain_down = run_dbt(
            [
                "ls",
                "--resource-type",
                "model",
                "--select",
                "boundary:downstream:tag:edge",
            ]
        )
        assert set(chain_down) == {"model.test.edge_end", "model.test.edge_start"}

    def test_isolated_node_is_its_own_boundary(self, project):
        upstream = run_dbt(
            ["ls", "--resource-type", "model", "--select", "boundary:upstream:tag:solo"]
        )
        downstream = run_dbt(
            ["ls", "--resource-type", "model", "--select", "boundary:downstream:tag:solo"]
        )
        assert set(upstream) == {"model.test.lone"}
        assert set(downstream) == {"model.test.lone"}

    def test_composes_with_union_intersection_and_exclude(self, project):
        assert self._models(["--selector", "roots_minus_chain"]) == {
            "model.test.froot",
            "model.test.edge_start",
            "model.test.lone",
        }
        assert self._models(["--selector", "chain_roots_intersection"]) == {"model.test.a"}

    def test_empty_selection_is_empty_boundary(self, project):
        results = run_dbt(["ls", "--selector", "empty_boundary"])
        assert results == []

    def test_inline_shorthand(self, project):
        assert self._models(["--select", "boundary:upstream:tag:chain"]) == {"model.test.a"}
        assert self._models(["--select", "boundary:downstream:tag:fork"]) == {
            "model.test.fleft",
            "model.test.fright",
        }

    def test_indirect_tests_follow_boundary(self, project):
        # Downstream boundary is `c`: the generic test attached to c is
        # indirectly selected, while the test attached to non-boundary `a` is not.
        results = run_dbt(["ls", "--selector", "chain_down"])
        assert "model.test.c" in results
        assert "model.test.a" not in results
        assert any(node.startswith("test.test.not_null_c_id") for node in results)
        assert not any(node.startswith("test.test.not_null_a_id") for node in results)

        # The --indirect-selection empty strategy skips attached tests entirely.
        results = run_dbt(["ls", "--indirect-selection", "empty", "--selector", "chain_down"])
        assert "model.test.c" in results
        assert not any(node.startswith("test.") for node in results)

    def test_invalid_direction_raises_clear_error(self, project):
        _, log_output = run_dbt_and_capture(
            ["ls", "--selector", "invalid_direction"], expect_pass=False
        )
        assert "Invalid boundary direction" in log_output
        assert "upstream" in log_output and "downstream" in log_output
