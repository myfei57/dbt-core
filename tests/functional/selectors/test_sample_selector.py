import pytest

from dbt.tests.util import run_dbt, run_dbt_and_capture

_MODELS = {f"model_{i}.sql": "select 1 as id" for i in range(8)}


_SCHEMA_YML = """
version: 2
models:
{models}
""".format(
    models="\n".join(
        f"""  - name: model_{i}
    columns:
      - name: id
        tests:
          - not_null"""
        for i in range(8)
    )
)


selectors_yml = """
selectors:
  - name: sample_three
    description: Reproducibly sample three models
    definition:
      sample:
        definition:
          resource_type: model
        count: 3
        seed: 42
  - name: sample_zero
    description: Sampling zero nodes returns an empty selection
    definition:
      sample:
        definition: resource_type:model
        count: 0
        seed: 42
  - name: sample_all
    description: A count above the candidate count selects every candidate
    definition:
      sample:
        definition: resource_type:model
        count: 100
        seed: 42
  - name: sample_three_with_exclusion
    description: Sampling participates in union/exclude combinations
    definition:
      union:
        - sample:
            definition: resource_type:model
            count: 3
            seed: 42
        - exclude:
            - fqn:model_0
  - name: sample_two_eager
    description: Sampling keeps indirect tests of sampled models (eager mode)
    definition:
      sample:
        definition: resource_type:model
        count: 2
        seed: 42
"""


bad_negative_count_selectors_yml = """
selectors:
  - name: bad_selector
    definition:
      sample:
        definition: resource_type:model
        count: -1
        seed: 42
"""


missing_seed_selectors_yml = """
selectors:
  - name: bad_selector
    definition:
      sample:
        definition: resource_type:model
        count: 3
"""


class TestSampleSelector:
    @pytest.fixture(scope="class")
    def models(self):
        return {**_MODELS, "schema.yml": _SCHEMA_YML}

    @pytest.fixture(scope="class")
    def selectors(self):
        return selectors_yml

    def test_sample_has_requested_size(self, project):
        results = run_dbt(["ls", "--selector", "sample_three", "--resource-type", "model"])
        assert len(results) == 3

    def test_sample_is_reproducible_across_processes(self, project):
        # each run_dbt call is a fresh dbt process
        first = run_dbt(["ls", "--selector", "sample_three", "--resource-type", "model"])
        second = run_dbt(["ls", "--selector", "sample_three", "--resource-type", "model"])
        assert sorted(first) == sorted(second)

    def test_sample_zero_returns_nothing(self, project):
        results = run_dbt(["ls", "--selector", "sample_zero"])
        assert results == []

    def test_sample_count_above_candidates_selects_all(self, project):
        results = run_dbt(["ls", "--selector", "sample_all", "--resource-type", "model"])
        assert sorted(results) == sorted(f"test.model.model_{i}" for i in range(8))

    def test_sample_participates_in_exclusion(self, project):
        sampled = run_dbt(["ls", "--selector", "sample_three", "--resource-type", "model"])
        excluded = run_dbt(
            ["ls", "--selector", "sample_three_with_exclusion", "--resource-type", "model"]
        )
        assert "test.model.model_0" not in excluded
        assert sorted(excluded) == sorted(set(sampled) - {"test.model.model_0"})

    def test_sample_keeps_indirect_tests_of_sampled_models(self, project):
        sampled_models = set(
            run_dbt(["ls", "--selector", "sample_two_eager", "--resource-type", "model"])
        )
        assert len(sampled_models) == 2

        all_results = set(run_dbt(["ls", "--selector", "sample_two_eager"]))

        sampled_names = {name.split(".")[-1] for name in sampled_models}
        sampled_tests = {f"test.test.not_null_{name}_id" for name in sampled_names}

        # the sampled models and their tests are selected
        assert sampled_models <= all_results
        assert sampled_tests <= all_results

        # tests attached to un-sampled models are not selected
        un_sampled_names = {f"model_{i}" for i in range(8)} - sampled_names
        un_sampled_tests = {f"test.test.not_null_{name}_id" for name in un_sampled_names}
        un_sampled_models = {f"test.model.{name}" for name in un_sampled_names}
        assert not (all_results & un_sampled_tests)
        assert not (all_results & un_sampled_models)


class TestSampleSelectorNegativeCount:
    @pytest.fixture(scope="class")
    def models(self):
        return _MODELS

    @pytest.fixture(scope="class")
    def selectors(self):
        return bad_negative_count_selectors_yml

    def test_negative_count_is_rejected(self, project):
        _, log_output = run_dbt_and_capture(
            ["ls", "--selector", "bad_selector"], expect_pass=False
        )
        assert "non-negative integer" in log_output


class TestSampleSelectorMissingSeed:
    @pytest.fixture(scope="class")
    def models(self):
        return _MODELS

    @pytest.fixture(scope="class")
    def selectors(self):
        return missing_seed_selectors_yml

    def test_missing_seed_is_rejected(self, project):
        _, log_output = run_dbt_and_capture(
            ["ls", "--selector", "bad_selector"], expect_pass=False
        )
        assert "seed" in log_output
