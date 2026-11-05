import pytest

from dbt.cli.exceptions import DbtUsageException
from dbt.contracts.results import RunStatus, TestStatus
from dbt.exceptions import DbtRuntimeError
from dbt.tests.util import run_dbt, run_dbt_and_capture, write_file
from tests.functional.retry.fixtures import (
    macros__alter_timezone_sql,
    models__sample_model,
    models__second_model,
    models__union_model,
    schema_yml,
)


class BaseRetryStatusesTest:
    @pytest.fixture(scope="class")
    def models(self):
        return {
            "sample_model.sql": models__sample_model,
            "second_model.sql": models__second_model,
            "union_model.sql": models__union_model,
            "schema.yml": schema_yml,
        }


class TestRetryStatusesErrorOnly(BaseRetryStatusesTest):
    def test_error_only(self, project):
        # Regular build: sample_model errors, union_model and two tests are
        # skipped, second_model succeeds and one test warns.
        run_dbt(["build"], expect_pass=False)

        # Only nodes whose previous status was "error" are retried. Skipped and
        # warning nodes are excluded.
        results = run_dbt(["retry", "--retry-statuses", "error"], expect_pass=False)
        assert {n.node.name: n.status for n in results.results} == {
            "sample_model": RunStatus.Error,
        }


class TestRetryStatusesMultiple(BaseRetryStatusesTest):
    def test_multiple_space_separated_statuses(self, project):
        run_dbt(["build"], expect_pass=False)

        # Fix sample_model so the selected error/skipped nodes can succeed.
        write_file("select 1 as id, 1 as foo", "models", "sample_model.sql")

        # Multiple statuses may be provided space-separated on a single flag.
        results = run_dbt(["retry", "--retry-statuses", "error", "skipped"])
        assert {n.node.name: n.status for n in results.results} == {
            "sample_model": RunStatus.Success,
            "union_model": RunStatus.Success,
            "accepted_values_union_model_sum3__False__3": TestStatus.Pass,
            "accepted_values_sample_model_foo__False__3": TestStatus.Warn,
        }


class TestRetryStatusesRepeatedAsSet(BaseRetryStatusesTest):
    def test_repeated_statuses_treated_as_set(self, project):
        run_dbt(["build"], expect_pass=False)

        # Repeated values (across and within flag usages) collapse to a set.
        results = run_dbt(
            [
                "retry",
                "--retry-statuses",
                "error",
                "error",
                "--retry-statuses",
                "error",
            ],
            expect_pass=False,
        )
        assert {n.node.name: n.status for n in results.results} == {
            "sample_model": RunStatus.Error,
        }


class TestRetryStatusesNoMatch(BaseRetryStatusesTest):
    def test_no_match_succeeds_and_explains(self, project):
        run_dbt(["build"], expect_pass=False)

        # No node from the previous run has a "fail" status.
        results, log_output = run_dbt_and_capture(["retry", "--retry-statuses", "fail"])
        assert len(results) == 0
        assert "matched the requested retry status(es): 'fail'" in log_output
        assert "Nothing will be retried" in log_output


class TestRetryStatusesNoMatchSkipsHooks:
    @pytest.fixture(scope="class")
    def project_config_update(self):
        # This hook is guaranteed to fail; the retry must not run it when no
        # retry targets matched.
        return {"on-run-start": ["select no_such_column from no_such_relation"]}

    @pytest.fixture(scope="class")
    def models(self):
        return {"sample_model.sql": models__sample_model}

    def test_no_match_does_not_run_hooks(self, project):
        run_dbt(["run"], expect_pass=False)

        results, log_output = run_dbt_and_capture(["retry", "--retry-statuses", "fail"])
        assert len(results) == 0
        assert "matched the requested retry status(es): 'fail'" in log_output
        assert "START hook" not in log_output


class TestRetryStatusesSuccessRejectedBeforeExecution:
    @pytest.fixture(scope="class")
    def models(self):
        return {"sample_model.sql": models__second_model}

    def test_success_status_is_rejected(self, project):
        # There is no previous run here; validating the filter must fail before
        # dbt even looks for previous run results.
        with pytest.raises(
            DbtRuntimeError,
            match="Cannot retry nodes with status 'pass', 'success'",
        ):
            run_dbt(["retry", "--retry-statuses", "success", "pass"])


class TestRetryStatusesUnknownRejected:
    @pytest.fixture(scope="class")
    def models(self):
        return {"sample_model.sql": models__second_model}

    def test_unknown_status_is_rejected(self, project):
        with pytest.raises(DbtUsageException, match="not a valid node status"):
            run_dbt(["retry", "--retry-statuses", "totally-made-up"])


class TestRetryStatusesWarn:
    @pytest.fixture(scope="class")
    def models(self):
        return {
            "second_model.sql": models__second_model,
            "schema.yml": schema_yml,
        }

    def test_warn_requires_warn_error(self, project):
        run_dbt(["build", "--select", "second_model"])

        # "warn" is not retryable on a normal invocation.
        with pytest.raises(DbtRuntimeError, match="--warn-error"):
            run_dbt(["retry", "--retry-statuses", "warn"])

        # With --warn-error, the previously warning node is selected and is
        # re-attempted (and fails because warnings are now errors).
        results = run_dbt(["--warn-error", "retry", "--retry-statuses", "warn"], expect_pass=False)
        assert {n.node.name: n.status for n in results.results} == {
            "accepted_values_second_model_bar__False__3": TestStatus.Fail,
        }


class TestRetryStatusesInheritsPreviousFlags:
    @pytest.fixture(scope="class")
    def models(self):
        return {
            "sample_model.sql": (
                "select {{ var('myvar_a', '1') + var('myvar_b', '2') }} as mycol"
            ),
        }

    def test_filter_uses_previous_command_and_flags(self, project):
        run_dbt(["run", "--vars", '{"myvar_a": "12", "myvar_b": "3 4"}'], expect_pass=False)

        # The invalid --vars from the previous run are still restored, so the
        # filtered retry keeps failing.
        run_dbt(["retry", "--retry-statuses", "error"], expect_pass=False)

        # Explicit CLI overrides remain allowed alongside the status filter.
        results = run_dbt(
            [
                "retry",
                "--retry-statuses",
                "error",
                "--vars",
                '{"myvar_a": "12", "myvar_b": "34"}',
            ]
        )
        assert len(results) == 1


class TestRetryStatusesRunOperationNoMatch:
    @pytest.fixture(scope="class")
    def macros(self):
        return {"alter_timezone.sql": macros__alter_timezone_sql}

    def test_failed_run_operation_no_match_succeeds(self, project):
        # The run-operation fails, so the previous run is not successful.
        results, _ = run_dbt_and_capture(
            ["run-operation", "alter_timezone", "--args", "{timezone: abc}"],
            expect_pass=False,
        )
        assert {n.unique_id: n.status for n in results.results} == {
            "macro.test.alter_timezone": RunStatus.Error
        }

        # A filter that doesn't match the failed operation ends successfully,
        # without re-executing the operation.
        results, log_output = run_dbt_and_capture(["retry", "--retry-statuses", "fail"])
        assert "running a macro!" not in log_output
        assert "matched the requested retry status(es): 'fail'" in log_output
        assert {n.unique_id: n.status for n in results.results} == {
            "macro.test.alter_timezone": RunStatus.Error
        }
