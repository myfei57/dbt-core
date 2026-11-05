from types import SimpleNamespace

import pytest
from click.exceptions import BadParameter

from dbt.artifacts.schemas.results import NodeStatus
from dbt.cli.option_types import RetryStatusType
from dbt.cli.types import Command
from dbt.task.retry import (
    CMD_DICT,
    RETRYABLE_STATUSES,
    TASK_DICT,
    _get_retry_batch_maps,
    _get_retryable_statuses,
    _validate_retry_status_filter,
)
from dbt_common.exceptions import DbtRuntimeError

EXCLUDED_COMMANDS = {
    "clean",
    "debug",
    "deps",
    "freshness",
    "init",
    "list",
    "parse",
    "retry",
    "show",
    "serve",
}


def test_task_cmd_dicts():
    assert TASK_DICT.keys() == CMD_DICT.keys()


def test_exhaustive_commands():
    assert set(TASK_DICT.keys()).union(EXCLUDED_COMMANDS) == set(i.value.lower() for i in Command)


class TestGetRetryableStatuses:
    def test_default_retryable_statuses(self):
        statuses = _get_retryable_statuses(False)
        assert statuses == RETRYABLE_STATUSES
        # Successful statuses are never retryable by default.
        for success_status in (
            NodeStatus.Success,
            NodeStatus.Pass,
            NodeStatus.NoOp,
            NodeStatus.Reused,
            NodeStatus.Warn,
        ):
            assert success_status not in statuses

    def test_warn_error_adds_warn(self):
        statuses = _get_retryable_statuses(True)
        assert NodeStatus.Warn in statuses
        assert statuses == RETRYABLE_STATUSES | {NodeStatus.Warn}

    def test_does_not_mutate_base_set(self):
        before = set(RETRYABLE_STATUSES)
        _get_retryable_statuses(True)
        assert set(RETRYABLE_STATUSES) == before
        assert NodeStatus.Warn not in RETRYABLE_STATUSES


class TestValidateRetryStatusFilter:
    def test_no_filter_returns_none(self):
        # When unspecified, the full retryable set continues to apply.
        assert _validate_retry_status_filter((), RETRYABLE_STATUSES) is None

    def test_valid_statuses_collapse_to_a_set(self):
        # Repeated values are treated as a set.
        result = _validate_retry_status_filter(
            (NodeStatus.Error, NodeStatus.Error, NodeStatus.Skipped),
            RETRYABLE_STATUSES,
        )
        assert result == frozenset({NodeStatus.Error, NodeStatus.Skipped})

    @pytest.mark.parametrize(
        "status",
        (NodeStatus.Success, NodeStatus.Pass, NodeStatus.NoOp, NodeStatus.Reused),
    )
    def test_successful_statuses_are_rejected(self, status):
        with pytest.raises(DbtRuntimeError, match="Cannot retry nodes with status"):
            _validate_retry_status_filter((status,), RETRYABLE_STATUSES)

    def test_warn_rejected_without_warn_error(self):
        with pytest.raises(DbtRuntimeError, match="--warn-error"):
            _validate_retry_status_filter((NodeStatus.Warn,), RETRYABLE_STATUSES)

    def test_warn_allowed_with_warn_error(self):
        retryable = _get_retryable_statuses(True)
        assert _validate_retry_status_filter((NodeStatus.Warn,), retryable) == frozenset(
            {NodeStatus.Warn}
        )

    def test_mix_of_valid_and_invalid_is_rejected(self):
        with pytest.raises(DbtRuntimeError, match="Cannot retry nodes with status 'success'"):
            _validate_retry_status_filter(
                (NodeStatus.Error, NodeStatus.Success), RETRYABLE_STATUSES
            )


class TestRetryStatusType:
    @pytest.mark.parametrize(
        "raw,expected",
        (
            ("error", NodeStatus.Error),
            ("ERROR", NodeStatus.Error),
            (" Fail ", NodeStatus.Fail),
            ("runtime_error", NodeStatus.RuntimeErr),
            ("runtime-error", NodeStatus.RuntimeErr),
            ("partial_success", NodeStatus.PartialSuccess),
            ("partial-success", NodeStatus.PartialSuccess),
            ("warn", NodeStatus.Warn),
        ),
    )
    def test_convert(self, raw, expected):
        assert RetryStatusType().convert(raw, None, None) == expected

    def test_convert_tuple(self):
        converted = RetryStatusType().convert(("error", "skipped"), None, None)
        assert converted == (NodeStatus.Error, NodeStatus.Skipped)

    @pytest.mark.parametrize("raw", ("bogus", "123", "errors"))
    def test_unknown_status_fails(self, raw):
        with pytest.raises(BadParameter, match="not a valid node status"):
            RetryStatusType().convert(raw, None, None)


class TestGetRetryBatchMaps:
    def _result(self, unique_id, batch_results=None, overload_results=None):
        return SimpleNamespace(
            unique_id=unique_id,
            batch_results=batch_results,
            overload_results=overload_results,
        )

    def test_only_selected_microbatch_models_have_their_failed_batches_retried(self):
        failed_batch = object()
        successful_batch = object()
        selected_partial = SimpleNamespace(
            successful=[successful_batch, successful_batch], failed=[failed_batch]
        )
        unselected_partial = SimpleNamespace(successful=[successful_batch], failed=[failed_batch])
        previous_results = SimpleNamespace(
            results=[
                self._result("model.test.selected", batch_results=selected_partial),
                self._result("model.test.unselected", batch_results=unselected_partial),
            ]
        )

        batch_map, _ = _get_retry_batch_maps(previous_results, {"model.test.selected"})

        # Only the selected model's batch info is carried into the retry...
        assert set(batch_map.keys()) == {"model.test.selected"}
        # ...and it carries forward the successful batches, so the runner only
        # re-attempts the previously failed batch.
        assert batch_map["model.test.selected"].successful == [
            successful_batch,
            successful_batch,
        ]
        assert batch_map["model.test.selected"].failed == [failed_batch]

    def test_successful_only_batches_are_not_retried(self):
        # All batches succeeded -> nothing to retry for the model.
        batches = SimpleNamespace(successful=[object(), object()], failed=[])
        previous_results = SimpleNamespace(
            results=[self._result("model.test.m", batch_results=batches)]
        )
        batch_map, _ = _get_retry_batch_maps(previous_results, {"model.test.m"})
        assert batch_map == {}

    def test_all_failed_batches_omit_batch_info(self):
        # No successful batches -> batch info is omitted so that a full refresh
        # remains possible.
        batches = SimpleNamespace(successful=[], failed=[object(), object()])
        previous_results = SimpleNamespace(
            results=[self._result("model.test.m", batch_results=batches)]
        )
        batch_map, _ = _get_retry_batch_maps(previous_results, {"model.test.m"})
        assert batch_map == {}

    def test_overload_map_restricted_to_selected_nodes(self):
        overloads = SimpleNamespace(failed=[object()], successful=[object()])
        previous_results = SimpleNamespace(
            results=[
                self._result("function.test.selected", overload_results=overloads),
                self._result("function.test.unselected", overload_results=overloads),
            ]
        )
        _, overload_map = _get_retry_batch_maps(previous_results, {"function.test.selected"})
        assert set(overload_map.keys()) == {"function.test.selected"}


class TestRetryInterpretResults:
    def _task(self):
        from dbt.task.retry import RetryTask

        class FailingTask:
            @classmethod
            def interpret_results(cls, results):
                return False

        task = RetryTask.__new__(RetryTask)
        task.task_class = FailingTask
        task.no_matching_nodes = False
        return task, FailingTask

    def test_no_matching_nodes_is_success_even_when_previous_run_failed(self):
        task, failing_task = self._task()
        previous_failed_results = SimpleNamespace(
            results=[SimpleNamespace(status=NodeStatus.Error)]
        )
        assert failing_task.interpret_results(previous_failed_results) is False

        task.no_matching_nodes = True
        assert task.interpret_results(previous_failed_results) is True

    def test_delegates_when_nodes_were_retried(self):
        task, _ = self._task()
        assert task.interpret_results(SimpleNamespace(results=[])) is False


class TestRetryableResultSelection:
    def _result(self, unique_id, status):
        return SimpleNamespace(
            unique_id=unique_id,
            status=status,
            batch_results=None,
            overload_results=None,
        )

    def _task(self, results, command_name="run", status_filter=None):
        # Build the minimal RetryTask surface used by the selection logic
        # without invoking its heavy __init__.
        from dbt.task.retry import RetryTask

        task = RetryTask.__new__(RetryTask)
        task.previous_command_name = command_name
        task.retryable_statuses = RETRYABLE_STATUSES
        task.retry_status_filter = status_filter
        task.previous_results = SimpleNamespace(results=results)
        return task

    def test_full_retryable_set_is_used_without_filter(self):
        task = self._task(
            [
                self._result("model.test.ok", NodeStatus.Success),
                self._result("model.test.bad", NodeStatus.Error),
                self._result("model.test.skipped", NodeStatus.Skipped),
                self._result("test.test.t", NodeStatus.Warn),
            ]
        )
        unique_ids = {
            result.unique_id
            for result in task.previous_results.results
            if task._is_retryable_result(result)
        }
        assert unique_ids == {"model.test.bad", "model.test.skipped"}

    def test_status_filter_restricts_selection(self):
        task = self._task(
            [
                self._result("model.test.bad", NodeStatus.Error),
                self._result("model.test.skipped", NodeStatus.Skipped),
            ],
            status_filter=frozenset({NodeStatus.Error}),
        )
        unique_ids = {
            result.unique_id
            for result in task.previous_results.results
            if task._is_retryable_result(result)
        }
        assert unique_ids == {"model.test.bad"}

    def test_operation_nodes_excluded_unless_run_operation(self):
        results = [
            self._result("operation.test.on-run-start-0", NodeStatus.Error),
            self._result("model.test.m", NodeStatus.Error),
        ]

        run_task = self._task(results, command_name="run")
        assert {r.unique_id for r in results if run_task._is_retryable_result(r)} == {
            "model.test.m"
        }

        run_op_task = self._task(results, command_name="run-operation")
        assert {r.unique_id for r in results if run_op_task._is_retryable_result(r)} == {
            "operation.test.on-run-start-0",
            "model.test.m",
        }
