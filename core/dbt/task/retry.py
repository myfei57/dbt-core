from datetime import datetime, timezone
from pathlib import Path
from typing import FrozenSet, Iterable, Optional, Set

from click import get_current_context
from click.core import ParameterSource

from dbt.artifacts.schemas.results import NodeStatus
from dbt.cli.flags import Flags
from dbt.cli.types import Command as CliCommand
from dbt.config import RuntimeConfig
from dbt.config.catalogs import load_catalogs
from dbt.constants import RUN_RESULTS_FILE_NAME
from dbt.contracts.state import load_result_state
from dbt.flags import get_flags, set_flags
from dbt.graph import GraphQueue
from dbt.parser.manifest import parse_manifest
from dbt.task.base import ConfiguredTask
from dbt.task.build import BuildTask
from dbt.task.clone import CloneTask
from dbt.task.compile import CompileTask
from dbt.task.docs.generate import GenerateTask
from dbt.task.run import RunTask
from dbt.task.run_operation import RunOperationTask
from dbt.task.seed import SeedTask
from dbt.task.snapshot import SnapshotTask
from dbt.task.test import TestTask
from dbt_common.events.functions import fire_event
from dbt_common.events.types import Note
from dbt_common.exceptions import DbtRuntimeError

RETRYABLE_STATUSES: FrozenSet[NodeStatus] = frozenset(
    {
        NodeStatus.Error,
        NodeStatus.Fail,
        NodeStatus.Skipped,
        NodeStatus.RuntimeErr,
        NodeStatus.PartialSuccess,
    }
)
IGNORE_PARENT_FLAGS = {
    "log_path",
    "output_path",
    "profiles_dir",
    "profiles_dir_exists_false",
    "project_dir",
    "defer_state",
    "deprecated_state",
    "target_path",
    "warn_error",
}

ALLOW_CLI_OVERRIDE_FLAGS = {"vars", "threads"}

TASK_DICT = {
    "build": BuildTask,
    "compile": CompileTask,
    "clone": CloneTask,
    "generate": GenerateTask,
    "seed": SeedTask,
    "snapshot": SnapshotTask,
    "test": TestTask,
    "run": RunTask,
    "run-operation": RunOperationTask,
}

CMD_DICT = {
    "build": CliCommand.BUILD,
    "compile": CliCommand.COMPILE,
    "clone": CliCommand.CLONE,
    "generate": CliCommand.DOCS_GENERATE,
    "seed": CliCommand.SEED,
    "snapshot": CliCommand.SNAPSHOT,
    "test": CliCommand.TEST,
    "run": CliCommand.RUN,
    "run-operation": CliCommand.RUN_OPERATION,
}


def _get_retryable_statuses(warn_error: bool) -> FrozenSet[NodeStatus]:
    """Return the set of statuses that can be retried under the current
    invocation configuration.

    ``NodeStatus.Warn`` is only retryable when --warn-error is set, in which
    case nodes that previously warned are re-attempted (and may fail).
    """
    statuses: Set[NodeStatus] = set(RETRYABLE_STATUSES)
    if warn_error:
        statuses.add(NodeStatus.Warn)
    return frozenset(statuses)


def _format_statuses(statuses: Iterable[NodeStatus]) -> str:
    quoted = (f"'{status.value}'" for status in sorted(statuses, key=lambda s: s.value))
    return ", ".join(quoted)


def _get_retry_batch_maps(previous_results, selected_unique_ids: Set[str]):
    """Build the ``batch_map`` (microbatch) and ``overload_map`` (functions)
    for the retry, restricted to the nodes selected for retry.

    Only nodes in ``selected_unique_ids`` (i.e. whose previous status matched
    the --retry-statuses filter) can have their batches/overloads retried. For
    microbatch models only previously failed batches are retried; already
    successful batches are carried forward. Batch info is deliberately omitted
    when there were no successful batches, because providing it forces the
    microbatch process into incremental mode while a full refresh may be
    required.
    """
    batch_map = {
        result.unique_id: result.batch_results
        for result in previous_results.results
        if result.unique_id in selected_unique_ids
        and result.batch_results is not None
        and len(result.batch_results.successful) != 0
        and len(result.batch_results.failed) > 0
    }
    overload_map = {
        result.unique_id: result.overload_results
        for result in previous_results.results
        if result.unique_id in selected_unique_ids
        and result.overload_results is not None
        and len(result.overload_results.failed) > 0
    }
    return batch_map, overload_map


def _validate_retry_status_filter(
    requested_statuses: Iterable[NodeStatus],
    retryable_statuses: FrozenSet[NodeStatus],
) -> Optional[FrozenSet[NodeStatus]]:
    """Validate the values passed to ``dbt retry --retry-statuses``.

    Unknown statuses are rejected by the click parameter type; here we reject
    statuses that are known but not retryable under the current invocation
    (e.g. successful nodes, or 'warn' without --warn-error). Repeated values
    are treated as a set. Returns ``None`` when no status filter was provided,
    in which case the full retryable set applies.
    """
    if not requested_statuses:
        return None

    requested = frozenset(requested_statuses)
    invalid = requested - retryable_statuses
    if invalid:
        msg = (
            f"Cannot retry nodes with status {_format_statuses(invalid)}: `dbt retry` only "
            f"retries nodes that did not succeed in the previous run. Retryable statuses for "
            f"the current invocation are: {_format_statuses(retryable_statuses)}."
        )
        if NodeStatus.Warn in invalid:
            msg += " Nodes with status 'warn' are only retryable when using --warn-error."
        raise DbtRuntimeError(msg)

    return requested


class RetryTask(ConfiguredTask):
    def __init__(self, args: Flags, config: RuntimeConfig) -> None:
        # Resolve which statuses are retryable under the current invocation and
        # validate any explicit status filter before doing any other work (e.g.
        # loading the previous run results), so invalid input is rejected up
        # front, before execution.
        self.retryable_statuses = _get_retryable_statuses(bool(args.warn_error))
        self.retry_status_filter = _validate_retry_status_filter(
            getattr(args, "retry_statuses", None) or (), self.retryable_statuses
        )

        # Set when a status filter matched no nodes and retry ends early as a
        # success without running anything.
        self.no_matching_nodes = False

        # load previous run results
        state_path = args.state or config.target_path
        self.previous_results = load_result_state(
            Path(config.project_root) / Path(state_path) / RUN_RESULTS_FILE_NAME
        )
        if not self.previous_results:
            raise DbtRuntimeError(
                f"Could not find previous run in '{state_path}' target directory"
            )
        self.previous_args = self.previous_results.args
        self.previous_command_name = self.previous_args.get("which")

        # Reslove flags and config
        cli_command = CMD_DICT.get(self.previous_command_name)  # type: ignore
        # Remove these args when their default values are present, otherwise they'll raise an exception
        args_to_remove = {
            "show": lambda x: True,
            "resource_types": lambda x: x == [],
            "warn_error_options": lambda x: x == {"warn": [], "error": [], "silence": []},
        }
        for k, v in args_to_remove.items():
            if k in self.previous_args and v(self.previous_args[k]):
                del self.previous_args[k]
        previous_args = {
            k: v for k, v in self.previous_args.items() if k not in IGNORE_PARENT_FLAGS
        }
        click_context = get_current_context()
        current_args = {
            k: v
            for k, v in args.__dict__.items()
            if k in IGNORE_PARENT_FLAGS
            or (
                click_context.get_parameter_source(k) == ParameterSource.COMMANDLINE
                and k in ALLOW_CLI_OVERRIDE_FLAGS
            )
        }
        combined_args = {**previous_args, **current_args}
        retry_flags = Flags.from_dict(cli_command, combined_args)  # type: ignore
        set_flags(retry_flags)
        retry_config = RuntimeConfig.from_args(args=retry_flags)

        # Parse manifest using resolved config/flags
        manifest = parse_manifest(retry_config, False, True, retry_flags.write_json, [])  # type: ignore
        catalogs = load_catalogs(
            str(retry_config.project_root),
            retry_config.project_name,
            retry_config.cli_vars,
        )
        super().__init__(args, retry_config, manifest, catalogs=catalogs)
        self.task_class = TASK_DICT.get(self.previous_command_name)  # type: ignore

    def _is_retryable_result(self, result) -> bool:
        # Avoid retrying operation nodes unless we are retrying the run-operation command
        is_operation_node = result.unique_id.startswith("operation.") and (
            self.previous_command_name != "run-operation"
        )
        statuses_to_retry = self.retry_status_filter or self.retryable_statuses
        return result.status in statuses_to_retry and not is_operation_node

    def run(self):
        unique_ids = {
            result.unique_id
            for result in self.previous_results.results
            if self._is_retryable_result(result)
        }

        # We need this so that re-running of a microbatch model will only rerun
        # batches that previously failed. Both maps are restricted to the
        # selected unique_ids, so a model whose status was not filtered in is
        # never re-run (batch or otherwise), and previously successful batches
        # of selected models are carried forward instead of being re-run.
        batch_map, overload_map = _get_retry_batch_maps(self.previous_results, unique_ids)

        # Tasks without get_graph_queue (e.g. run-operation) and no failed nodes to retry.
        if not unique_ids and not hasattr(self.task_class, "get_graph_queue"):
            if self.retry_status_filter is not None:
                # A filter was given but matched nothing: succeed without
                # re-running the operation, even if the previous run failed.
                self.no_matching_nodes = True
                fire_event(
                    Note(msg=self._no_matching_statuses_message()),
                )
            # Return early with the previous results as the past invocation was successful
            return self.previous_results

        class TaskWrapper(self.task_class):
            def get_graph_queue(self):
                new_graph = self.graph.get_subset_graph(unique_ids)
                return GraphQueue(
                    new_graph.graph,
                    self.manifest,
                    unique_ids,
                )

        task = TaskWrapper(
            get_flags(),
            self.config,
            self.manifest,
            catalogs=self.catalogs,
        )

        if (
            not unique_ids
            and self.retry_status_filter is not None
            # `dbt docs generate` reports results via a CatalogArtifact rather
            # than an empty execution result; let it follow the normal task
            # path (which itself handles an empty node selection).
            and self.task_class is not GenerateTask
        ):
            # A status filter was explicitly provided but matched no nodes from
            # the previous run. Succeed without executing any models or hooks,
            # and clearly communicate that no retry targets were matched.
            self.no_matching_nodes = True
            fire_event(
                Note(msg=self._no_matching_statuses_message()),
            )
            return task.get_result(
                results=[],
                elapsed_time=0.0,
                generated_at=datetime.now(timezone.utc).replace(tzinfo=None),
            )

        if self.task_class == RunTask or self.task_class == BuildTask:
            task.batch_map = batch_map
            task.original_invocation_started_at = (
                self.previous_results.metadata.invocation_started_at
                or self.previous_results.metadata.generated_at
            )

        task.overload_map = overload_map

        return_value = task.run()
        return return_value

    def _no_matching_statuses_message(self) -> str:
        return (
            "No nodes from the previous run matched the requested retry status(es): "
            f"{_format_statuses(self.retry_status_filter or ())}. Nothing will be retried."
        )

    def interpret_results(self, *args, **kwargs):
        # An explicit status filter that matched nothing is a successful
        # no-op, even if the previous run itself failed.
        if self.no_matching_nodes:
            return True
        return self.task_class.interpret_results(*args, **kwargs)
