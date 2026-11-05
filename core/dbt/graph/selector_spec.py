import os
import random
import re
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple, Union

from dbt.exceptions import InvalidSelectorError
from dbt.flags import get_flags
from dbt_common.dataclass_schema import StrEnum, dbtClassMixin
from dbt_common.exceptions import DbtInternalError, DbtRuntimeError, DbtValidationError

from .graph import UniqueId
from .selector_methods import MethodName

RAW_SELECTOR_PATTERN = re.compile(
    r"\A"
    r"(?P<childrens_parents>(\@))?"
    r"(?P<parents>((?P<parents_depth>(\d*))\+))?"
    r"((?P<method>([\w.]+)):)?(?P<value>(.*?))"
    r"(?P<children>(\+(?P<children_depth>(\d*))))?"
    r"\Z"
)
SELECTOR_METHOD_SEPARATOR = "."


class IndirectSelection(StrEnum):
    Eager = "eager"
    Cautious = "cautious"
    Buildable = "buildable"
    Empty = "empty"


def _probably_path(value: str):
    """Decide if the value is probably a path. Windows has two path separators, so
    we should check both sep ('\\') and altsep ('/') there.
    """
    if os.path.sep in value:
        return True
    elif os.path.altsep is not None and os.path.altsep in value:
        return True
    else:
        return False


def _match_to_int(match: Dict[str, str], key: str) -> Optional[int]:
    raw = match.get(key)
    # turn the empty string into None, too.
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise DbtRuntimeError(f"Invalid node spec - could not handle parent depth {raw}") from exc


SelectionSpec = Union[
    "SelectionCriteria",
    "SelectionIntersection",
    "SelectionDifference",
    "SelectionUnion",
    "SelectionSample",
]


@dataclass
class SelectionCriteria:
    raw: Any
    method: MethodName
    method_arguments: List[str]
    value: Any
    childrens_parents: bool
    parents: bool
    parents_depth: Optional[int]
    children: bool
    children_depth: Optional[int]
    indirect_selection: IndirectSelection = IndirectSelection.Eager

    def __post_init__(self):
        if self.children and self.childrens_parents:
            raise DbtRuntimeError(
                f'Invalid node spec {self.raw} - "@" prefix and "+" suffix ' "are incompatible"
            )

    @classmethod
    def default_method(cls, value: str) -> MethodName:
        if _probably_path(value):
            return MethodName.Path
        elif value.lower().endswith((".sql", ".py", ".csv")):
            return MethodName.File
        else:
            return MethodName.FQN

    @classmethod
    def parse_method(cls, groupdict: Dict[str, Any]) -> Tuple[MethodName, List[str]]:
        raw_method = groupdict.get("method")
        if raw_method is None:
            return cls.default_method(groupdict["value"]), []

        method_parts: List[str] = raw_method.split(SELECTOR_METHOD_SEPARATOR)
        try:
            method_name = MethodName(method_parts[0])
        except ValueError as exc:
            raise InvalidSelectorError(f"'{method_parts[0]}' is not a valid method name") from exc

        # Following is for cases like config.severity and config.materialized
        method_arguments: List[str] = method_parts[1:]

        return method_name, method_arguments

    @classmethod
    def selection_criteria_from_dict(
        cls,
        raw: Any,
        dct: Dict[str, Any],
    ) -> "SelectionCriteria":
        if "value" not in dct:
            raise DbtRuntimeError(f'Invalid node spec "{raw}" - no search value!')
        method_name, method_arguments = cls.parse_method(dct)

        parents_depth = _match_to_int(dct, "parents_depth")
        children_depth = _match_to_int(dct, "children_depth")

        # If defined field in selector, override CLI flag
        indirect_selection = IndirectSelection(
            dct.get("indirect_selection", get_flags().INDIRECT_SELECTION)
        )

        return cls(
            raw=raw,
            method=method_name,
            method_arguments=method_arguments,
            value=dct["value"],
            childrens_parents=bool(dct.get("childrens_parents")),
            parents=bool(dct.get("parents")),
            parents_depth=parents_depth,
            children=bool(dct.get("children")),
            children_depth=children_depth,
            indirect_selection=indirect_selection,
        )

    @classmethod
    def dict_from_single_spec(cls, raw: str):
        result = RAW_SELECTOR_PATTERN.match(raw)
        if result is None:
            return {"error": "Invalid selector spec"}
        dct: Dict[str, Any] = result.groupdict()
        method_name, method_arguments = cls.parse_method(dct)
        meth_name = str(method_name)
        if method_arguments:
            meth_name += "." + ".".join(method_arguments)
        dct["method"] = meth_name
        dct = {k: v for k, v in dct.items() if (v is not None and v != "")}
        if "childrens_parents" in dct:
            dct["childrens_parents"] = bool(dct.get("childrens_parents"))
        if "parents" in dct:
            dct["parents"] = bool(dct.get("parents"))
        if "children" in dct:
            dct["children"] = bool(dct.get("children"))
        return dct

    @classmethod
    def from_single_spec(cls, raw: str) -> "SelectionCriteria":
        result = RAW_SELECTOR_PATTERN.match(raw)
        if result is None:
            # bad spec!
            raise DbtRuntimeError(f'Invalid selector spec "{raw}"')

        return cls.selection_criteria_from_dict(raw, result.groupdict())


class BaseSelectionGroup(dbtClassMixin, Iterable[SelectionSpec], metaclass=ABCMeta):
    def __init__(
        self,
        components: Iterable[SelectionSpec],
        indirect_selection: IndirectSelection = IndirectSelection.Eager,
        expect_exists: bool = False,
        raw: Any = None,
    ) -> None:
        self.components: List[SelectionSpec] = list(components)
        self.expect_exists = expect_exists
        self.raw = raw
        self.indirect_selection = indirect_selection

    def __iter__(self) -> Iterator[SelectionSpec]:
        for component in self.components:
            yield component

    @abstractmethod
    def combine_selections(
        self,
        selections: List[Set[UniqueId]],
    ) -> Set[UniqueId]:
        raise NotImplementedError("_combine_selections not implemented!")

    def combined(self, selections: List[Set[UniqueId]]) -> Set[UniqueId]:
        if not selections:
            return set()

        return self.combine_selections(selections)


class SelectionIntersection(BaseSelectionGroup):
    def combine_selections(
        self,
        selections: List[Set[UniqueId]],
    ) -> Set[UniqueId]:
        return set.intersection(*selections)


class SelectionDifference(BaseSelectionGroup):
    def combine_selections(
        self,
        selections: List[Set[UniqueId]],
    ) -> Set[UniqueId]:
        return set.difference(*selections)


class SelectionUnion(BaseSelectionGroup):
    def combine_selections(
        self,
        selections: List[Set[UniqueId]],
    ) -> Set[UniqueId]:
        return set.union(*selections)


def validate_sample_count(value: Any) -> int:
    """Validate the ``count`` of a sample selector. Only non-negative integers
    are accepted (booleans are rejected even though they are integers in
    Python)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise DbtValidationError(
            f"The 'count' of a sample selector must be an integer, got "
            f"{value!r} of type {type(value).__name__}"
        )
    if value < 0:
        raise DbtValidationError(
            f"The 'count' of a sample selector must be a non-negative integer, got {value}"
        )
    return value


def validate_sample_seed(value: Any) -> int:
    """Validate the ``seed`` of a sample selector. The seed is required and
    must be an integer so that sampling is reproducible across processes."""
    if value is None:
        raise DbtValidationError(
            "A sample selector requires an explicit integer 'seed' so that "
            "sampling is reproducible"
        )
    if isinstance(value, bool) or not isinstance(value, int):
        raise DbtValidationError(
            f"The 'seed' of a sample selector must be an integer, got "
            f"{value!r} of type {type(value).__name__}"
        )
    return value


def sample_selection(
    nodes: Iterable[UniqueId],
    count: int,
    seed: int,
) -> Set[UniqueId]:
    """Deterministically pick ``count`` distinct nodes from ``nodes``.

    The candidate unique IDs are deduplicated and sorted before sampling, so
    the result depends only on the candidate set, ``count`` and ``seed`` --
    never on the iteration order of manifest dictionaries or set hashing. The
    same inputs always produce the same output, including in different
    processes.

    - ``count`` of 0 returns an empty set
    - ``count`` greater than or equal to the number of candidates returns all
      of them
    """
    candidates = sorted(set(nodes))
    if count == 0 or not candidates:
        return set()
    if count >= len(candidates):
        return set(candidates)
    rng = random.Random(seed)
    return set(rng.sample(candidates, count))


class SelectionSample(BaseSelectionGroup):
    """Reproducibly sample a fixed number of distinct nodes from the directly
    selected nodes of a single wrapped selection definition.

    The wrapped definition is available as the single component of this
    group. Indirect test expansion is applied to the sampled nodes (not to
    the wrapped definition's full selection) using this spec's
    ``indirect_selection`` setting, so nodes that were not sampled are not
    selected merely because they were part of the wrapped definition.
    """

    def __init__(
        self,
        components: Iterable[SelectionSpec],
        count: int,
        seed: int,
        indirect_selection: IndirectSelection = IndirectSelection.Eager,
        expect_exists: bool = False,
        raw: Any = None,
    ) -> None:
        super().__init__(
            components=components,
            indirect_selection=indirect_selection,
            expect_exists=expect_exists,
            raw=raw,
        )
        self.count = count
        self.seed = seed

    def combine_selections(
        self,
        selections: List[Set[UniqueId]],
    ) -> Set[UniqueId]:
        # A sample wraps a single definition and is resolved directly by the
        # NodeSelector; the generic group combination path never applies.
        raise DbtInternalError("SelectionSample is resolved directly by the NodeSelector")

    def sample(self, nodes: Iterable[UniqueId]) -> Set[UniqueId]:
        return sample_selection(nodes, self.count, self.seed)
