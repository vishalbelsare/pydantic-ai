from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Self

from pydantic_graph.v2.id_types import NodeId
from pydantic_graph.v2.paths import Path, PathBuilder
from pydantic_graph.v2.transform import TransformFunction
from pydantic_graph.v2.util import TypeOrTypeExpression

if TYPE_CHECKING:
    from pydantic_graph.v2.node_types import DestinationNode


@dataclass
class Decision[StateT, DepsT, SourceT]:
    id: NodeId
    branches: list[DecisionBranch]
    # TODO: Add a field for the label for the input edge
    note: str | None  # TODO: Add a way to set this in the graph.add_decision method

    def branch[BranchSourceT](
        self,
        source: TypeOrTypeExpression[BranchSourceT],
        *,
        matches: Callable[[Any], bool] | None = None,
    ) -> DecisionBranchBuilder[StateT, DepsT, BranchSourceT, BranchSourceT, SourceT]:
        new_path_builder = PathBuilder[StateT, DepsT, BranchSourceT](working_items=[])
        return DecisionBranchBuilder(decision=self, source=source, matches=matches, path_builder=new_path_builder)

    def _force_source_invariant(self, source: SourceT) -> SourceT:
        raise RuntimeError('This method should never be called, it is just defined for typing purposes.')


@dataclass
class DecisionBranch:
    source: TypeOrTypeExpression[Any]
    matches: Callable[[Any], bool] | None
    path: Path


@dataclass
class DecisionBranchBuilder[StateT, DepsT, OutputT, BranchSourceT, DecisionSourceT]:
    decision: Decision[StateT, DepsT, DecisionSourceT]
    source: TypeOrTypeExpression[BranchSourceT]
    matches: Callable[[Any], bool] | None
    path_builder: PathBuilder[StateT, DepsT, OutputT]

    def to(
        self,
        destination: DestinationNode[StateT, DepsT, OutputT],
        /,
        *extra_destinations: DestinationNode[StateT, DepsT, OutputT],
    ) -> Decision[StateT, DepsT, DecisionSourceT | BranchSourceT]:
        new_branch = DecisionBranch(
            source=self.source, matches=self.matches, path=self.path_builder.to(destination, *extra_destinations)
        )
        return Decision(id=self.decision.id, branches=self.decision.branches + [new_branch], note=self.decision.note)

    def fork(
        self, get_forks: Callable[[Self], Sequence[Decision[StateT, DepsT, DecisionSourceT | BranchSourceT]]], /
    ) -> Decision[StateT, DepsT, DecisionSourceT | BranchSourceT]:
        n_initial_branches = len(self.decision.branches)
        fork_decisions = get_forks(self)
        new_paths = [b.path for fd in fork_decisions for b in fd.branches[n_initial_branches:]]
        new_branch = DecisionBranch(source=self.source, matches=self.matches, path=self.path_builder.fork(new_paths))
        return Decision(id=self.decision.id, branches=self.decision.branches + [new_branch], note=self.decision.note)

    def transform[NewOutputT](
        self, func: TransformFunction[StateT, DepsT, OutputT, NewOutputT], /
    ) -> DecisionBranchBuilder[StateT, DepsT, NewOutputT, BranchSourceT, DecisionSourceT]:
        return DecisionBranchBuilder(
            decision=self.decision,
            source=self.source,
            matches=self.matches,
            path_builder=self.path_builder.transform(func),
        )

    def spread[T](
        self: DecisionBranchBuilder[StateT, DepsT, Sequence[T], BranchSourceT, DecisionSourceT],
    ) -> DecisionBranchBuilder[StateT, DepsT, T, BranchSourceT, DecisionSourceT]:
        return DecisionBranchBuilder(
            decision=self.decision, source=self.source, matches=self.matches, path_builder=self.path_builder.spread()
        )

    def label(self, label: str) -> DecisionBranchBuilder[StateT, DepsT, OutputT, BranchSourceT, DecisionSourceT]:
        return DecisionBranchBuilder(
            decision=self.decision,
            source=self.source,
            matches=self.matches,
            path_builder=self.path_builder.label(label),
        )
