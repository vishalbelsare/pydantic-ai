from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Self

from pydantic_graph.v2.id_types import ForkId, NodeId
from pydantic_graph.v2.paths import Path, PathBuilder
from pydantic_graph.v2.transform import TransformFunction
from pydantic_graph.v2.util import TypeOrTypeExpression

if TYPE_CHECKING:
    from pydantic_graph.v2.node_types import DestinationNode


@dataclass
class Decision[StateT, DepsT, SourceT]:
    id: NodeId
    branches: list[DecisionBranch[Any]]
    # TODO: Add a field for the label for the input edge
    note: str | None  # TODO: Add a way to set this in the graph.add_decision method(?)

    def branch[S](self, branch: DecisionBranch[S]) -> Decision[StateT, DepsT, SourceT | S]:
        return Decision(id=self.id, branches=self.branches + [branch], note=self.note)

    def _force_source_invariant(self, source: SourceT) -> SourceT:
        raise RuntimeError('This method should never be called, it is just defined for typing purposes.')


@dataclass
class DecisionBranch[SourceT]:
    source: TypeOrTypeExpression[SourceT]
    matches: Callable[[Any], bool] | None
    path: Path


@dataclass
class DecisionBranchBuilder[StateT, DepsT, OutputT, BranchSourceT, DecisionSourceT]:
    decision: Decision[StateT, DepsT, DecisionSourceT]
    source: TypeOrTypeExpression[BranchSourceT]
    matches: Callable[[Any], bool] | None
    path_builder: PathBuilder[StateT, DepsT, OutputT]

    @property
    def last_fork_id(self) -> ForkId | None:
        last_fork = self.path_builder.last_fork
        if last_fork is None:
            return None
        return last_fork.fork_id

    def to(
        self,
        destination: DestinationNode[StateT, DepsT, OutputT],
        /,
        *extra_destinations: DestinationNode[StateT, DepsT, OutputT],
    ) -> DecisionBranch[BranchSourceT]:
        return DecisionBranch(
            source=self.source, matches=self.matches, path=self.path_builder.to(destination, *extra_destinations)
        )

    def fork(
        self, get_forks: Callable[[Self], Sequence[Decision[StateT, DepsT, DecisionSourceT | BranchSourceT]]], /
    ) -> DecisionBranch[BranchSourceT]:
        n_initial_branches = len(self.decision.branches)
        fork_decisions = get_forks(self)
        new_paths = [b.path for fd in fork_decisions for b in fd.branches[n_initial_branches:]]
        return DecisionBranch(source=self.source, matches=self.matches, path=self.path_builder.fork(new_paths))

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
