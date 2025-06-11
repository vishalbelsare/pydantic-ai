from __future__ import annotations

from collections.abc import Awaitable
from typing import Protocol

from pydantic_graph.v2.id_types import NodeId


# TODO(P3): Make InputT default to object so it can be dropped when not relevant?
class StepContext[StateT, DepsT, InputT]:
    """The main reason this is not a dataclass is that we need it to be covariant in its type parameters."""

    def __init__(self, state: StateT, deps: DepsT, inputs: InputT):
        self._state = state
        self._deps = deps
        self._inputs = inputs

    @property
    def state(self) -> StateT:
        return self._state

    @property
    def deps(self) -> DepsT:
        return self._deps

    @property
    def inputs(self) -> InputT:
        return self._inputs

    def __repr__(self):
        return f'{self.__class__.__name__}(deps={self.deps}, inputs={self.inputs})'


class StepCallProtocol[StateT, DepsT, InputT, OutputT](Protocol):
    """The purpose of this is to make it possible to deserialize step calls similar to how Evaluators work."""

    def __call__(self, ctx: StepContext[StateT, DepsT, InputT]) -> Awaitable[OutputT]:
        raise NotImplementedError


class Step[StateT, DepsT, InputT, OutputT]:
    """The main reason this is not a dataclass is that we need appropriate variance in the type parameters."""

    def __init__(
        self, id: NodeId, call: StepCallProtocol[StateT, DepsT, InputT, OutputT], user_label: str | None = None
    ):
        self.id = id
        self._call = call
        self.user_label = user_label

    @property
    def call(self) -> StepCallProtocol[StateT, DepsT, InputT, OutputT]:
        # The use of a property here is necessary to ensure that Step is covariant/contravariant as appropriate.
        return self._call

    @property
    def label(self) -> str | None:
        return self.user_label
