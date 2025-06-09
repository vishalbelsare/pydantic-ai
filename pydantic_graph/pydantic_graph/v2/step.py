from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from typing import Protocol

from pydantic_graph.v2.id_types import NodeId


class StateManager[StateT](ABC):
    @abstractmethod
    async def get_immutable_state(self) -> StateT:
        raise NotImplementedError

    @abstractmethod
    @asynccontextmanager
    async def get_mutable_state(self) -> AsyncIterator[StateT]:
        raise NotImplementedError
        yield


# TODO: Should StepContext be passed to joins/forks/decisions? Like, unified with ReducerContext etc.?
# TODO: Make InputT default to object so it can be dropped when not relevant?
class StepContext[StateT, DepsT, InputT]:
    """The main reason this is not a dataclass is that we need it to be covariant in its type parameters."""

    def __init__(self, state_manager: StateManager[StateT], deps: DepsT, inputs: InputT):
        self._state_manager = state_manager
        self._deps = deps
        self._inputs = inputs

    @asynccontextmanager
    async def get_mutable_state(self) -> AsyncIterator[StateT]:
        async with self._state_manager.get_mutable_state() as state:
            yield state

    async def get_immutable_state(self) -> StateT:
        return await self._state_manager.get_immutable_state()

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
