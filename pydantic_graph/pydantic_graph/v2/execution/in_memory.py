from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass, field

from anyio import Lock

from pydantic_graph.v2.execution.graph_runner import GraphRunner
from pydantic_graph.v2.execution.graph_walker import GraphWalker
from pydantic_graph.v2.graph import Graph
from pydantic_graph.v2.id_types import GraphRunId
from pydantic_graph.v2.state import StateManager, StateManagerFactory


@dataclass
class _InMemoryStateManager[StateT](StateManager[StateT]):
    _state: StateT
    _state_lock: Lock = field(default_factory=Lock)

    async def get_immutable_state(self) -> StateT:
        async with self._state_lock:
            return await self._get_state_unsynchronized()

    @asynccontextmanager
    async def get_mutable_state(self) -> AsyncIterator[StateT]:
        async with self._state_lock:
            state = await self._get_state_unsynchronized()
            yield state
            await self._set_state_unsynchronized(state)

    async def _get_state_unsynchronized(self) -> StateT:
        return deepcopy(self._state)

    async def _set_state_unsynchronized(self, state: StateT) -> None:
        self._state = state


class InMemoryStateManagerFactory(StateManagerFactory):
    def get_instance[StateT](
        self, state_type: type[StateT], run_id: GraphRunId, initial_value: StateT
    ) -> StateManager[StateT]:
        return _InMemoryStateManager[StateT](initial_value)


@dataclass
class InMemoryGraphRunner[StateT, DepsT, InputT, OutputT](GraphRunner[StateT, DepsT, InputT, OutputT]):
    graph: Graph[StateT, DepsT, InputT, OutputT]
    state_manager_factory: InMemoryStateManagerFactory

    async def run(
        self,
        state: StateT,
        deps: DepsT,
        inputs: InputT,
    ) -> tuple[StateT, OutputT]:
        graph_walker = GraphWalker(self.graph, self.state_manager_factory)
        state_manager, result = await graph_walker.run(state, deps, inputs)
        state = await state_manager.get_immutable_state()
        return state, result
