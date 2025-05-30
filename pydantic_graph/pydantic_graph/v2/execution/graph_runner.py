from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any

from pydantic_graph.v2.execution.graph_task import GraphTask
from pydantic_graph.v2.graph import Graph
from pydantic_graph.v2.id_types import JoinId, NodeRunId, TaskId
from pydantic_graph.v2.join import Reducer
from pydantic_graph.v2.util import Maybe


class GraphRunAPI[StateT, DepsT](ABC):
    async def get_immutable_state(self) -> StateT:
        async with self._state_lock():
            return await self._get_state_unsynchronized()

    @asynccontextmanager
    async def get_mutable_state(self) -> AsyncIterator[StateT]:
        async with self._state_lock():
            state = await self._get_state_unsynchronized()
            yield state
            await self._set_state_unsynchronized(state)

    @abstractmethod
    async def store_requested_task(self, task: GraphTask) -> None:
        raise NotImplementedError

    @abstractmethod
    async def complete_requested_task(self, task_id: TaskId) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get_requested_tasks(self) -> Mapping[TaskId, GraphTask]:
        raise NotImplementedError

    @abstractmethod
    async def any_tasks_remain(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def is_fork_run_completed(self, fork_run_id: NodeRunId) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def get_active_reducer_state(
        self, join_id: JoinId, fork_run_id: NodeRunId
    ) -> Reducer[StateT, DepsT, Any, Any] | None:
        raise NotImplementedError

    @abstractmethod
    async def set_active_reducer_state(
        self, join_id: JoinId, fork_run_id: NodeRunId, reducer: Reducer[StateT, DepsT, Any, Any]
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def complete_active_reducer(self, join_id: JoinId, fork_run_id: NodeRunId) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get_active_reducers_with_fork_run_id(
        self, fork_run_ids: Sequence[NodeRunId]
    ) -> list[tuple[tuple[JoinId, NodeRunId], Reducer[StateT, DepsT, Any, Any]]]:
        raise NotImplementedError

    @abstractmethod
    async def request_task(self, task: GraphTask) -> None:
        raise NotImplementedError

    @abstractmethod
    async def set_result(self, result: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get_result(self) -> Maybe[Any]:
        raise NotImplementedError

    @abstractmethod
    async def mark_finished(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def wait(self) -> Maybe[Any]:
        raise NotImplementedError

    @abstractmethod
    async def initialize_state(self, state: StateT) -> None:
        raise NotImplementedError

    @abstractmethod
    @asynccontextmanager
    async def _state_lock(self) -> AsyncIterator[None]:
        raise NotImplementedError
        yield

    @abstractmethod
    async def _get_state_unsynchronized(self) -> StateT:
        raise NotImplementedError

    @abstractmethod
    async def _set_state_unsynchronized(self, state: StateT) -> None:
        raise NotImplementedError


class GraphRunner[StateT, DepsT, InputT, OutputT](ABC):
    graph: Graph[StateT, DepsT, InputT, OutputT]

    async def run(
        self,
        state: StateT,
        deps: DepsT,
        inputs: InputT,
    ) -> tuple[StateT, OutputT]:
        raise NotImplementedError
