from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from anyio import Event, Lock, create_memory_object_stream, create_task_group
from anyio.abc import TaskGroup
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

from pydantic_graph.v2.execution.graph_runner import GraphRunAPI, GraphRunner
from pydantic_graph.v2.execution.graph_task import GraphTask
from pydantic_graph.v2.execution.graph_walker import GraphWalker
from pydantic_graph.v2.graph import Graph
from pydantic_graph.v2.id_types import GraphRunId, JoinId, NodeRunId, TaskId
from pydantic_graph.v2.join import Reducer
from pydantic_graph.v2.util import Maybe, Some


@dataclass
class _InMemoryRunState:
    task_group: TaskGroup
    """note: I think this is not currently used in the run, but I think it may be necessary for cancellation"""
    receive_task_stream: MemoryObjectReceiveStream[GraphTask]
    send_task_stream: MemoryObjectSendStream[GraphTask]

    state: Maybe[Any] = None  # might be better to make this non-Maybe and only create instances when you have this

    active_reducers: dict[tuple[JoinId, NodeRunId], Reducer[Any, Any, Any, Any]] = field(default_factory=dict)
    requested_tasks: dict[TaskId, GraphTask] = field(default_factory=dict)
    result: Maybe[Any] = field(default=None)

    finish_event: Event = field(default_factory=Event)
    lock: Lock = field(default_factory=Lock)


@dataclass
class _InMemoryGraphRunAPI[StateT, DepsT](GraphRunAPI[StateT, DepsT]):
    _run_state: _InMemoryRunState

    @property
    def _active_reducers(self):
        return self._run_state.active_reducers

    @property
    def _requested_tasks(self):
        return self._run_state.requested_tasks

    @property
    def _state(self) -> StateT:
        if self._run_state.state is None:
            raise RuntimeError('State has not been initialized for this graph run engine.')
        return self._run_state.state.value

    @_state.setter
    def _state(self, value: StateT) -> None:
        self._run_state.state = Some(value)

    @property
    def _finish_event(self):
        return self._run_state.finish_event

    async def store_requested_task(self, task: GraphTask) -> None:
        self._requested_tasks[task.task_id] = task

    async def complete_requested_task(self, task_id: TaskId) -> None:
        self._requested_tasks.pop(task_id)

    async def get_requested_tasks(self) -> Mapping[TaskId, GraphTask]:
        return self._requested_tasks

    async def any_tasks_remain(self) -> bool:
        return len(self._requested_tasks) > 0

    async def is_fork_run_completed(self, fork_run_id: NodeRunId) -> bool:
        for walk in self._requested_tasks.values():
            # might be a good idea to hold walks_by_fork_id in memory to reduce overhead here
            if fork_run_id in {x[1] for x in walk.fork_stack}:
                return False
        return True

    async def get_active_reducer_state(
        self, join_id: JoinId, fork_run_id: NodeRunId
    ) -> Reducer[StateT, DepsT, Any, Any] | None:
        return self._active_reducers.get((join_id, fork_run_id))

    async def set_active_reducer_state(
        self, join_id: JoinId, fork_run_id: NodeRunId, reducer: Reducer[StateT, DepsT, Any, Any]
    ) -> None:
        # Probably need a way to set reducer state / otherwise rely on it being serializable
        self._active_reducers[(join_id, fork_run_id)] = reducer

    async def complete_active_reducer(self, join_id: JoinId, fork_run_id: NodeRunId) -> None:
        # Probably need a way to set reducer state / otherwise rely on it being serializable
        self._active_reducers.pop((join_id, fork_run_id))

    async def get_active_reducers_with_fork_run_id(
        self, fork_run_ids: Sequence[NodeRunId]
    ) -> list[tuple[tuple[JoinId, NodeRunId], Reducer[StateT, DepsT, Any, Any]]]:
        # This is a helper method to get all reducers for a specific fork run id
        # Note: should return a copy of any dicts etc. to prevent errors due to modifications during iteration
        return [(k, r) for k, r in self._active_reducers.items() if k[1] in fork_run_ids]

    async def request_task(self, task: GraphTask) -> None:
        await self.store_requested_task(task)
        await self._run_state.send_task_stream.send(task)

    async def set_result(self, result: Any) -> None:
        if self._run_state.result is not None:
            raise RuntimeError(f'Result has already been set for this graph run: result={self._run_state.result.value} state={self._state}')
        self._run_state.result = Some(result)

    async def get_result(self) -> Maybe[Any]:
        return self._run_state.result

    async def mark_finished(self) -> None:
        self._finish_event.set()

    async def wait(self) -> Maybe[Any]:
        return await self._finish_event.wait()

    async def initialize_state(self, state: StateT) -> None:
        async with self._state_lock():
            assert self._run_state.state is None, 'State has already been initialized for this graph run engine.'
            self._run_state.state = Some(state)

    async def get_immutable_state(self) -> StateT:
        async with self._state_lock():
            return await self._get_state_unsynchronized()

    @asynccontextmanager
    async def get_mutable_state(self) -> AsyncIterator[StateT]:
        async with self._state_lock():
            state = await self._get_state_unsynchronized()
            yield state
            await self._set_state_unsynchronized(state)

    @asynccontextmanager
    async def _state_lock(self) -> AsyncIterator[None]:
        async with self._run_state.lock:
            yield

    async def _get_state_unsynchronized(self) -> StateT:
        return deepcopy(self._state)

    async def _set_state_unsynchronized(self, state: StateT) -> None:
        self._state = state


@dataclass
class InMemoryGraphRunner[StateT, DepsT, InputT, OutputT](GraphRunner[StateT, DepsT, InputT, OutputT]):
    graph: Graph[StateT, DepsT, InputT, OutputT]

    def __post_init__(self):
        self._runs: dict[GraphRunId, _InMemoryRunState] = {}

    async def run(
        self,
        state: StateT,
        deps: DepsT,
        inputs: InputT,
    ) -> tuple[StateT, OutputT]:
        run_id = GraphRunId(str(uuid.uuid4()))

        try:
            send_task_stream, receive_task_stream = create_memory_object_stream[GraphTask]()

            async with create_task_group() as tg:
                run = _InMemoryRunState(tg, receive_task_stream, send_task_stream)

                async def process_tasks(receive_stream: MemoryObjectReceiveStream[GraphTask]) -> None:
                    async def handle_task(t: GraphTask):
                        engine_ = _InMemoryGraphRunAPI[StateT, DepsT](run)
                        walker = GraphWalker(self.graph, engine_, deps)
                        await walker.handle_task(t)

                    async with receive_stream:
                        async for task in receive_stream:
                            tg.start_soon(handle_task, task)

                tg.start_soon(process_tasks, receive_task_stream)
                async with send_task_stream:
                    engine = _InMemoryGraphRunAPI[StateT, DepsT](run)
                    result = await GraphWalker(self.graph, engine, deps).run(state, inputs)
                    state = await engine.get_immutable_state()
                    tg.cancel_scope.cancel()

            return state, result
        finally:
            self._runs.pop(run_id, None)  # clean up
