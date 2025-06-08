# from __future__ import annotations
#
# from collections import defaultdict
# from collections.abc import AsyncIterator, Sequence
# from contextlib import asynccontextmanager
# from copy import deepcopy
# from dataclasses import dataclass, field
# from typing import Any
#
# from anyio import Event, Lock, create_memory_object_stream, create_task_group
# from anyio.abc import TaskGroup
# from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
# from temporalio import workflow, activity
# from temporalio.client import Client
# from temporalio.worker import Worker
#
# from pydantic_graph.v2.execution.graph_runner import GraphRunAPI, GraphRunner
# from pydantic_graph.v2.execution.graph_task import GraphTask
# from pydantic_graph.v2.execution.graph_walker import GraphWalker, GraphActivityRunner
# from pydantic_graph.v2.graph import Graph
# from pydantic_graph.v2.id_types import JoinId, NodeRunId, TaskId
# from pydantic_graph.v2.join import Reducer, ReducerContext
# from pydantic_graph.v2.util import Maybe, Some
#
#
# @workflow.defn
# @dataclass
# class GraphWorkflow[StateT, DepsT, InputT, OutputT]:
#     graph: Graph[StateT, DepsT, InputT, OutputT]
#     deps: DepsT
#
#     @workflow.run
#     async def run(
#         self,
#         state: StateT,
#         deps: DepsT,
#         inputs: InputT
#     ) -> tuple[StateT, OutputT]:
#         return await TemporalGraphRunner(graph).run(state, deps, inputs)
#
#
# @dataclass
# class _InMemoryRunState:
#     task_group: TaskGroup
#     """note: I think this is not currently used in the run, but I think it may be necessary for cancellation"""
#     receive_task_stream: MemoryObjectReceiveStream[GraphTask]
#     send_task_stream: MemoryObjectSendStream[GraphTask]
#
#     state: Maybe[Any] = None  # might be better to make this non-Maybe and only create instances when you have this
#
#     active_reducers: dict[tuple[JoinId, NodeRunId], Reducer[Any, Any, Any, Any]] = field(default_factory=dict)
#     requested_tasks: dict[TaskId, GraphTask] = field(default_factory=dict)
#     result: Maybe[Any] = field(default=None)
#
#     finish_event: Event = field(default_factory=Event)
#     state_lock: Lock = field(default_factory=Lock)
#     reducer_locks: dict[NodeRunId, Lock] = field(default_factory=lambda: defaultdict(Lock))
#
# @dataclass
# class _InMemoryGraphRunAPI[StateT, DepsT](GraphRunAPI[StateT, DepsT]):
#     # graph: Graph[StateT, DepsT, Any, Any] = field(repr=False)
#     _run_state: _InMemoryRunState
#
#     async def start_task_soon(self, task: GraphTask) -> None:
#         self._requested_tasks[task.task_id] = task
#         await self._run_state.send_task_stream.send(task)
#
#     async def mark_task_completed(self, task_id: TaskId, deps: DepsT) -> list[tuple[JoinId, Any]]:
#         popped_task = self._requested_tasks.pop(task_id)
#         popped_task_fork_stack = popped_task.fork_stack
#
#         # If the popped task was the last item preventing one or more joins, those joins can now be finalized
#         task_fork_run_indices = {fork_run_id: i for i, (_, fork_run_id, _) in enumerate(popped_task_fork_stack)}
#
#         finalized_joins: list[tuple[JoinId, Any]] = []
#
#         # Note: might be more efficient to maintain a better data structure for looking up reducers by join_id and
#         # fork_run_id without iterating through every item. This only matters if there is a large number of reducers.
#         for (join_id, fork_run_id), reducer in await self._get_reducers_with_fork_run_id(list(task_fork_run_indices)):
#             fork_run_index = task_fork_run_indices.get(fork_run_id)
#             assert fork_run_index is not None  # should be filtered by the _get_reducers_with_fork_run_id method
#
#             # This reducer _may_ now be ready to finalize:
#             if await self._is_fork_run_completed(fork_run_id):
#                 self._active_reducers.pop((join_id, fork_run_id))
#                 state = await self.get_immutable_state()
#                 ctx = ReducerContext(state, deps, None)
#                 output = reducer.finalize(ctx)
#                 finalized_joins.append((join_id, output))
#         return finalized_joins
#
#     async def any_tasks_remain(self) -> bool:
#         return len(self._requested_tasks) > 0
#
#     async def initialize_state(self, state: StateT) -> None:
#         async with self._state_lock():
#             assert self._run_state.state is None, 'State has already been initialized for this graph run engine.'
#             self._run_state.state = Some(state)
#
#     async def get_immutable_state(self) -> StateT:
#         async with self._state_lock():
#             return await self._get_state_unsynchronized()
#
#     @asynccontextmanager
#     async def get_mutable_state(self) -> AsyncIterator[StateT]:
#         async with self._state_lock():
#             state = await self._get_state_unsynchronized()
#             yield state
#             await self._set_state_unsynchronized(state)
#
#     @asynccontextmanager
#     async def get_reducer(
#         self, join_id: JoinId, fork_run_id: NodeRunId
#     ) -> AsyncIterator[Reducer[StateT, DepsT, Any, Any] | None]:
#         async with self._reducer_lock(fork_run_id):
#             yield self._active_reducers.get((join_id, fork_run_id))
#
#     async def set_reducer(
#         self, join_id: JoinId, fork_run_id: NodeRunId, fork_thread_index: int, reducer: Reducer[StateT, DepsT, Any, Any]
#     ) -> None:
#         self._active_reducers[(join_id, fork_run_id)] = reducer
#
#     async def set_run_result(self, result: Any) -> None:
#         if self._run_state.result is not None:
#             raise RuntimeError(
#                 f'Result has already been set for this graph run: result={self._run_state.result.value} state={self._state}'
#             )
#         self._run_state.result = Some(result)
#
#     async def mark_run_finished(self) -> None:
#         self._finish_event.set()
#
#     async def wait(self) -> Maybe[Any]:
#         await self._finish_event.wait()
#         return self._run_state.result
#
#     # Internal helpers
#     @property
#     def _active_reducers(self):
#         return self._run_state.active_reducers
#
#     @property
#     def _requested_tasks(self):
#         return self._run_state.requested_tasks
#
#     @property
#     def _state(self) -> StateT:
#         if self._run_state.state is None:
#             raise RuntimeError('State has not been initialized for this graph run engine.')
#         return self._run_state.state.value
#
#     @_state.setter
#     def _state(self, value: StateT) -> None:
#         self._run_state.state = Some(value)
#
#     @property
#     def _finish_event(self):
#         return self._run_state.finish_event
#
#     @asynccontextmanager
#     async def _state_lock(self) -> AsyncIterator[None]:
#         async with self._run_state.state_lock:
#             yield
#
#     @asynccontextmanager
#     async def _reducer_lock(self, fork_run_id: NodeRunId) -> AsyncIterator[None]:
#         async with self._run_state.reducer_locks[fork_run_id]:
#             yield
#
#     async def _get_state_unsynchronized(self) -> StateT:
#         return deepcopy(self._state)
#
#     async def _set_state_unsynchronized(self, state: StateT) -> None:
#         self._state = state
#
#     async def _get_reducers_with_fork_run_id(
#         self, fork_run_ids: Sequence[NodeRunId]
#     ) -> list[tuple[tuple[JoinId, NodeRunId], Reducer[StateT, DepsT, Any, Any]]]:
#         # This is a helper method to get all reducers for a specific fork run id
#         # Note: should return a copy of any dicts etc. to prevent errors due to modifications during iteration
#         return [(k, r) for k, r in self._active_reducers.items() if k[1] in fork_run_ids]
#
#     async def _is_fork_run_completed(self, fork_run_id: NodeRunId) -> bool:
#         for walk in self._requested_tasks.values():
#             # might be a good idea to hold walks_by_fork_id in memory to reduce overhead here
#             if fork_run_id in {x[1] for x in walk.fork_stack}:
#                 return False
#         return True
#
# # @activity.defn
# def run_task(graph: Graph[Any, Any, Any, Any], task: GraphTask):
#     pass
#
#
# @dataclass
# class TemporalGraphRunner[StateT, DepsT, InputT, OutputT](GraphRunner[StateT, DepsT, InputT, OutputT]):
#     graph: Graph[StateT, DepsT, InputT, OutputT]
#
#     async def run(
#         self,
#         state: StateT,
#         deps: DepsT,
#         inputs: InputT,
#     ) -> tuple[StateT, OutputT]:
#         async with self.worker(deps) as run_state:
#             run_api = _InMemoryGraphRunAPI[StateT, DepsT](run_state)
#             result = await GraphWalker(self.graph, run_api, deps).run(state, inputs)
#             state = await run_api.get_immutable_state()
#         return state, result
#
#     @asynccontextmanager
#     async def worker(self):
#         # Start client
#         client = await Client.connect("localhost:7233")
#
#         # Run a worker for the workflow
#         async with Worker(
#             client,
#             task_queue="graph-queue",
#             workflows=[GraphWorkflow],
#             activities=[order_apples, order_bananas, order_cherries, order_oranges],
#         ):
#     @asynccontextmanager
#     async def worker(self, deps: DepsT) -> AsyncIterator[_InMemoryRunState]:
#         send_task_stream, receive_task_stream = create_memory_object_stream[GraphTask]()
#
#         async with create_task_group() as tg:
#             run = _InMemoryRunState(tg, receive_task_stream, send_task_stream)
#
#             async def process_tasks(receive_stream: MemoryObjectReceiveStream[GraphTask]) -> None:
#                 async def handle_task(t: GraphTask):
#                     engine_ = _InMemoryGraphRunAPI[StateT, DepsT](run)
#                     walker = GraphWalker(self.graph, engine_, deps)
#                     await GraphActivityRunner(self.graph, engine_, deps).handle_task(t)
#
#                 async with receive_stream:
#                     async for task in receive_stream:
#                         tg.start_soon(handle_task, task)
#
#             tg.start_soon(process_tasks, receive_task_stream)
#             async with send_task_stream:
#                 yield run
#                 tg.cancel_scope.cancel()
#
