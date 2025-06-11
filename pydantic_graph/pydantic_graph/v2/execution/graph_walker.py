from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Iterable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, assert_never, cast, get_args, get_origin

from typing_extensions import Literal

from pydantic_graph.v2.decision import Decision
from pydantic_graph.v2.execution.graph_task import GraphTask
from pydantic_graph.v2.graph import Graph
from pydantic_graph.v2.id_types import ForkStack, ForkStackItem, GraphRunId, JoinId, NodeRunId, TaskId
from pydantic_graph.v2.join import Join, Reducer, ReducerContext
from pydantic_graph.v2.node import (
    EndNode,
    Fork,
    StartNode,
)
from pydantic_graph.v2.node_types import AnyNode
from pydantic_graph.v2.paths import BroadcastMarker, DestinationMarker, LabelMarker, Path, SpreadMarker, TransformMarker
from pydantic_graph.v2.step import Step, StepContext
from pydantic_graph.v2.util import unpack_type_expression


@dataclass
class EndMarker[OutputT]:
    value: OutputT


@dataclass
class JoinItem:
    join_id: JoinId
    inputs: Any
    fork_stack: ForkStack


class GraphRunner[StateT, DepsT, InputT, OutputT]:
    def __init__(self, graph: Graph[StateT, DepsT, InputT, OutputT]):
        self.graph = graph

    async def run(self, state: StateT, deps: DepsT, inputs: InputT) -> tuple[StateT, OutputT]:
        async with self.iter(state, deps, inputs) as graph_run:
            # TODO: This would probably be better using `async for _ in graph_run`, but this tests the `next` method,
            #  which I'm less confident will be implemented correctly if not used on the critical path. We can change it
            #  once we have tests, etc.
            event: Any = None
            while True:
                try:
                    event = await graph_run.next(event)
                except StopAsyncIteration:
                    assert isinstance(event, EndMarker), 'Graph run should end with an EndMarker.'
                    return state, cast(EndMarker[OutputT], event).value

    @asynccontextmanager
    async def iter(self, state: StateT, deps: DepsT, inputs: InputT) -> AsyncIterator[GraphRun[StateT, DepsT, OutputT]]:
        yield GraphRun[StateT, DepsT, OutputT](
            graph=self.graph,
            initial_state=state,
            deps=deps,
            inputs=inputs,
        )


class GraphRun[StateT, DepsT, OutputT]:
    def __init__(
        self,
        graph: Graph[StateT, DepsT, Any, OutputT],
        initial_state: StateT,
        deps: DepsT,
        inputs: Any,
    ):
        self._next: EndMarker[OutputT] | JoinItem | Sequence[GraphTask] | None = None
        self._iterator = _iter(graph, initial_state, deps, inputs)

    def __aiter__(self) -> AsyncIterator[EndMarker[OutputT] | JoinItem | Sequence[GraphTask]]:
        return self

    async def __anext__(self) -> EndMarker[OutputT] | JoinItem | Sequence[GraphTask]:
        if self._next is None:
            self._next = await anext(self._iterator)
        else:
            self._next = await self._iterator.asend(self._next)
        return self._next

    async def next(
        self, value: EndMarker[OutputT] | JoinItem | Sequence[GraphTask] | None = None
    ) -> EndMarker[OutputT] | JoinItem | Sequence[GraphTask]:
        """Allows for sending a value to the iterator, which is useful for resuming the iteration."""
        if value is not None:
            self._next = value
        return await self.__anext__()


# TODO: Need to implement some pattern for stream handling that works.
#  I think it will be fine with channels in deps in the in-memory case; in the temporal case, you'll just need to
#  pass the relevant channel-building configuration (e.g., for a redis streams connection or similar) through
#  (serializable) deps.
async def _iter[StateT, DepsT, InputsT, OutputT](
    graph: Graph[StateT, DepsT, InputsT, OutputT], state: StateT, deps: DepsT, inputs: InputsT
) -> AsyncGenerator[
    EndMarker[OutputT] | JoinItem | Sequence[GraphTask], EndMarker[OutputT] | JoinItem | Sequence[GraphTask]
]:
    run_id = GraphRunId(str(uuid.uuid4()))
    initial_fork_stack: ForkStack = (ForkStackItem(StartNode.id, NodeRunId(run_id), 0),)

    start_task = GraphTask(node_id=StartNode.id, inputs=inputs, fork_stack=initial_fork_stack)

    tasks_by_id = {start_task.task_id: start_task}
    pending: set[asyncio.Task[EndMarker[OutputT] | JoinItem | Sequence[GraphTask]]] = {
        asyncio.create_task(_handle_task(graph, state, deps, start_task), name=start_task.task_id)
    }

    def _start_task(t: GraphTask) -> None:
        """Helper function to start a new task while doing all necessary tracking."""
        tasks_by_id[t.task_id] = t
        pending.add(asyncio.create_task(_handle_task(graph, state, deps, t), name=t.task_id))

    active_reducers: dict[tuple[JoinId, NodeRunId], Reducer[Any, Any, Any]] = {}

    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            result = task.result()
            source_task = tasks_by_id.pop(TaskId(task.get_name()))
            result = yield result
            if isinstance(result, EndMarker):
                for t in pending:
                    t.cancel()
                # yield result
                return

            # yield result
            if isinstance(result, JoinItem):
                parent_fork_id = graph.get_parent_fork(result.join_id).fork_id
                fork_run_id = [x.node_run_id for x in result.fork_stack[::-1] if x.fork_id == parent_fork_id][0]
                reducer = active_reducers.get((result.join_id, fork_run_id))
                if reducer is None:
                    join_node = graph.nodes[result.join_id]
                    assert isinstance(join_node, Join)
                    # Note: if we wanted to access `state` in the ReducerContext, we'd need reducing to be an activity in temporal
                    # That might be reasonable, but things seem simpler like this, and it seems maybe unnecessary
                    reducer = join_node.create_reducer(ReducerContext(None, result.inputs))
                    active_reducers[(result.join_id, fork_run_id)] = reducer
                else:
                    # Note: if we wanted to access `state` in the ReducerContext, we'd need reducing to be an activity in temporal
                    # That might be reasonable, but things seem simpler like this, and it seems maybe unnecessary
                    reducer.reduce(ReducerContext(None, result.inputs))
            else:
                for new_task in result:
                    _start_task(new_task)

            for join_id, fork_run_id, fork_stack in _get_completed_fork_runs(
                graph, source_task, tasks_by_id.values(), active_reducers.keys()
            ):
                reducer = active_reducers.pop((join_id, fork_run_id))

                ctx = ReducerContext(None, None)
                output = reducer.finalize(ctx)
                join_node = graph.nodes[join_id]
                assert isinstance(join_node, Join)  # We could drop this but if it fails it means there is a bug.
                new_tasks = _handle_edges(graph, join_node, output, fork_stack)
                for new_task in new_tasks:
                    _start_task(new_task)

    raise RuntimeError(
        'Graph run completed, but no result was produced. This is either a bug in the graph or a bug in the graph runner.'
    )


async def _handle_task[StateT, DepsT, OutputT](
    graph: Graph[StateT, DepsT, Any, OutputT],
    state: StateT,
    deps: DepsT,
    task: GraphTask,
) -> Sequence[GraphTask] | JoinItem | EndMarker[OutputT]:
    node_id = task.node_id
    inputs = task.inputs
    fork_stack = task.fork_stack

    node = graph.nodes[node_id]
    if isinstance(node, (StartNode, Fork)):
        return _handle_edges(graph, node, inputs, fork_stack)
    elif isinstance(node, Step):
        step_context = StepContext[StateT, DepsT, Any](state, deps, inputs)
        output = await node.call(step_context)
        return _handle_edges(graph, node, output, fork_stack)
    elif isinstance(node, Join):
        return JoinItem(node_id, inputs, fork_stack)
    elif isinstance(node, Decision):
        return _handle_decision(graph, node, inputs, fork_stack)
    elif isinstance(node, EndNode):
        return EndMarker(inputs)
    else:
        assert_never(node)


def _handle_decision[StateT, DepsT](
    graph: Graph[StateT, DepsT, Any, Any], decision: Decision[StateT, DepsT, Any], inputs: Any, fork_stack: ForkStack
) -> Sequence[GraphTask]:
    for branch in decision.branches:
        match_tester = branch.matches
        if match_tester is not None:
            inputs_match = match_tester(inputs)
        else:
            branch_source = unpack_type_expression(branch.source)

            if branch_source in {Any, object}:
                inputs_match = True
            elif get_origin(branch_source) is Literal:
                inputs_match = inputs in get_args(branch_source)
            else:
                try:
                    inputs_match = isinstance(inputs, branch_source)
                except TypeError as e:
                    raise RuntimeError(f'Decision branch source {branch_source} is not a valid type.') from e

        if inputs_match:
            return _handle_path(graph, branch.path, inputs, fork_stack)

    raise RuntimeError(f'No branch matched inputs {inputs} for decision node {decision}.')


def _get_completed_fork_runs(
    graph: Graph[Any, Any, Any, Any],
    t: GraphTask,
    active_tasks: Iterable[GraphTask],
    reducers_keys: Iterable[tuple[JoinId, NodeRunId]],
) -> list[tuple[JoinId, NodeRunId, ForkStack]]:
    completed_fork_runs: list[tuple[JoinId, NodeRunId, ForkStack]] = []

    fork_run_indices = {fsi.node_run_id: i for i, fsi in enumerate(t.fork_stack)}
    for join_id, fork_run_id in reducers_keys:
        fork_run_index = fork_run_indices.get(fork_run_id)
        if fork_run_index is None:
            continue  # The fork_run_id is not in the current task's fork stack, so this task didn't complete it.

        new_fork_stack = t.fork_stack[:fork_run_index]
        # This reducer _may_ now be ready to finalize:
        if _is_fork_run_completed(graph, active_tasks, join_id, fork_run_id):
            completed_fork_runs.append((join_id, fork_run_id, new_fork_stack))

    return completed_fork_runs


def _handle_path(
    graph: Graph[Any, Any, Any, Any], path: Path, inputs: Any, fork_stack: ForkStack
) -> Sequence[GraphTask]:
    if not path.items:
        return []

    item = path.items[0]
    if isinstance(item, DestinationMarker):
        return [GraphTask(item.destination_id, inputs, fork_stack)]
    elif isinstance(item, SpreadMarker):
        node_run_id = NodeRunId(str(uuid.uuid4()))
        return [
            GraphTask(item.fork_id, input_item, fork_stack + (ForkStackItem(item.fork_id, node_run_id, thread_index),))
            for thread_index, input_item in enumerate(inputs)
        ]
    elif isinstance(item, BroadcastMarker):
        return [GraphTask(item.fork_id, inputs, fork_stack)]
    elif isinstance(item, TransformMarker):
        # TODO(P1): Transforms need to become (anonymous) nodes so that we can do this as a node.
        raise NotImplementedError
        # state = await self.run_api.get_immutable_state()
        # ctx = TransformContext(state, self.deps, inputs)
        # inputs = item.transform(ctx)
        # return self._handle_path(path.next_path, inputs, fork_stack)
    elif isinstance(item, LabelMarker):
        return _handle_path(graph, path.next_path, inputs, fork_stack)
    else:
        assert_never(item)


def _handle_edges(
    graph: Graph[Any, Any, Any, Any], node: AnyNode, inputs: Any, fork_stack: ForkStack
) -> Sequence[GraphTask]:
    edges = graph.edges_by_source.get(node.id, [])
    assert len(edges) == 1 or isinstance(node, Fork)  # this should have already been ensured during graph building

    new_tasks: list[GraphTask] = []
    for path in edges:
        new_tasks.extend(_handle_path(graph, path, inputs, fork_stack))
    return new_tasks


def _is_fork_run_completed(
    graph: Graph[Any, Any, Any, Any], tasks: Iterable[GraphTask], join_id: JoinId, fork_run_id: NodeRunId
) -> bool:
    # Check if any of the tasks in the graph have this fork_run_id in their fork_stack
    # If this is the case, then the fork run is not yet completed
    parent_fork = graph.get_parent_fork(join_id)
    for t in tasks:
        if fork_run_id in {x.node_run_id for x in t.fork_stack}:
            if t.node_id in parent_fork.intermediate_nodes:
                return False
    return True
