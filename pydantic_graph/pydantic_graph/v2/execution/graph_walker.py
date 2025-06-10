from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Iterable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, assert_never, get_args, get_origin

from typing_extensions import Literal

from pydantic_graph.v2.decision import Decision
from pydantic_graph.v2.execution.graph_task import GraphTask
from pydantic_graph.v2.graph import Graph
from pydantic_graph.v2.id_types import ForkStack, ForkStackItem, GraphRunId, JoinId, NodeId, NodeRunId, TaskId
from pydantic_graph.v2.join import Join, Reducer, ReducerContext
from pydantic_graph.v2.node import (
    EndNode,
    Fork,
    StartNode,
)
from pydantic_graph.v2.node_types import AnyNode
from pydantic_graph.v2.paths import BroadcastMarker, DestinationMarker, LabelMarker, Path, SpreadMarker, TransformMarker
from pydantic_graph.v2.state import StateManager, StateManagerFactory
from pydantic_graph.v2.step import Step, StepContext
from pydantic_graph.v2.util import unpack_type_expression


@dataclass
class EndMarker:
    value: Any


@dataclass
class JoinItem:
    join_id: JoinId
    inputs: Any
    fork_stack: ForkStack


# TODO(P2/3): We need to implement node-specific deserialization using type adapters so that when the temporal activity
#  is called and gets "less-structured" data, since it won't know the relevant type, we can deserialize it as expected
#  by the node.
@dataclass
class GraphActivityInputs[StateT, DepsT]:
    node_id: NodeId
    inputs: Any
    fork_stack: ForkStack
    state_manager: StateManager[StateT]
    deps: DepsT


class GraphWalker[StateT, DepsT, InputT, OutputT]:
    def __init__(
        self,
        graph: Graph[StateT, DepsT, InputT, OutputT],
        state_manager_factory: StateManagerFactory,
        handle_graph_node: Callable[
            [GraphActivityInputs[StateT, DepsT]], Awaitable[Sequence[GraphTask] | JoinItem | EndMarker]
        ]
        | None = None,
    ):
        self.graph = graph
        self.state_manager_factory = state_manager_factory
        self.handle_graph_node = handle_graph_node or partial(handle_node, graph=graph)

    # TODO: Need to implement some form of "iteration", and ensure that some pattern for stream handling works.
    #  I think it will be fine with channels in deps in the in-memory case; in the temporal case, you'll just need to
    #  pass the relevant channel-building configuration (e.g., for a redis streams connection or similar) through
    #  (serializable) deps.
    async def run(self, state: StateT, deps: DepsT, inputs: InputT) -> tuple[StateManager[StateT], OutputT]:
        run_id = GraphRunId(str(uuid.uuid4()))
        initial_fork_stack: ForkStack = (ForkStackItem(StartNode.id, NodeRunId(run_id), 0),)
        state_manager = self.state_manager_factory.get_instance(self.graph.state_type, run_id, state)

        start_task = GraphTask(node_id=StartNode.id, inputs=inputs, fork_stack=initial_fork_stack)

        tasks_by_id = {start_task.task_id: start_task}
        pending: set[asyncio.Task[EndMarker | JoinItem | Sequence[GraphTask]]] = {
            asyncio.create_task(self.handle_task(start_task, state_manager, deps), name=start_task.task_id)
        }

        def _start_task(t: GraphTask) -> None:
            """Helper function to start a new task while doing all necessary tracking."""
            tasks_by_id[t.task_id] = t
            pending.add(asyncio.create_task(self.handle_task(t, state_manager, deps), name=t.task_id))

        active_reducers: dict[tuple[JoinId, NodeRunId], Reducer[Any, Any, Any]] = {}

        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                result = task.result()
                source_task = tasks_by_id.pop(TaskId(task.get_name()))
                if isinstance(result, EndMarker):
                    for t in pending:
                        t.cancel()
                    return state_manager, result.value

                if isinstance(result, JoinItem):
                    parent_fork_id = self.graph.get_parent_fork(result.join_id).fork_id
                    fork_run_id = [x.node_run_id for x in result.fork_stack[::-1] if x.fork_id == parent_fork_id][0]
                    reducer = active_reducers.get((result.join_id, fork_run_id))
                    if reducer is None:
                        join_node = self.graph.nodes[result.join_id]
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
                    self.graph, source_task, tasks_by_id.values(), active_reducers.keys()
                ):
                    reducer = active_reducers.pop((join_id, fork_run_id))

                    ctx = ReducerContext(None, None)
                    output = reducer.finalize(ctx)
                    join_node = self.graph.nodes[join_id]
                    assert isinstance(join_node, Join)  # We could drop this but if it fails it means there is a bug.
                    new_tasks = _handle_edges(self.graph, join_node, output, fork_stack)
                    for new_task in new_tasks:
                        _start_task(new_task)
        raise RuntimeError(
            'Graph run completed, but no result was produced. This is either a bug in the graph or a bug in the graph runner.'
        )

    async def handle_task(
        self, task: GraphTask, state_manager: StateManager[StateT], deps: DepsT
    ) -> Sequence[GraphTask] | JoinItem | EndMarker:
        inputs = GraphActivityInputs(task.node_id, task.inputs, task.fork_stack, state_manager, deps)
        return await self.handle_graph_node(inputs)


async def handle_node[StateT, DepsT](
    activity_inputs: GraphActivityInputs[StateT, DepsT], graph: Graph[StateT, DepsT, Any, Any]
) -> Sequence[GraphTask] | JoinItem | EndMarker:
    node_id = activity_inputs.node_id
    inputs = activity_inputs.inputs
    fork_stack = activity_inputs.fork_stack
    state_manager = activity_inputs.state_manager
    deps = activity_inputs.deps

    node = graph.nodes[node_id]
    if isinstance(node, (StartNode, Fork)):
        return _handle_edges(graph, node, inputs, fork_stack)
    elif isinstance(node, Step):
        step_context = StepContext[StateT, DepsT, Any](state_manager, deps, inputs)
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
