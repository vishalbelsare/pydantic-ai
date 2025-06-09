from __future__ import annotations

import asyncio
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, assert_never, get_args, get_origin

from typing_extensions import Literal

from pydantic_graph.v2.decision import Decision
from pydantic_graph.v2.execution.graph_task import GraphTask
from pydantic_graph.v2.graph import Graph
from pydantic_graph.v2.id_types import ForkStack, ForkStackItem, JoinId, NodeId, NodeRunId, TaskId
from pydantic_graph.v2.join import Join, Reducer, ReducerContext
from pydantic_graph.v2.node import (
    EndNode,
    Fork,
    StartNode,
)
from pydantic_graph.v2.node_types import AnyNode
from pydantic_graph.v2.paths import BroadcastMarker, DestinationMarker, LabelMarker, Path, SpreadMarker, TransformMarker
from pydantic_graph.v2.step import StateManager, Step, StepContext
from pydantic_graph.v2.util import TypeExpression


@dataclass
class EndMarker:
    value: Any


@dataclass
class JoinItem:
    join_id: JoinId
    inputs: Any
    fork_stack: ForkStack


@dataclass(init=False)
class GraphWalker[StateT, DepsT, InputT, OutputT]:
    graph: Graph[StateT, DepsT, InputT, OutputT]
    state_manager: StateManager[StateT]
    deps: DepsT

    # TODO: Need to figure out how to turn this into something that works as a temporal workflow.
    # Probably need to make it so the state, deps, and inputs are passed to `run`.
    # Then, need to move the `_handle_node` method to a standalone method that can be used as an activity.
    # In the short term, it probably makes the most sense for the graph to be an attribute of both
    # and part of the registered activities etc., but eventually it should be possible for it to be dynamic...
    def __init__(
        self,
        graph: Graph[StateT, DepsT, InputT, OutputT],
        state_manager: StateManager[StateT],
        deps: DepsT,
    ):
        self.graph = graph
        self.state_manager = state_manager
        self.deps = deps

    async def run(self, inputs: InputT) -> OutputT:
        initial_fork_stack: ForkStack = (ForkStackItem(StartNode.id, NodeRunId(str(uuid.uuid4())), 0),)

        start_task = GraphTask(node_id=StartNode.id, inputs=inputs, fork_stack=initial_fork_stack)

        tasks_by_id = {start_task.task_id: start_task}
        pending: set[asyncio.Task[EndMarker | JoinItem | Sequence[GraphTask]]] = {
            asyncio.create_task(self.handle_task(start_task), name=start_task.task_id)
        }

        def _start_task(t: GraphTask) -> None:
            """Helper function to start a new task while doing all necessary tracking."""
            tasks_by_id[t.task_id] = t
            pending.add(asyncio.create_task(self.handle_task(t), name=t.task_id))

        active_reducers: dict[tuple[JoinId, NodeRunId], Reducer[Any, Any, Any]] = {}

        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                result = task.result()
                source_task = tasks_by_id.pop(TaskId(task.get_name()))
                if isinstance(result, EndMarker):
                    for t in pending:
                        t.cancel()
                    return result.value

                if isinstance(result, JoinItem):
                    parent_fork_id = self.graph.get_parent_fork(result.join_id).fork_id
                    fork_run_id = [x.node_run_id for x in result.fork_stack[::-1] if x.fork_id == parent_fork_id][0]
                    reducer = active_reducers.get((result.join_id, fork_run_id))
                    if reducer is None:
                        join_node = self.graph.nodes[result.join_id]
                        assert isinstance(join_node, Join)
                        # TODO: Make it so reducers don't use state or deps, otherwise they have to use activities
                        #  (It might be possible to make it work with deps..)
                        reducer = join_node.create_reducer(ReducerContext(None, result.inputs))
                        active_reducers[(result.join_id, fork_run_id)] = reducer
                    else:
                        # TODO: Make it so reducers don't use state or deps, otherwise they have to use activities
                        #  (It might be possible to make it work with deps..)
                        reducer.reduce(ReducerContext(None, result.inputs))
                else:
                    for new_task in result:
                        _start_task(new_task)

                for join_id, fork_run_id, fork_stack in self._get_completed_fork_runs(
                    source_task, tasks_by_id.values(), active_reducers.keys()
                ):
                    reducer = active_reducers.pop((join_id, fork_run_id))

                    # TODO: Remove state/deps from ReducerContext if possible. (At least state)
                    ctx = ReducerContext(None, None)
                    output = reducer.finalize(ctx)
                    join_node = self.graph.nodes[join_id]
                    assert isinstance(join_node, Join)  # We could drop this but if it fails it means there is a bug.
                    new_tasks = self._handle_edges(join_node, output, fork_stack)
                    for new_task in new_tasks:
                        _start_task(new_task)
        raise RuntimeError(
            'Graph run completed, but no result was produced. This is either a bug in the graph or a bug in the graph runner.'
        )

    async def handle_task(self, task: GraphTask) -> Sequence[GraphTask] | JoinItem | EndMarker:
        return await self._handle_node(task.node_id, task.inputs, task.fork_stack)

    # TODO: I believe the following function is what should be the activity in the temporal graph walker.
    async def _handle_node(
        self, node_id: NodeId, inputs: Any, fork_stack: ForkStack
    ) -> Sequence[GraphTask] | JoinItem | EndMarker:
        node = self.graph.nodes[node_id]
        if isinstance(node, (StartNode, Fork)):
            return self._handle_edges(node, inputs, fork_stack)
        elif isinstance(node, Step):
            step_context = StepContext[StateT, DepsT, Any](self.state_manager, self.deps, inputs)
            output = await node.call(step_context)
            return self._handle_edges(node, output, fork_stack)
        elif isinstance(node, Join):
            return JoinItem(node_id, inputs, fork_stack)
        elif isinstance(node, Decision):
            return self._handle_decision(node, inputs, fork_stack)
        elif isinstance(node, EndNode):
            return EndMarker(inputs)
        else:
            assert_never(node)

    def _handle_decision(
        self, decision: Decision[StateT, DepsT, Any], inputs: Any, fork_stack: ForkStack
    ) -> Sequence[GraphTask]:
        for branch in decision.branches:
            match_tester = branch.matches
            if match_tester is not None:
                inputs_match = match_tester(inputs)
            else:
                branch_source = branch.source
                if get_origin(branch_source) is TypeExpression:
                    branch_source = get_args(branch_source)[0]

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
                return self._handle_path(branch.path, inputs, fork_stack)

        raise RuntimeError(f'No branch matched inputs {inputs} for decision node {decision}.')

    def _handle_path(self, path: Path, inputs: Any, fork_stack: ForkStack) -> Sequence[GraphTask]:
        if not path.items:
            return []

        item = path.items[0]
        if isinstance(item, DestinationMarker):
            return [GraphTask(item.destination_id, inputs, fork_stack)]
        elif isinstance(item, SpreadMarker):
            node_run_id = NodeRunId(str(uuid.uuid4()))
            return [
                GraphTask(
                    item.fork_id, input_item, fork_stack + (ForkStackItem(item.fork_id, node_run_id, thread_index),)
                )
                for thread_index, input_item in enumerate(inputs)
            ]
        elif isinstance(item, BroadcastMarker):
            return [GraphTask(item.fork_id, inputs, fork_stack)]
        elif isinstance(item, TransformMarker):
            # TODO: Transforms need to become (anonymous) nodes so that we can do this as a node.
            raise NotImplementedError
            # state = await self.run_api.get_immutable_state()
            # ctx = TransformContext(state, self.deps, inputs)
            # inputs = item.transform(ctx)
            # return self._handle_path(path.next_path, inputs, fork_stack)
        elif isinstance(item, LabelMarker):
            return self._handle_path(path.next_path, inputs, fork_stack)
        else:
            assert_never(item)

    def _handle_edges(self, node: AnyNode, inputs: Any, fork_stack: ForkStack) -> Sequence[GraphTask]:
        edges = self.graph.edges_by_source.get(node.id, [])
        assert len(edges) == 1 or isinstance(node, Fork)  # this should have already been ensured during graph building

        new_tasks: list[GraphTask] = []
        for path in edges:
            new_tasks.extend(self._handle_path(path, inputs, fork_stack))
        return new_tasks

    def _get_completed_fork_runs(
        self, t: GraphTask, active_tasks: Iterable[GraphTask], reducers_keys: Iterable[tuple[JoinId, NodeRunId]]
    ) -> list[tuple[JoinId, NodeRunId, ForkStack]]:
        completed_fork_runs: list[tuple[JoinId, NodeRunId, ForkStack]] = []

        fork_run_indices = {fsi.node_run_id: i for i, fsi in enumerate(t.fork_stack)}
        for join_id, fork_run_id in reducers_keys:
            fork_run_index = fork_run_indices.get(fork_run_id)
            if fork_run_index is None:
                continue  # The fork_run_id is not in the current task's fork stack, so this task didn't complete it.

            new_fork_stack = t.fork_stack[:fork_run_index]
            # This reducer _may_ now be ready to finalize:
            if self._is_fork_run_completed(active_tasks, join_id, fork_run_id):
                completed_fork_runs.append((join_id, fork_run_id, new_fork_stack))

        return completed_fork_runs

    def _is_fork_run_completed(self, tasks: Iterable[GraphTask], join_id: JoinId, fork_run_id: NodeRunId) -> bool:
        # Check if any of the tasks in the graph have this fork_run_id in their fork_stack
        # If this is the case, then the fork run is not yet completed
        parent_fork = self.graph.get_parent_fork(join_id)
        for t in tasks:
            if fork_run_id in {x.node_run_id for x in t.fork_stack}:
                if t.node_id in parent_fork.intermediate_nodes:
                    return False
        return True
