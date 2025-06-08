from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any, get_args, get_origin, Sequence, assert_never, Iterable

from typing_extensions import Literal

from pydantic_graph.v2.decision import Decision
from pydantic_graph.v2.execution.graph_runner import GraphRunAPI
from pydantic_graph.v2.execution.graph_task import GraphTask
from pydantic_graph.v2.graph import Graph
from pydantic_graph.v2.id_types import ForkStack, NodeRunId, ThreadIndex, NodeId, TaskId, JoinId, ForkStackItem
from pydantic_graph.v2.join import Join, ReducerContext, Reducer
from pydantic_graph.v2.node import (
    EndNode,
    Fork,
    StartNode,
)
from pydantic_graph.v2.node_types import AnyNode
from pydantic_graph.v2.paths import BroadcastMarker, DestinationMarker, LabelMarker, Path, SpreadMarker, TransformMarker
from pydantic_graph.v2.step import Step, StepContext
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
    run_api: GraphRunAPI[StateT, DepsT]
    deps: DepsT

    def __init__(
        self,
        graph: Graph[StateT, DepsT, InputT, OutputT],
        api: GraphRunAPI[StateT, DepsT],
        deps: DepsT,
    ):
        self.graph = graph
        self.run_api = api
        self.deps = deps

    async def run(self, state: StateT, inputs: InputT) -> OutputT:
        await self.run_api.initialize_state(state)

        initial_fork_stack: ForkStack = (ForkStackItem(StartNode.id, NodeRunId(str(uuid.uuid4())), ThreadIndex(0)),)

        start_task = GraphTask(node_id=StartNode.id, inputs=inputs, fork_stack=initial_fork_stack)

        tasks_by_id = {start_task.task_id: start_task}
        pending: set[asyncio.Task[EndMarker | JoinItem | Sequence[GraphTask]]] = {
            asyncio.create_task(self.handle_task(start_task), name=start_task.task_id)
        }
        active_reducers: dict[tuple[JoinId, NodeRunId], Reducer[Any, Any, Any, Any]] = {}

        def _finalize_joins(t: GraphTask) -> list[tuple[JoinId, Any, ForkStack]]:
            finalized_joins: list[tuple[JoinId, Any, ForkStack]] = []
            task_fork_run_indices = {fsi.node_run_id: i for i, fsi in enumerate(t.fork_stack)}
            matching_reducers = [(k, r) for k, r in active_reducers.items() if k[1] in task_fork_run_indices]
            for (join_id, fork_run_id), reducer in matching_reducers:
                fork_run_index = task_fork_run_indices.get(fork_run_id)
                assert fork_run_index is not None  # should be filtered by the _get_reducers_with_fork_run_id method
                new_fork_stack = t.fork_stack[:fork_run_index]

                # This reducer _may_ now be ready to finalize:
                if _is_fork_run_completed(tasks_by_id.values(), fork_run_id):
                    active_reducers.pop((join_id, fork_run_id))
                    # TODO: Remove state/deps from ReducerContext if possible. (At least state)
                    ctx = ReducerContext(None, None, None)
                    output = reducer.finalize(ctx)
                    finalized_joins.append((join_id, output, new_fork_stack))
            return finalized_joins

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
                        reducer = join_node.create_reducer(ReducerContext(None, None, result.inputs))
                        active_reducers[(result.join_id, fork_run_id)] = reducer
                    else:
                        # TODO: Make it so reducers don't use state or deps, otherwise they have to use activities
                        #  (It might be possible to make it work with deps..)
                        reducer.reduce(ReducerContext(None, None, result.inputs))
                else:
                    for new_task in result:
                        tasks_by_id[new_task.task_id] = new_task
                        pending.add(asyncio.create_task(self.handle_task(new_task), name=new_task.task_id))

                finalized_joins = _finalize_joins(source_task)
                for join_id, output, fork_stack in finalized_joins:
                    join_node = self.graph.nodes[join_id]
                    assert isinstance(join_node, Join)
                    new_tasks = self._handle_edges(join_node, output, fork_stack)
                    for new_task in new_tasks:
                        tasks_by_id[new_task.task_id] = new_task
                        pending.add(asyncio.create_task(self.handle_task(new_task), name=new_task.task_id))
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
            step_context = StepContext[StateT, DepsT, Any](self.run_api, self.deps, inputs)
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
            # TODO: DestinationMarker should have ids not actual destination nodes
            return [GraphTask(item.destination.id, inputs, fork_stack)]
        elif isinstance(item, SpreadMarker):
            node_run_id = NodeRunId(str(uuid.uuid4()))
            tasks: list[GraphTask] = []
            for i, input_item in enumerate(inputs):
                # TODO: Can probably remove the ThreadIndex if we do the joins inside the workflow
                #  That would allow us to replace this whole thing with a list comprehension
                new_fork_stack = fork_stack + (ForkStackItem(item.fork_id, node_run_id, ThreadIndex(i)),)
                tasks.append(GraphTask(item.fork_id, input_item, new_fork_stack))
            return tasks
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

def _is_fork_run_completed(tasks: Iterable[GraphTask], fork_run_id: NodeRunId) -> bool:
    # Check if any of the tasks in the graph have this fork_run_id in their fork_stack
    # If this is the case, then the fork run is not yet completed
    # TODO: We actually only need to check if the node is upstream of the fork run
    for t in tasks:
        if fork_run_id in {x.node_run_id for x in t.fork_stack}:
            return False
    return True