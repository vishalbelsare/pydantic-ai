from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, get_args, get_origin

from anyio import create_task_group
from typing_extensions import Literal

from pydantic_graph.v2.decision import Decision
from pydantic_graph.v2.execution.graph_runner import GraphRunAPI
from pydantic_graph.v2.execution.graph_task import GraphTask
from pydantic_graph.v2.graph import Graph
from pydantic_graph.v2.id_types import ForkId, NodeRunId
from pydantic_graph.v2.join import Join, ReducerContext
from pydantic_graph.v2.node import (
    EndNode,
    Fork,
    StartNode,
)
from pydantic_graph.v2.node_types import AnyNode
from pydantic_graph.v2.paths import BroadcastMarker, DestinationMarker, LabelMarker, Path, SpreadMarker, TransformMarker
from pydantic_graph.v2.step import Step, StepContext
from pydantic_graph.v2.transform import TransformContext
from pydantic_graph.v2.util import Maybe, TypeExpression

ForkStack = tuple[tuple[ForkId, NodeRunId], ...]


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

        self.result: Maybe[OutputT] = None

    async def run(self, state: StateT, inputs: InputT) -> OutputT:
        await self.run_api.initialize_state(state)

        start_task = GraphTask(node_id=StartNode.id, inputs=inputs, fork_stack=())
        await self._request_task(start_task)
        await self.run_api.wait()

        result = await self.run_api.get_result()
        if result is None:
            raise RuntimeError(
                'Graph run completed, but no result was produced. This is either a bug in the graph or a bug in the graph runner.'
            )

        return result.value

    async def handle_task(self, task: GraphTask) -> None:
        node = self.graph.nodes[task.node_id]
        await self._handle_node(node, task.inputs, task.fork_stack)
        await self._clean_up_task(task)

    async def _handle_node(self, node: AnyNode, inputs: Any, fork_stack: ForkStack):
        if isinstance(node, (StartNode, Fork)):
            await self._handle_edges(node, inputs, fork_stack)
        elif isinstance(node, Step):
            await self._handle_step(node, inputs, fork_stack)
        elif isinstance(node, Join):
            # TODO: When reducing a join, need to somehow record that it happened to prevent issues if the task is
            #  restarted or similar
            await self._handle_reduce_join(node, inputs, fork_stack)
        elif isinstance(node, Decision):
            await self._handle_decision(node, inputs, fork_stack)
        elif isinstance(node, EndNode):
            await self._handle_end(inputs)

    async def _handle_step(self, step: Step[Any, Any, Any, Any], inputs: Any, fork_stack: ForkStack) -> None:
        step_context = StepContext[StateT, DepsT, Any](self.run_api, self.deps, inputs)
        output = await step.call(step_context)
        await self._handle_edges(step, output, fork_stack)

    async def _handle_reduce_join(self, join: Join[Any, Any, Any, Any], inputs: Any, fork_stack: ForkStack) -> None:
        # Find the matching fork run id in the stack; this will be used to look for an active reducer
        parent_fork = self.graph.get_parent_fork(join.id)
        matching_fork_run_id = next(iter(x[1] for x in fork_stack[::-1] if x[0] == parent_fork.fork_id), None)
        if matching_fork_run_id is None:
            raise RuntimeError(
                f'Fork {parent_fork.fork_id} not found in stack {fork_stack}. This means the dominating fork is not dominating (this is a bug).'
            )

        # Get or create the active reducer
        reducer = await self.run_api.get_active_reducer_state(join.id, matching_fork_run_id)
        state = await self.run_api.get_immutable_state()
        ctx = ReducerContext(state, self.deps, inputs)

        # TODO: Need to store reductions in the database and have a way to ensure executions are idempotent
        if reducer is None:
            reducer = join.reducer_factory(ctx)
            await self.run_api.set_active_reducer_state(join.id, matching_fork_run_id, reducer)
        else:
            reducer[0](ctx)

    async def _handle_decision(
        self, decision: Decision[StateT, DepsT, Any], inputs: Any, fork_stack: ForkStack
    ) -> None:
        for branch in decision.branches:
            match_tester = branch.matches
            branch_source = branch.source
            if get_origin(branch_source) is TypeExpression:
                branch_source = get_args(branch_source)[0]
            if match_tester is not None:
                inputs_match = match_tester(inputs)
            elif branch_source in {Any, object}:
                inputs_match = True
            elif get_origin(branch_source) is Literal:
                inputs_match = inputs in get_args(branch_source)
            else:
                try:
                    inputs_match = isinstance(inputs, branch_source)
                except TypeError as e:
                    raise RuntimeError(f'Decision branch source {branch_source} is not a valid type.') from e

            if inputs_match:
                await self._handle_path(branch.path, inputs, fork_stack)
                return

        raise RuntimeError(f'No branch matched inputs {inputs} for decision node {decision}.')

    async def _handle_end(self, inputs: Any) -> None:
        await self.run_api.set_result(inputs)

    async def _handle_path(self, path: Path, inputs: Any, fork_stack: ForkStack) -> None:
        if not path.items:
            return

        item = path.items[0]
        if isinstance(item, DestinationMarker):
            await self._handle_node(item.destination, inputs, fork_stack)
        elif isinstance(item, SpreadMarker):
            node_run_id = NodeRunId(str(uuid.uuid4()))
            fork_stack += ((item.fork_id, node_run_id),)
            for input_item in inputs:
                await self.run_api.request_task(GraphTask(item.fork_id, input_item, fork_stack))
        elif isinstance(item, BroadcastMarker):
            node_run_id = NodeRunId(str(uuid.uuid4()))
            fork_stack += ((item.fork_id, node_run_id),)
            await self.run_api.request_task(GraphTask(item.fork_id, inputs, fork_stack))
        elif isinstance(item, TransformMarker):
            state = await self.run_api.get_immutable_state()
            ctx = TransformContext(state, self.deps, inputs)
            inputs = item.transform(ctx)
            await self._handle_path(path.next_path, inputs, fork_stack)
        elif isinstance(item, LabelMarker):
            await self._handle_path(path.next_path, inputs, fork_stack)

    async def _handle_edges(self, node: AnyNode, inputs: Any, fork_stack: ForkStack) -> None:
        edges = self.graph.edges_by_source.get(node.id, [])
        assert len(edges) == 1 or isinstance(node, Fork)  # this should have already been ensured during graph building

        async with create_task_group() as task_group:
            for path in edges:
                task_group.start_soon(self._handle_path, path, inputs, fork_stack)

    async def _request_task(self, task: GraphTask) -> None:
        await self.run_api.request_task(task)

    async def _clean_up_task(self, task: GraphTask) -> None:
        await self.run_api.complete_requested_task(task.task_id)
        await self._handle_finalize_joins(task.fork_stack)
        if not await self.run_api.any_tasks_remain():
            await self.run_api.mark_finished()

    async def _handle_finalize_joins(self, popped_task_fork_stack: ForkStack) -> None:
        # If the popped task was the last item preventing one or more joins, those joins can now be finalized
        task_fork_run_indices = {fork_run_id: i for i, (_, fork_run_id) in enumerate(popped_task_fork_stack)}

        # Note: might be more efficient to maintain a better data structure for looking up reducers by join_id and
        # fork_run_id without iterating through every item. This only matters if there is a large number of reducers.
        for (join_id, fork_run_id), reducer in await self.run_api.get_active_reducers_with_fork_run_id(
            list(task_fork_run_indices)
        ):
            fork_run_index = task_fork_run_indices.get(fork_run_id)
            assert fork_run_index is not None  # should be filtered by the _get_reducers_with_fork_run_id method

            # This reducer _may_ now be ready to finalize:
            if await self.run_api.is_fork_run_completed(fork_run_id):
                state = await self.run_api.get_immutable_state()
                ctx = ReducerContext(state, self.deps, None)
                output = reducer[1](ctx)
                new_fork_stack = popped_task_fork_stack[:fork_run_index]
                await self.run_api.complete_active_reducer(join_id, fork_run_id)

                # Should _now_ traverse the edges leaving this join
                node = self.graph.nodes[join_id]

                await self._handle_edges(node, output, new_fork_stack)
