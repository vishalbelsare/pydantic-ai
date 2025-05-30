from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, get_args, get_origin

from typing_extensions import Literal, assert_never

from pydantic_graph.v2.decision import Decision
from pydantic_graph.v2.execution.graph_runner import GraphRunAPI
from pydantic_graph.v2.execution.graph_task import GraphTask
from pydantic_graph.v2.graph import Graph
from pydantic_graph.v2.id_types import ForkId, NodeRunId
from pydantic_graph.v2.join import Join, ReducerContext
from pydantic_graph.v2.node import (
    START,
    EndNode,
    Spread,
    StartNode,
)
from pydantic_graph.v2.step import Step, StepContext
from pydantic_graph.v2.transform import TransformContext
from pydantic_graph.v2.util import Maybe


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

        start_task = GraphTask(node_id=START.id, context_inputs=inputs, node_inputs=inputs, fork_stack=())
        await self.request_task(start_task)
        await self.run_api.wait()

        result = await self.run_api.get_result()
        if result is None:
            raise RuntimeError(
                'Graph run completed, but no result was produced. This is either a bug in the graph or a bug in the graph runner.'
            )

        return result.value

    async def request_task(self, task: GraphTask) -> None:
        await self.run_api.request_task(task)

    async def handle_task(self, task: GraphTask) -> None:
        node = self.graph.nodes[task.node_id]

        if isinstance(node, StartNode):
            await self._handle_start(task)
        elif isinstance(node, Step):
            await self._handle_step(node, task)
        elif isinstance(node, Join):
            await self._handle_reduce_join(node, task)
        elif isinstance(node, Spread):
            await self._handle_spread(task)
        elif isinstance(node, Decision):
            await self._handle_decision(node, task)
        elif isinstance(node, EndNode):
            await self._handle_end(task)

        await self._clean_up_task(task)

    async def _clean_up_task(self, task: GraphTask) -> None:
        await self.run_api.complete_requested_task(task.task_id)
        await self._handle_finalize_joins(task)
        if not await self.run_api.any_tasks_remain():
            await self.run_api.mark_finished()

    async def _handle_start(self, walk: GraphTask) -> None:
        # nothing to do besides start the graph
        await self._handle_edges(walk, walk.context_inputs, walk.node_inputs)

    async def _handle_step(self, step: Step[Any, Any, Any, Any], task: GraphTask):
        step_context = StepContext[StateT, DepsT, Any](self.run_api, self.deps, task.context_inputs)
        output = await step.call(step_context)
        await self._handle_edges(task, output, output)

    async def _handle_reduce_join(self, join: Join[Any, Any, Any, Any], walk: GraphTask) -> None:
        # Find the matching fork run id in the stack; this will be used to look for an active reducer
        parent_fork = self.graph.get_parent_fork(join.id)
        matching_fork_run_id = next(iter(x[1] for x in walk.fork_stack[::-1] if x[0] == parent_fork.fork_id), None)
        if matching_fork_run_id is None:
            raise RuntimeError(
                f'Fork {parent_fork.fork_id} not found in stack {walk.fork_stack}. This means the dominating fork is not dominating (this is a bug).'
            )

        # Get or create the active reducer
        reducer = await self.run_api.get_active_reducer_state(join.id, matching_fork_run_id)
        state = await self.run_api.get_immutable_state()
        ctx = ReducerContext(state, self.deps, walk.node_inputs)

        if reducer is None:
            reducer = join.reducer_factory(ctx)
            await self.run_api.set_active_reducer_state(join.id, matching_fork_run_id, reducer)
        else:
            reducer[0](ctx)

    async def _handle_spread(self, walk: GraphTask):
        await self._handle_edges(walk, walk.context_inputs, walk.node_inputs)

    async def _handle_decision(self, decision: Decision[StateT, DepsT, Any, Any], walk: GraphTask) -> None:
        for branch in decision.branches:
            assert not branch.spread, 'Spreads decisions should be converted into spreads as part of graph-building'

            match_tester = branch.matches
            if match_tester is not None:
                inputs_match = match_tester(walk.node_inputs)
            elif branch.source in {Any, object}:
                inputs_match = True
            elif get_origin(branch.source) is Literal:
                inputs_match = walk.node_inputs in get_args(branch.source)
            else:
                inputs_match = isinstance(walk.node_inputs, branch.source)

            if inputs_match:
                node_inputs = walk.node_inputs
                for transform in branch.transforms:
                    state = await self.run_api.get_immutable_state()
                    ctx = TransformContext(state, self.deps, walk.context_inputs, node_inputs)
                    node_inputs = transform(ctx)
                await self.request_task(
                    GraphTask(branch.route_to.id, walk.context_inputs, walk.node_inputs, walk.fork_stack),
                )
                break

    async def _handle_end(self, walk: GraphTask) -> None:
        await self.run_api.set_result(walk.node_inputs)

    async def _handle_finalize_joins(self, popped_walk: GraphTask) -> None:
        # If the popped walk was the last item preventing one or more joins, those joins can now be finalized
        walk_fork_run_indices = {fork_run_id: i for i, (_, fork_run_id) in enumerate(popped_walk.fork_stack)}

        # Note: might be more efficient to maintain a better data structure for looking up reducers by join_id and
        # fork_run_id without iterating through every item. This only matters if there is a large number of reducers.
        for (join_id, fork_run_id), reducer in await self.run_api.get_active_reducers_with_fork_run_id(
            list(walk_fork_run_indices)
        ):
            fork_run_index = walk_fork_run_indices.get(fork_run_id)
            assert fork_run_index is not None  # should be filtered by the _get_reducers_with_fork_run_id method

            # This reducer _may_ now be ready to finalize:
            if await self.run_api.is_fork_run_completed(fork_run_id):
                state = await self.run_api.get_immutable_state()
                ctx = ReducerContext(state, self.deps, None)
                output = reducer[1](ctx)
                new_fork_stack = popped_walk.fork_stack[:fork_run_index]
                await self.run_api.complete_active_reducer(join_id, fork_run_id)

                # Should _now_ traverse the edges leaving this join
                await self._handle_edges(GraphTask(join_id, None, None, new_fork_stack), output, output)

    async def _handle_edges(self, walk: GraphTask, context_inputs: Any, next_node_inputs: Any) -> None:
        edges = self.graph.edges_by_source.get(walk.node_id, [])
        node = self.graph.nodes[walk.node_id]

        fork_stack = walk.fork_stack
        if len(edges) > 1 or isinstance(node, Spread):
            # first condition is a broadcast fork; note that the way graph building works,
            # spread nodes should never be broadcast edges; even if there are multiple spreads between two nodes
            # these should result in distinct spread instances
            node_run_id = NodeRunId(str(uuid.uuid4()))
            fork_stack += ((ForkId(walk.node_id), node_run_id),)

        assert not isinstance(node, (Decision, EndNode)), 'This method should not be called for Decision, EndNode'
        if isinstance(node, (StartNode, Step, Join)):
            # Edge transitions should be fast, so maybe don't need to be handled in parallel
            for edge in edges:
                if edge.transform is not None:
                    state = await self.run_api.get_immutable_state()
                    transform_context = TransformContext(state, self.deps, context_inputs, next_node_inputs)
                    next_node_inputs = edge.transform(transform_context)

                await self.request_task(GraphTask(edge.destination_id, context_inputs, next_node_inputs, fork_stack))
        elif isinstance(node, Spread):
            for edge in edges:
                for item in next_node_inputs:
                    if edge.transform is not None:
                        state = await self.run_api.get_immutable_state()
                        transform_context = TransformContext(state, self.deps, context_inputs, item)
                        item = edge.transform(transform_context)
                    await self.request_task(GraphTask(edge.destination_id, context_inputs, item, fork_stack))
        else:
            assert_never(node)
