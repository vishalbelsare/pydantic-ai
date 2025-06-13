from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from temporalio import activity, workflow

from pydantic_graph.v2.execution.graph_walker import EndMarker, GraphRunner
from pydantic_graph.v2.graph import Graph
from pydantic_graph.v2.step import Step


def get_temporal_graph[StateT, InputsT, OutputT](
    graph: Graph[StateT, InputsT, OutputT],
) -> tuple[Graph[StateT, InputsT, OutputT], list[Any]]:
    new_nodes = {k: copy(v) for k, v in graph.nodes.items()}
    activities: list[Any] = []
    for node in new_nodes.values():
        if isinstance(node, Step) and node.activity:
            node_activity = activity.defn(name=f'node:{node.id}')(node.call)

            async def new_node_call(ctx: Any) -> Any:
                return await workflow.execute_activity(node_activity, ctx)  # pyright: ignore[reportUnknownMemberType]

            node._call = new_node_call  # pyright: ignore[reportPrivateUsage]
            node.activity = False  # The call no longer needs to be converted into an activity
            activities.append(node_activity)

    new_graph = Graph(
        state_type=graph.state_type,
        input_type=graph.input_type,
        output_type=graph.output_type,
        nodes=new_nodes,
        edges_by_source=graph.edges_by_source,
        parent_forks=graph.parent_forks,
    )
    return new_graph, activities


type ConcreteStateT = Any
type ConcreteGraphInputT = Any
type ConcreteGraphOutputT = Any


@dataclass
class GraphWorkflowInputs[StateT, InputT]:
    state: StateT
    inputs: InputT


# TODO(P3): Implement a version that supports a dynamic graph, which would be one of the inputs.
#  (This would depend on only making use of serializable nodes, e.g. dataclasses, along with an appropriate loader.)
def get_simple_workflow[StateT, InputsT, OutputT](graph: Graph[StateT, InputsT, OutputT]):
    if not TYPE_CHECKING:
        # Do this to ensure that we get the actual types that can be introspected for serialization/deserialization
        ConcreteStateT = graph.state_type
        ConcreteGraphInputT = graph.input_type
        ConcreteGraphOutputT = graph.output_type

    # Some simple example workflows; you can do whatever you want with the graph
    @workflow.defn
    class RunGraphWorkflow:
        @workflow.init
        def __init__(self, inputs: GraphWorkflowInputs[ConcreteStateT, ConcreteGraphInputT]):
            self.state = inputs.state
            self.inputs = inputs.inputs

        @workflow.run
        async def run(self, inputs: GraphWorkflowInputs[ConcreteStateT, ConcreteGraphInputT]):
            self.state = inputs.state
            self.inputs = inputs.inputs

            async with GraphRunner(graph).iter(self.state, self.inputs) as graph_run:
                async for event in graph_run:
                    if isinstance(event, EndMarker):
                        return self.state, event.value

            raise RuntimeError('Graph run did not end with an EndMarker, which is unexpected.')

    return RunGraphWorkflow

    # @workflow.defn
    # class IterGraphWorkflow:
    #     @workflow.init
    #     def __init__(self, inputs: GraphWorkflowInputs[ConcreteStateT, ConcreteDepsT]):
    #         self.state = inputs.state
    #
    #         self._next = None
    #         self._next_is_ready = True
    #
    #     @workflow.query
    #     def get_state(self) -> ConcreteStateT:
    #         return self.state
    #
    #     # TODO: This should probably use a wrapper dataclass as input
    #     @workflow.update
    #     def set_state(self, state: ConcreteStateT) -> None:
    #         self.state = state
    #
    #     # TODO: This should probably use a wrapper dataclass as input
    #     @workflow.update
    #     def next(self, value: EndMarker[OutputT] | JoinItem | Sequence[GraphTask] | None) -> None:
    #         if value is not None:
    #             self._next = value
    #         self._next_is_ready = True
    #
    #     @workflow.run
    #     async def run(self, inputs: GraphWorkflowInputs[ConcreteStateT, ConcreteDepsT]):
    #         # Weird that this is necessary...
    #         self.state = inputs.state
    #
    #         async with GraphRunner(graph).iter(self.state, inputs.deps, inputs.inputs) as graph_run:
    #             while True:
    #                 await workflow.wait_condition(lambda: self.waiting_for_next)
    #                 self.waiting_for_next = False
    #                 try:
    #                     self._next = await graph_run.next(self._next)
    #                 except StopAsyncIteration:
    #                     assert isinstance(self._next, EndMarker), 'Graph run did not end with an EndMarker, which is unexpected.'
    #                     return state, cast(EndMarker[OutputT], self._next).value

    # return handle_graph_node_activity, RunGraphWorkflow, IterGraphWorkflow
