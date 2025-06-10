from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic_graph.v2.execution.graph_task import GraphTask
from pydantic_graph.v2.execution.graph_walker import EndMarker, GraphActivityInputs, GraphWalker, JoinItem, handle_node
from pydantic_graph.v2.graph import Graph
from pydantic_graph.v2.state import StateManagerFactory

type ConcreteStateT = Any
type ConcreteDepsT = Any


@dataclass
class GraphWorkflowInputs[StateT, DepsT]:
    state: StateT
    deps: DepsT
    inputs: Any


# TODO(P1): Need to confirm if this works as a temporal workflow...
# TODO(P1): Need to make a TemporalGraphRunner or otherwise change the API of GraphRunner
# TODO(P3): Implement a version that supports a dynamic graph, which would be one of the inputs.
#   (This is dependent on being able to serialize/deserialize the graph, but I think that should be doable as long as
#   nodes can be serialized/deserialized by way of references to their underlying functions, similar to how
#   Evaluators work.)
def get_temporal_stuff[StateT, DepsT, InputsT, OutputT](
    graph: Graph[StateT, DepsT, InputsT, OutputT],
    state_manager_factory: StateManagerFactory,
):
    if not TYPE_CHECKING:
        # Do this to ensure that we get the actual types that can be introspected for serialization/deserialization
        ConcreteStateT = graph.state_type
        ConcreteDepsT = graph.deps_type

    # from temporalio import activity, workflow
    # @activity.defn
    async def handle_graph_node_activity(
        inputs: GraphActivityInputs[ConcreteStateT, ConcreteDepsT],
    ) -> Sequence[GraphTask] | JoinItem | EndMarker:
        return await handle_node(inputs, graph)

    # @workflow.defn
    class GraphWorkflow:
        # @workflow.run
        async def run(self, inputs: GraphWorkflowInputs[ConcreteStateT, ConcreteDepsT]):
            return await GraphWalker(graph, state_manager_factory, handle_graph_node_activity).run(
                inputs.state, inputs.deps, inputs.inputs
            )

    return handle_graph_node_activity, GraphWorkflow
