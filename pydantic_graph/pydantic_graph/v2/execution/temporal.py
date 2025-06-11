from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence, cast

from temporalio import activity, workflow

from pydantic_graph.v2.execution.graph_task import GraphTask
from pydantic_graph.v2.execution.graph_walker import EndMarker, GraphRunner, JoinItem
from pydantic_graph.v2.graph import Graph
from pydantic_graph.v2.step import Step, StepContext

type ConcreteStateT = Any
type ConcreteDepsT = Any


@dataclass
class GraphWorkflowInputs[StateT, DepsT]:
    state: StateT
    deps: DepsT
    inputs: Any


# TODO(P1): Need to confirm if this works as a temporal workflow...
# TODO(P3): Implement a version that supports a dynamic graph, which would be one of the inputs.
#   (This is dependent on being able to serialize/deserialize the graph, but I think that should be doable as long as
#   nodes can be serialized/deserialized by way of references to their underlying functions, similar to how
#   Evaluators work.)
def get_temporal_stuff[StateT, DepsT, InputsT, OutputT](
    graph: Graph[StateT, DepsT, InputsT, OutputT],
):
    if not TYPE_CHECKING:
        # Do this to ensure that we get the actual types that can be introspected for serialization/deserialization
        ConcreteStateT = graph.state_type
        ConcreteDepsT = graph.deps_type

    # TODO: Does anything besides a Step need to be automatically-converted into an activity? (I think not.)
    #   Note even steps don't _need_ to be converted, it just might make things a bit more convenient.
    activities: list[Any] = []
    for node in graph.nodes:
        if isinstance(node, Step) and node.is_activity:
            original_call = node.call
            # TODO: Need to track input and output types for the step so that we can get proper serialization/deserialization.
            #  Right now this will only work if the inputs and outputs of the step are trivially serializable.
            @activity.defn
            async def node_activity(ctx: StepContext[StateT, DepsT, Any]) -> Any:
                return await workflow.execute_activity(original_call, ctx)  # type: ignore
            node._call = node_activity
            activities.append(node_activity)

    @workflow.defn
    class RunGraphWorkflow:
        @workflow.query
        def get_state(self) -> ConcreteStateT:
            return self.state

        # TODO: This should probably use a wrapper dataclass as input
        @workflow.update
        def set_state(self, state: ConcreteStateT) -> None:
            # This makes it possible to do "interrupt" workflows within nodes while relying purely on state.
            self.state = state

        @workflow.run
        async def run(self, inputs: GraphWorkflowInputs[ConcreteStateT, ConcreteDepsT]):
            # Weird that this is necessary...
            self.state = inputs.state
            self.deps = inputs.deps
            self.inputs = inputs.inputs

            async with GraphRunner(graph).iter(self.state, self.deps, self.inputs) as graph_run:
                async for event in graph_run:
                    if isinstance(event, EndMarker):
                        return self.state, event.value

            raise RuntimeError('Graph run did not end with an EndMarker, which is unexpected.')

    @workflow.defn
    class IterGraphWorkflow:
        @workflow.init
        def __init__(self, inputs: GraphWorkflowInputs[ConcreteStateT, ConcreteDepsT]):
            self.state = inputs.state

            self._next = None
            self._next_is_ready = True

        @workflow.query
        def get_state(self) -> ConcreteStateT:
            return self.state

        # TODO: This should probably use a wrapper dataclass as input
        @workflow.update
        def set_state(self, state: ConcreteStateT) -> None:
            self.state = state

        # TODO: This should probably use a wrapper dataclass as input
        @workflow.update
        def next(self, value: EndMarker[OutputT] | JoinItem | Sequence[GraphTask] | None) -> None:
            if value is not None:
                self._next = value
            self._next_is_ready = True

        @workflow.run
        async def run(self, inputs: GraphWorkflowInputs[ConcreteStateT, ConcreteDepsT]):
            # Weird that this is necessary...
            self.state = inputs.state

            async with GraphRunner(graph).iter(self.state, inputs.deps, inputs.inputs) as graph_run:
                while True:
                    await workflow.wait_condition(lambda: self.waiting_for_next)
                    self.waiting_for_next = False
                    try:
                        self._next = await graph_run.next(self._next)
                    except StopAsyncIteration:
                        assert isinstance(self._next, EndMarker), 'Graph run did not end with an EndMarker, which is unexpected.'
                        return state, cast(EndMarker[OutputT], self._next).value

    return handle_graph_node_activity, RunGraphWorkflow, IterGraphWorkflow
