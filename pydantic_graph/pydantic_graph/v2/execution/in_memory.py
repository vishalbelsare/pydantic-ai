from __future__ import annotations

from dataclasses import dataclass

from pydantic_graph.v2.execution.graph_runner import GraphRunner
from pydantic_graph.v2.execution.graph_walker import GraphWalker
from pydantic_graph.v2.graph import Graph


@dataclass
class InMemoryGraphRunner[StateT, DepsT, InputT, OutputT](GraphRunner[StateT, DepsT, InputT, OutputT]):
    graph: Graph[StateT, DepsT, InputT, OutputT]

    async def run(
        self,
        state: StateT,
        deps: DepsT,
        inputs: InputT,
    ) -> tuple[StateT, OutputT]:
        graph_walker = GraphWalker(self.graph)
        return await graph_walker.run(state, deps, inputs)
