from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic_graph.v2.graph import Graph

type GraphRunResult[StateT, OutputT] = tuple[StateT, OutputT]


class GraphRunner[StateT, DepsT, InputT, OutputT](ABC):
    graph: Graph[StateT, DepsT, InputT, OutputT]

    @abstractmethod
    async def run(
        self,
        state: StateT,
        deps: DepsT,
        inputs: InputT,
    ) -> GraphRunResult[StateT, OutputT]:
        raise NotImplementedError

    # @abstractmethod
    # async def run_soon(
    #     self,
    #     state: StateT,
    #     deps: DepsT,
    #     inputs: InputT,
    # ) -> GraphRunId:
    #     raise NotImplementedError
    #
    # @abstractmethod
    # async def get_result(self, run_id: GraphRunId, clean_up: bool = False) -> Maybe[GraphRunResult[StateT, OutputT]]:
    #     raise NotImplementedError
    #
    # @abstractmethod
    # async def pause(self, run_id: GraphRunId) -> None:
    #     raise NotImplementedError
    #
    # @abstractmethod
    # async def terminate(self, run_id: GraphRunId) -> None:
    #     raise NotImplementedError
