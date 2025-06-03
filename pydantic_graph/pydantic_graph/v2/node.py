from __future__ import annotations

from dataclasses import dataclass

from pydantic_graph.v2.id_types import ForkId, NodeId


class StartNode[OutputT]:
    id = ForkId(NodeId('__start__'))


class EndNode[InputT]:
    id = NodeId('__end__')

    def _force_variance(self, inputs: InputT) -> None:
        raise RuntimeError('This method should never be called, it is just defined for typing purposes.')

    # def _force_variance(self) -> InputT:
    #     raise RuntimeError('This method should never be called, it is just defined for typing purposes.')


@dataclass
class Fork[InputT, OutputT]:
    id: ForkId

    is_spread: bool  # if is_spread is True, InputT must be Sequence[OutputT]; otherwise InputT must be OutputT

    def _force_variance(self, inputs: InputT) -> OutputT:
        raise RuntimeError('This method should never be called, it is just defined for typing purposes.')
