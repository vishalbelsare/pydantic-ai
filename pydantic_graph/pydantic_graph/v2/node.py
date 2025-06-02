from __future__ import annotations

from dataclasses import dataclass

from pydantic_graph.v2.id_types import ForkId, NodeId


class StartNode[OutputT]:
    id = NodeId('__start__')


class EndNode[InputT]:
    id = NodeId('__end__')

    def _force_variance(self, inputs: InputT) -> None:
        raise RuntimeError('This method should never be called, it is just defined for typing purposes.')


@dataclass
class Spread[InputT, OutputT]:
    # Note — the InputT should always be Sequence[OutputT]; we enforce this by making it hard to instantiate in any other way
    id: ForkId

    def _force_variance(self, inputs: InputT) -> OutputT:
        raise RuntimeError('This method should never be called, it is just defined for typing purposes.')
