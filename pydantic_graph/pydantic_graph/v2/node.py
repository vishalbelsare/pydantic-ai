from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pydantic_graph.v2.id_types import ForkId, NodeId


class StartNode(str, Enum):
    start = 'start'

    @property
    def id(self) -> NodeId:
        return NodeId(f'__{self.value}__')


class EndNode(str, Enum):
    end = 'end'

    @property
    def id(self) -> NodeId:
        return NodeId(f'__{self.value}__')


@dataclass
class Spread[InputT, OutputT]:
    # Note — the InputT should always be Sequence[OutputT]; we enforce this by making it hard to instantiate in any other way
    id: ForkId

    def _force_variance(self, inputs: InputT) -> OutputT:
        raise RuntimeError('This method should never be called, it is just defined for typing purposes.')


START = StartNode.start
END = EndNode.end
