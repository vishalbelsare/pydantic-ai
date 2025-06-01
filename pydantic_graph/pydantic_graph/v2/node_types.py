from __future__ import annotations

from typing import Any

from typing_extensions import TypeGuard

from pydantic_graph.v2.decision import Decision
from pydantic_graph.v2.join import Join
from pydantic_graph.v2.node import EndNode, Spread, StartNode
from pydantic_graph.v2.step import Step

type MiddleNode[StateT, DepsT, InputT, OutputT] = (
    Step[StateT, DepsT, InputT, OutputT] | Join[StateT, DepsT, InputT, OutputT] | Spread[InputT, OutputT]
)

type SourceNode[StateT, DepsT, OutputT] = MiddleNode[StateT, DepsT, Any, OutputT]  # | StartNode
type DestinationNode[StateT, DepsT, InputT, GraphOutputT] = (
    MiddleNode[StateT, DepsT, InputT, Any] | Decision[StateT, DepsT, InputT, GraphOutputT]
)  # | EndNode

type AnySourceNode = SourceNode[Any, Any, Any] | StartNode
type AnyDestinationNode = DestinationNode[Any, Any, Any, Any] | EndNode
type AnyNode = AnySourceNode | AnyDestinationNode


def is_source(node: AnyNode) -> TypeGuard[AnySourceNode]:
    return isinstance(node, (StartNode, Step, Join))


def is_destination(node: AnyNode) -> TypeGuard[AnyDestinationNode]:
    return isinstance(node, (EndNode, Step, Join, Decision))
