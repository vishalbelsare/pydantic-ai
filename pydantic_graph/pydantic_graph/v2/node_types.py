from __future__ import annotations

from typing import Any

from typing_extensions import TypeGuard

from pydantic_graph.v2.decision import Decision
from pydantic_graph.v2.join import Join
from pydantic_graph.v2.node import EndNode, Spread, StartNode
from pydantic_graph.v2.step import Step

type AnyMiddleNode = Step[Any, Any, Any, Any] | Join[Any, Any, Any, Any] | Spread[Any, Any]
type AnySourceNode = AnyMiddleNode | StartNode
type AnyDestinationNode = AnyMiddleNode | EndNode | Decision[Any, Any, Any, Any]
type AnyNode = AnySourceNode | AnyDestinationNode


def is_source(node: AnyNode) -> TypeGuard[AnySourceNode]:
    return isinstance(node, (StartNode, Step, Join))


def is_destination(node: AnyNode) -> TypeGuard[AnyDestinationNode]:
    return isinstance(node, (EndNode, Step, Join, Decision))
