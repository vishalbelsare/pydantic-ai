from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import Any, Generic

from typing_extensions import TypeVar

from ..span_tree import SpanTree

__all__ = ('ScoringContext',)

InputsT = TypeVar('InputsT', default=dict[str, Any])
OutputT = TypeVar('OutputT', default=dict[str, Any])
MetadataT = TypeVar('MetadataT', default=dict[str, Any])


@dataclass
class ScoringContext(Generic[InputsT, OutputT, MetadataT]):
    """Context for scoring an evaluation case."""

    name: str
    inputs: InputsT
    metadata: MetadataT
    expected_output: OutputT | None

    output: OutputT
    duration: float
    span_tree: SpanTree

    attributes: dict[str, Any]
    metrics: dict[str, int | float]
