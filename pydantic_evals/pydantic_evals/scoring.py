from __future__ import annotations as _annotations

from dataclasses import dataclass, field
from typing import Any, Generic

from typing_extensions import TypeVar

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
    output: OutputT
    expected_output: OutputT | None

    attributes: dict[str, Any]
    metrics: dict[str, int | float]

    # TODO: Allow storing a reason with the scores
    scores: dict[str, int | float] = field(init=False, default_factory=dict)
    # TODO: Allow storing a reason with the labels
    labels: dict[str, bool | str] = field(init=False, default_factory=dict)

    def record_score(self, name: str, value: int | float) -> None:
        self.scores[name] = value

    def record_label(self, name: str, value: bool | str) -> None:
        self.labels[name] = value
