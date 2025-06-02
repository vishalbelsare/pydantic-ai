from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable, Self

from pydantic_graph.v2.node_types import DestinationNode, SourceNode
from pydantic_graph.v2.transform import TransformFunction


@dataclass
class _TransformMarker:
    transform: TransformFunction[Any, Any, Any, Any]


@dataclass
class _SpreadMarker:
    pass


@dataclass
class _ForkMarker:
    forks: Sequence[_PartialPath]


@dataclass
class _LabelMarker:
    label: str


@dataclass
class _DestinationMarker:
    destination: DestinationNode[Any, Any, Any, Any]


type _PathItem = _TransformMarker | _SpreadMarker | _ForkMarker | _LabelMarker | _DestinationMarker


@dataclass
class _PartialPath:
    items: Sequence[_PathItem]


@dataclass
class Path(_PartialPath):
    items: Sequence[_PathItem]


# @dataclass
class PathBuilder[StateT, DepsT, OutputT]:
    working_items: Sequence[_PathItem]

    def __init__(self, working_items: Sequence[_PathItem]):
        self.working_items = working_items

    def to(
        self,
        destination: DestinationNode[StateT, DepsT, OutputT],
        /,
        *extra_destinations: DestinationNode[StateT, DepsT, OutputT],
    ) -> Path:
        if extra_destinations:
            next_item = _ForkMarker(
                forks=[_PartialPath(items=[_DestinationMarker(d)]) for d in (destination,) + extra_destinations]
            )
        else:
            next_item = _DestinationMarker(destination=destination)
        return Path(items=[*self.working_items, next_item])

    def fork(self, forks: Sequence[Path], /) -> Path:
        next_item = _ForkMarker(forks=forks)
        return Path(items=[*self.working_items, next_item])

    def transform[NewOutputT](
        self, func: TransformFunction[StateT, DepsT, OutputT, NewOutputT], /
    ) -> PathBuilder[StateT, DepsT, NewOutputT]:
        next_item = _TransformMarker(func)
        return PathBuilder[StateT, DepsT, NewOutputT](working_items=[*self.working_items, next_item])

    def spread[T](self: PathBuilder[StateT, DepsT, Sequence[T]]) -> PathBuilder[StateT, DepsT, T]:
        next_item = _SpreadMarker()
        return PathBuilder[StateT, DepsT, T](working_items=[*self.working_items, next_item])

    def label(self, label: str) -> PathBuilder[StateT, DepsT, OutputT]:
        next_item = _LabelMarker(label)
        return PathBuilder[StateT, DepsT, OutputT](working_items=[*self.working_items, next_item])


@dataclass
class EdgePath[StateT, DepsT]:
    sources: Sequence[SourceNode[StateT, DepsT, Any]]
    path: Path


# @dataclass
class EdgePathBuilder[StateT, DepsT, OutputT]:
    sources: Sequence[SourceNode[StateT, DepsT, Any]]

    def __init__(
        self, sources: Sequence[SourceNode[StateT, DepsT, Any]], path_builder: PathBuilder[StateT, DepsT, OutputT]
    ):
        self.sources = sources
        self._path_builder = path_builder

    @property
    def path_builder(self) -> PathBuilder[StateT, DepsT, OutputT]:
        return self._path_builder

    # TODO: to makes things invariant... how can this be fixed?
    def to(
        self,
        destination: DestinationNode[StateT, DepsT, OutputT],
        /,
        *extra_destinations: DestinationNode[StateT, DepsT, OutputT],
    ) -> EdgePath[StateT, DepsT]:
        return EdgePath(sources=self.sources, path=self.path_builder.to(destination, *extra_destinations))

    def fork(self, get_forks: Callable[[Self], Sequence[EdgePath[StateT, DepsT]]], /) -> EdgePath[StateT, DepsT]:
        return EdgePath(
            sources=self.sources, path=self.path_builder.fork([Path(x.path.items) for x in get_forks(self)])
        )

    def transform[NewOutputT](
        self, func: TransformFunction[StateT, DepsT, OutputT, NewOutputT], /
    ) -> EdgePathBuilder[StateT, DepsT, NewOutputT]:
        return EdgePathBuilder(sources=self.sources, path_builder=self.path_builder.transform(func))

    def spread[T](self: EdgePathBuilder[StateT, DepsT, Sequence[T]]) -> EdgePathBuilder[StateT, DepsT, T]:
        return EdgePathBuilder(sources=self.sources, path_builder=self.path_builder.spread())

    def label(self, label: str) -> EdgePathBuilder[StateT, DepsT, OutputT]:
        return EdgePathBuilder(sources=self.sources, path_builder=self.path_builder.label(label))
