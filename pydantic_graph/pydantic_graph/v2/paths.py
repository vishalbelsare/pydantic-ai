from __future__ import annotations

import secrets
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Self, overload

from pydantic_graph.v2.id_types import ForkId, NodeId
from pydantic_graph.v2.transform import TransformFunction

if TYPE_CHECKING:
    from pydantic_graph.v2.node_types import DestinationNode, SourceNode


@dataclass
class TransformMarker:
    # TODO: Transforms need to be serializable so graphs can be serialized
    transform: TransformFunction[Any, Any, Any, Any]


@dataclass
class SpreadMarker:
    fork_id: ForkId


@dataclass
class BroadcastMarker:
    paths: Sequence[Path]
    fork_id: ForkId


@dataclass
class LabelMarker:
    label: str


@dataclass
class DestinationMarker:
    # TODO: destination here should be an ID, not an actual node, to help with serialization
    destination: DestinationNode[Any, Any, Any]


type PathItem = TransformMarker | SpreadMarker | BroadcastMarker | LabelMarker | DestinationMarker


@dataclass
class Path:
    items: Sequence[PathItem]

    @property
    def last_fork(self) -> BroadcastMarker | SpreadMarker | None:
        """Returns the last fork or spread marker in the path, if any."""
        for item in reversed(self.items):
            if isinstance(item, (BroadcastMarker, SpreadMarker)):
                return item
        return None

    @property
    def next_path(self) -> Path:
        return Path(self.items[1:])


# @dataclass  # TODO: Change this back to a dataclass if we can do so without variance issues
class PathBuilder[StateT, DepsT, OutputT]:
    working_items: Sequence[PathItem]

    def __init__(self, working_items: Sequence[PathItem]):
        self.working_items = working_items

    @property
    def last_fork(self) -> BroadcastMarker | SpreadMarker | None:
        """Returns the last fork or spread marker in the path, if any."""
        for item in reversed(self.working_items):
            if isinstance(item, (BroadcastMarker, SpreadMarker)):
                return item
        return None

    def to(
        self,
        destination: DestinationNode[StateT, DepsT, OutputT],
        /,
        *extra_destinations: DestinationNode[StateT, DepsT, OutputT],
        fork_id: str | None = None,
    ) -> Path:
        if extra_destinations:
            next_item = BroadcastMarker(
                paths=[Path(items=[DestinationMarker(d)]) for d in (destination,) + extra_destinations],
                fork_id=ForkId(NodeId(fork_id or 'extra_broadcast_' + secrets.token_hex(8))),
            )
        else:
            next_item = DestinationMarker(destination=destination)
        return Path(items=[*self.working_items, next_item])

    def fork(self, forks: Sequence[Path], /, *, fork_id: str | None = None) -> Path:
        next_item = BroadcastMarker(paths=forks, fork_id=ForkId(NodeId(fork_id or 'broadcast_' + secrets.token_hex(8))))
        return Path(items=[*self.working_items, next_item])

    def transform[NewOutputT](
        self, func: TransformFunction[StateT, DepsT, OutputT, NewOutputT], /
    ) -> PathBuilder[StateT, DepsT, NewOutputT]:
        next_item = TransformMarker(func)
        return PathBuilder[StateT, DepsT, NewOutputT](working_items=[*self.working_items, next_item])

    def spread[T](
        self: PathBuilder[StateT, DepsT, Iterable[T]], *, fork_id: str | None = None
    ) -> PathBuilder[StateT, DepsT, T]:
        next_item = SpreadMarker(fork_id=ForkId(NodeId(fork_id or 'spread_' + secrets.token_hex(8))))
        return PathBuilder[StateT, DepsT, T](working_items=[*self.working_items, next_item])

    def label(self, label: str, /) -> PathBuilder[StateT, DepsT, OutputT]:
        next_item = LabelMarker(label)
        return PathBuilder[StateT, DepsT, OutputT](working_items=[*self.working_items, next_item])


@dataclass
class EdgePath[StateT, DepsT]:
    sources: Sequence[SourceNode[StateT, DepsT, Any]]
    path: Path


# @dataclass  # TODO: Change this back to a dataclass if we can do so without variance issues
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

    @property
    def last_fork_id(self) -> ForkId | None:
        last_fork = self._path_builder.last_fork
        if last_fork is None:
            return None
        return last_fork.fork_id

    # TODO: does `to` make things invariant? If so, how can this be fixed?
    @overload
    def to(
        self, get_forks: Callable[[Self], Sequence[EdgePath[StateT, DepsT]]], /, *, fork_id: str | None = None
    ) -> EdgePath[StateT, DepsT]: ...

    @overload
    def to(
        self, /, *destinations: DestinationNode[StateT, DepsT, OutputT], fork_id: str | None = None
    ) -> EdgePath[StateT, DepsT]: ...

    def to(
        self,
        first_item: DestinationNode[StateT, DepsT, OutputT] | Callable[[Self], Sequence[EdgePath[StateT, DepsT]]],
        /,
        *extra_destinations: DestinationNode[StateT, DepsT, OutputT],
        fork_id: str | None = None,
    ) -> EdgePath[StateT, DepsT]:
        if callable(first_item):
            return EdgePath(
                sources=self.sources,
                path=self.path_builder.fork([Path(x.path.items) for x in first_item(self)], fork_id=fork_id),
            )

        return EdgePath(
            sources=self.sources, path=self.path_builder.to(first_item, *extra_destinations, fork_id=fork_id)
        )

    def spread[T](
        self: EdgePathBuilder[StateT, DepsT, Iterable[T]], fork_id: str | None = None
    ) -> EdgePathBuilder[StateT, DepsT, Any]:
        return EdgePathBuilder(sources=self.sources, path_builder=self.path_builder.spread(fork_id=fork_id))

    def transform[NewOutputT](
        self, func: TransformFunction[StateT, DepsT, OutputT, NewOutputT], /
    ) -> EdgePathBuilder[StateT, DepsT, NewOutputT]:
        return EdgePathBuilder(sources=self.sources, path_builder=self.path_builder.transform(func))

    def label(self, label: str) -> EdgePathBuilder[StateT, DepsT, OutputT]:
        return EdgePathBuilder(sources=self.sources, path_builder=self.path_builder.label(label))
