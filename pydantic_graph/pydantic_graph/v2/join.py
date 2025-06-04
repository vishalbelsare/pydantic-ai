from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Any

from pydantic import TypeAdapter
from pydantic_core import to_json

from pydantic_graph.v2.id_types import ForkId, JoinId


class ReducerContext[StateT, DepsT, InputT]:
    """The main reason this is not a dataclass is that we need it to be covariant in its type parameters."""

    def __init__(self, state: StateT, deps: DepsT, inputs: InputT):
        self._state = state
        self._deps = deps
        self._inputs = inputs

    @property
    def state(self) -> StateT:
        return self._state

    @property
    def deps(self) -> DepsT:
        return self._deps

    @property
    def inputs(self) -> InputT:
        return self._inputs

    def cancel_other_requests(self) -> None:
        raise NotImplementedError

    def __repr__(self):
        return f'{self.__class__.__name__}(state={self.state}, deps={self.deps}, inputs={self.inputs})'


@dataclass(init=False)
class Reducer[StateT, DepsT, InputT, OutputT](ABC):
    def __init__(self, ctx: ReducerContext[StateT, DepsT, InputT]) -> None:
        self.reduce(ctx)

    def reduce(self, ctx: ReducerContext[StateT, DepsT, InputT]) -> None:
        """Reduce the input data into the instance state."""
        pass

    def finalize(self, ctx: ReducerContext[StateT, DepsT, None]) -> OutputT:
        """Finalize the reduction and return the output."""
        raise NotImplementedError('Finalize method must be implemented in subclasses.')


@dataclass(init=False)
class NullReducer(Reducer[object, object, object, None]):
    def finalize(self, ctx: ReducerContext[object, object, None]) -> None:
        return None


@dataclass(init=False)
class ListReducer[T](Reducer[object, object, T, list[T]]):
    items: list[T] = field(default_factory=list)

    def reduce(self, ctx: ReducerContext[object, object, T]) -> None:
        self.items.append(ctx.inputs)

    def finalize(self, ctx: ReducerContext[object, object, None]) -> list[T]:
        return self.items


@dataclass(init=False)
class DictReducer[K, V](Reducer[object, object, dict[K, V], dict[K, V]]):
    data: dict[K, V] = field(default_factory=dict[K, V])

    def reduce(self, ctx: ReducerContext[object, object, dict[K, V]]) -> None:
        self.data.update(ctx.inputs)

    def finalize(self, ctx: ReducerContext[object, object, None]) -> dict[K, V]:
        return self.data


class Join[StateT, DepsT, InputT, OutputT]:
    def __init__(
        self, id: JoinId, reducer_type: type[Reducer[StateT, DepsT, InputT, OutputT]], joins: ForkId | None = None
    ) -> None:
        self.id = id
        self._reducer_type = reducer_type
        self.joins = joins

        self._type_adapter: TypeAdapter[Any] = TypeAdapter(reducer_type)  # needs to be annotated this way for variance

    def _force_covariant(self, inputs: InputT) -> OutputT:
        raise NotImplementedError

    def create_reducer(self, ctx: ReducerContext[StateT, DepsT, InputT]) -> Reducer[StateT, DepsT, InputT, OutputT]:
        """Create a reducer instance using the provided context."""
        return self._reducer_type(ctx)

    def serialize_reducer(self, instance: Reducer[Any, Any, Any, Any]) -> bytes:
        return to_json(instance)

    def deserialize_reducer(self, serialized: bytes) -> Reducer[StateT, DepsT, InputT, OutputT]:
        return self._type_adapter.validate_json(serialized)
