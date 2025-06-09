from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field

from pydantic_graph.v2.id_types import ForkId, JoinId


class ReducerContext[DepsT, InputT]:
    """This object holds the input argument to a reducer's `reduce` and `finalize` methods.

    It's important that we use a class rather than just providing the raw input value to the reducer methods as a way
    to backwards-compatibly add new fields, e.g., if we want to eventually make `state` available, or methods.

    While I initially implemented this with both `state` and `deps`, that complicated the implementation and it's hard
    to imagine cases where `state` is necessary/valuable. So I've dropped `state` for now, but we can certainly consider
    re-adding that to this class at some point.

    The only reason this is not a dataclass is that we need it to be covariant in each of its type parameters,
    and dataclasses are invariant in fields due to the ability to mutate the values of attributes.
    Once `ReadOnly` is available in `typing_extensions`, we should be able to convert this back to a dataclass safely.
    """

    def __init__(self, deps: DepsT, inputs: InputT):
        self._deps = deps
        self._inputs = inputs

    @property
    def deps(self) -> DepsT:
        return self._deps

    @property
    def inputs(self) -> InputT:
        return self._inputs

    def __repr__(self):
        return f'{self.__class__.__name__}(deps={self.deps}, inputs={self.inputs})'


@dataclass(init=False)
class Reducer[DepsT, InputT, OutputT](ABC):
    def __init__(self, ctx: ReducerContext[DepsT, InputT]) -> None:
        self.reduce(ctx)

    def reduce(self, ctx: ReducerContext[DepsT, InputT]) -> None:
        """Reduce the input data into the instance state."""
        pass

    def finalize(self, ctx: ReducerContext[DepsT, None]) -> OutputT:
        """Finalize the reduction and return the output."""
        raise NotImplementedError('Finalize method must be implemented in subclasses.')


@dataclass(init=False)
class NullReducer(Reducer[object, object, None]):
    def finalize(self, ctx: ReducerContext[object, object]) -> None:
        return None


@dataclass(init=False)
class ListReducer[T](Reducer[object, T, list[T]]):
    items: list[T] = field(default_factory=list)

    def reduce(self, ctx: ReducerContext[object, T]) -> None:
        self.items.append(ctx.inputs)

    def finalize(self, ctx: ReducerContext[object, None]) -> list[T]:
        return self.items


@dataclass(init=False)
class DictReducer[K, V](Reducer[object, dict[K, V], dict[K, V]]):
    data: dict[K, V] = field(default_factory=dict[K, V])

    def reduce(self, ctx: ReducerContext[object, dict[K, V]]) -> None:
        self.data.update(ctx.inputs)

    def finalize(self, ctx: ReducerContext[object, None]) -> dict[K, V]:
        return self.data


class Join[DepsT, InputT, OutputT]:
    def __init__(
        self, id: JoinId, reducer_type: type[Reducer[DepsT, InputT, OutputT]], joins: ForkId | None = None
    ) -> None:
        self.id = id
        self._reducer_type = reducer_type
        self.joins = joins

        # self._type_adapter: TypeAdapter[Any] = TypeAdapter(reducer_type)  # needs to be annotated this way for variance

    def create_reducer(self, ctx: ReducerContext[DepsT, InputT]) -> Reducer[DepsT, InputT, OutputT]:
        """Create a reducer instance using the provided context."""
        return self._reducer_type(ctx)

    # TODO: If we want the ability to snapshot graph-run state, we'll need to be able to serialize these things
    # def serialize_reducer(self, instance: Reducer[Any, Any, Any]) -> bytes:
    #     return to_json(instance)
    #
    # def deserialize_reducer(self, serialized: bytes) -> Reducer[InputT, OutputT]:
    #     return self._type_adapter.validate_json(serialized)

    def _force_covariant(self, inputs: InputT) -> OutputT:
        raise RuntimeError('This method should never be called, it is just defined for typing purposes.')
