from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field

from pydantic_graph.v2.id_types import ForkId, JoinId
from pydantic_graph.v2.step import StepContext


@dataclass(init=False)
class Reducer[StateT, InputT, OutputT](ABC):
    def __init__(self, ctx: StepContext[StateT, InputT]) -> None:
        self.reduce(ctx)

    def reduce(self, ctx: StepContext[StateT, InputT]) -> None:
        """Reduce the input data into the instance state."""
        pass

    def finalize(self, ctx: StepContext[StateT, None]) -> OutputT:
        """Finalize the reduction and return the output."""
        raise NotImplementedError('Finalize method must be implemented in subclasses.')


@dataclass(init=False)
class NullReducer(Reducer[object, object, None]):
    def finalize(self, ctx: StepContext[object, object]) -> None:
        return None


@dataclass(init=False)
class ListReducer[T](Reducer[object, T, list[T]]):
    items: list[T] = field(default_factory=list)

    def reduce(self, ctx: StepContext[object, T]) -> None:
        self.items.append(ctx.inputs)

    def finalize(self, ctx: StepContext[object, None]) -> list[T]:
        return self.items


@dataclass(init=False)
class DictReducer[K, V](Reducer[object, dict[K, V], dict[K, V]]):
    data: dict[K, V] = field(default_factory=dict[K, V])

    def reduce(self, ctx: StepContext[object, dict[K, V]]) -> None:
        self.data.update(ctx.inputs)

    def finalize(self, ctx: StepContext[object, None]) -> dict[K, V]:
        return self.data


class Join[StateT, InputT, OutputT]:
    def __init__(
        self, id: JoinId, reducer_type: type[Reducer[StateT, InputT, OutputT]], joins: ForkId | None = None
    ) -> None:
        self.id = id
        self._reducer_type = reducer_type
        self.joins = joins

        # self._type_adapter: TypeAdapter[Any] = TypeAdapter(reducer_type)  # needs to be annotated this way for variance

    def create_reducer(self, ctx: StepContext[StateT, InputT]) -> Reducer[StateT, InputT, OutputT]:
        """Create a reducer instance using the provided context."""
        return self._reducer_type(ctx)

    # TODO(P3): If we want the ability to snapshot graph-run state, we'll need a way to
    #  serialize/deserialize the associated reducers, something like this:
    # def serialize_reducer(self, instance: Reducer[Any, Any, Any]) -> bytes:
    #     return to_json(instance)
    #
    # def deserialize_reducer(self, serialized: bytes) -> Reducer[InputT, OutputT]:
    #     return self._type_adapter.validate_json(serialized)

    def _force_covariant(self, inputs: InputT) -> OutputT:
        raise RuntimeError('This method should never be called, it is just defined for typing purposes.')
