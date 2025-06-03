from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, cast


class TransformContext[StateT, DepsT, InputT]:
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

    def __repr__(self):
        return f'{self.__class__.__name__}(state={self.state}, deps={self.deps}, inputs={self.inputs})'


class TransformFunction[StateT, DepsT, InputT, OutputT](Protocol):
    def __call__(self, ctx: TransformContext[StateT, DepsT, InputT]) -> OutputT:
        raise NotImplementedError


type AnyTransformFunction = TransformFunction[Any, Any, Any, Any]


def f(x: TransformFunction[Any, Any, list[int], Any]):
    pass


f(cast(TransformFunction[Any, Any, Sequence[int], Any], None))
